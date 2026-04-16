#!/usr/bin/env python3
"""
Image-Based Semantic Segmentation Web UI
==========================================
Upload an image, select one or more models, and get segmented output.
Supports side-by-side comparison of FCN, SegFormer, and DeepLabV3
on ADE20K (150 classes) and Cityscapes (19 classes).

Usage:
    python app.py [--port 5555] [--device cuda:0]
"""

import argparse
import base64
import io
import os
import sys
import time
import warnings

import cv2
import numpy as np
import torch
from PIL import Image

warnings.filterwarnings('ignore', message='.*mmcv-lite.*')
warnings.filterwarnings('ignore', message='.*MultiScaleDeformableAttention.*')

from flask import Flask, render_template_string, request, jsonify

from mmseg.apis import init_model, inference_model
from mmseg.utils import get_classes, get_palette

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MMSEG_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'mmsegmentation'))

MODELS = {
    'cityscapes': {
        'fcn': {
            'config':     os.path.join(SCRIPT_DIR, 'configs/fcn_r50-d8_cityscapes.py'),
            'checkpoint': os.path.join(SCRIPT_DIR, 'weights/fcn_r50-d8_512x1024_40k_cityscapes.pth'),
            'name': 'FCN-R50-D8',
        },
        'segformer': {
            'config':     os.path.join(SCRIPT_DIR, 'configs/segformer_mit-b1_cityscapes.py'),
            'checkpoint': os.path.join(SCRIPT_DIR, 'weights/segformer_mit-b1_8x1_1024x1024_160k_cityscapes_20211208_064213-655c7b3f.pth'),
            'name': 'SegFormer-MiT-B1',
        },
        'deeplabv3': {
            'config':     os.path.join(MMSEG_DIR, 'configs/deeplabv3/deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024.py'),
            'checkpoint': os.path.join(SCRIPT_DIR, 'weights/deeplabv3_r50-d8_512x1024_40k_cityscapes_20200605_022449-acadc2f8.pth'),
            'name': 'DeepLabV3-R50-D8',
        },
    },
    'ade20k': {
        'fcn': {
            'config':     os.path.join(MMSEG_DIR, 'configs/fcn/fcn_r50-d8_4xb4-80k_ade20k-512x512.py'),
            'checkpoint': os.path.join(SCRIPT_DIR, 'weights/fcn_r50-d8_512x512_80k_ade20k_20200614_144016-f8ac5082.pth'),
            'name': 'FCN-R50-D8',
        },
        'segformer': {
            'config':     os.path.join(MMSEG_DIR, 'configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py'),
            'checkpoint': os.path.join(SCRIPT_DIR, 'weights/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth'),
            'name': 'SegFormer-MiT-B1',
        },
        'deeplabv3': {
            'config':     os.path.join(MMSEG_DIR, 'configs/deeplabv3/deeplabv3_r50-d8_4xb4-80k_ade20k-512x512.py'),
            'checkpoint': os.path.join(SCRIPT_DIR, 'weights/deeplabv3_r50-d8_512x512_80k_ade20k_20200614_185028-0bb3f844.pth'),
            'name': 'DeepLabV3-R50-D8',
        },
    },
}

DATASET_INFO = {
    'ade20k':     {'classes': get_classes('ade20k'),     'palette': np.array(get_palette('ade20k'),     dtype=np.uint8)},
    'cityscapes': {'classes': get_classes('cityscapes'), 'palette': np.array(get_palette('cityscapes'), dtype=np.uint8)},
}

# ─── Model cache (keyed by "dataset|model_key") ──────────────────────────────
_model_cache: dict = {}
_current_device: str = 'cpu'


def get_model(dataset: str, model_key: str):
    key = f"{dataset}|{model_key}"
    if key not in _model_cache:
        info = MODELS[dataset][model_key]
        print(f"[info] Loading {info['name']} ({dataset}) on {_current_device}…", flush=True)
        t0 = time.time()
        _model_cache[key] = init_model(info['config'], info['checkpoint'], device=_current_device)
        print(f"[info] Loaded in {time.time()-t0:.1f}s", flush=True)
    return _model_cache[key]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def colorize(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(palette)):
        color[mask == c] = palette[c]
    return color


def top_classes(mask, k, palette, class_names):
    ids, counts = np.unique(mask, return_counts=True)
    total = mask.size
    keep = counts / total > 0.005
    ids, counts = ids[keep], counts[keep]
    order = np.argsort(-counts)[:k]
    res = []
    for i in order:
        cid = ids[i]
        color = palette[cid]
        res.append({
            'name':  class_names[cid],
            'pct':   round(float(counts[i] / total * 100), 1),
            'color': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
        })
    return res


def run_inference(img_pil: Image.Image, dataset: str, model_key: str, opacity: float) -> dict:
    model = get_model(dataset, model_key)
    frame = np.array(img_pil.convert('RGB'))
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    t0 = time.time()
    result = inference_model(model, frame_bgr)
    elapsed = time.time() - t0

    pred = result.pred_sem_seg.data.cpu().numpy().squeeze().astype(np.uint8)
    h, w = frame_bgr.shape[:2]
    if pred.shape != (h, w):
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

    palette     = DATASET_INFO[dataset]['palette']
    class_names = DATASET_INFO[dataset]['classes']

    seg_color = colorize(pred, palette)
    blended   = (frame * (1 - opacity) + seg_color * opacity).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format='JPEG', quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # pure mask (no blend)
    buf2 = io.BytesIO()
    Image.fromarray(seg_color).save(buf2, format='JPEG', quality=85)
    mask_b64 = base64.b64encode(buf2.getvalue()).decode()

    return {
        'image':    img_b64,
        'mask':     mask_b64,
        'classes':  top_classes(pred, 8, palette, class_names),
        'elapsed':  round(elapsed * 1000),
        'model':    MODELS[dataset][model_key]['name'],
        'dataset':  dataset,
    }


# ─── Flask ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SegLab — Semantic Segmentation</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg:       #080810;
  --surface:  #0e0e1a;
  --surface2: #141428;
  --border:   rgba(255,255,255,0.07);
  --accent:   #7c6dff;
  --accent2:  #00e5c3;
  --text:     #d8d8ee;
  --muted:    #5a5a7a;
  --danger:   #ff4f6d;
}

*, *::before, *::after { box-sizing: border-box; margin:0; padding:0; }

body {
  font-family: 'DM Sans', sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── scanline texture ── */
body::before {
  content:'';
  position:fixed; inset:0; pointer-events:none; z-index:999;
  background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px);
}

/* ── header ── */
header {
  display:flex; align-items:center; justify-content:space-between;
  padding:18px 32px;
  border-bottom:1px solid var(--border);
  background: linear-gradient(180deg, rgba(124,109,255,0.06) 0%, transparent 100%);
  position: sticky; top:0; z-index:100;
  backdrop-filter: blur(12px);
}

.logo {
  font-family: 'Syne', sans-serif;
  font-weight:800; font-size:20px;
  letter-spacing:-0.5px;
  display:flex; align-items:center; gap:10px;
}
.logo-mark {
  width:30px; height:30px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  border-radius:8px;
  display:flex; align-items:center; justify-content:center;
  font-size:14px;
}

.header-right {
  display:flex; align-items:center; gap:16px;
  font-family:'DM Mono',monospace; font-size:12px; color:var(--muted);
}

.device-badge {
  padding:4px 10px;
  background: rgba(0,229,195,0.08);
  border: 1px solid rgba(0,229,195,0.2);
  border-radius:20px;
  color: var(--accent2);
}

/* ── layout ── */
.layout {
  display:grid;
  grid-template-columns: 300px 1fr;
  gap:0;
  height: calc(100vh - 62px);
}

/* ── sidebar ── */
.sidebar {
  background: var(--surface);
  border-right:1px solid var(--border);
  padding:24px 20px;
  overflow-y:auto;
  display:flex; flex-direction:column; gap:24px;
}

.section-label {
  font-family:'DM Mono',monospace;
  font-size:10px; font-weight:500;
  text-transform:uppercase; letter-spacing:2px;
  color:var(--muted);
  margin-bottom:10px;
}

/* ── upload zone ── */
.upload-zone {
  border:2px dashed rgba(124,109,255,0.3);
  border-radius:12px;
  padding:28px 16px;
  text-align:center;
  cursor:pointer;
  transition: all .25s;
  background: rgba(124,109,255,0.03);
  position:relative;
}
.upload-zone:hover, .upload-zone.drag {
  border-color: var(--accent);
  background: rgba(124,109,255,0.08);
}
.upload-zone input { position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%; }
.upload-icon { font-size:28px; margin-bottom:8px; }
.upload-zone p { font-size:13px; color:var(--muted); line-height:1.5; }
.upload-zone p strong { color:var(--text); }

.preview-thumb {
  width:100%; border-radius:8px; margin-top:12px;
  max-height:160px; object-fit:cover;
  display:none;
  border:1px solid var(--border);
}

/* ── model selector ── */
.dataset-tabs {
  display:flex; gap:6px; margin-bottom:12px;
}
.tab {
  flex:1; padding:7px;
  background: var(--surface2);
  border:1px solid var(--border);
  border-radius:7px;
  font-size:12px; font-weight:500;
  color:var(--muted); cursor:pointer;
  text-align:center; transition:.2s;
  font-family:'DM Mono',monospace;
}
.tab.active { background:rgba(124,109,255,0.15); border-color:var(--accent); color:var(--accent); }

.model-grid {
  display:flex; flex-direction:column; gap:8px;
}

.model-card {
  padding:12px 14px;
  background: var(--surface2);
  border:1px solid var(--border);
  border-radius:10px;
  cursor:pointer;
  transition:.2s;
  display:flex; align-items:center; justify-content:space-between;
}
.model-card:hover { border-color: rgba(124,109,255,0.4); }
.model-card.selected {
  border-color: var(--accent);
  background: rgba(124,109,255,0.1);
}
.model-card.compare-selected {
  border-color: var(--accent2);
  background: rgba(0,229,195,0.07);
}
.model-name { font-size:13px; font-weight:500; }
.model-desc { font-size:11px; color:var(--muted); margin-top:2px; font-family:'DM Mono',monospace; }
.model-check { width:18px;height:18px; border-radius:4px; border:1px solid var(--border); display:flex; align-items:center; justify-content:center; font-size:10px; }
.model-card.selected .model-check { background:var(--accent); border-color:var(--accent); color:#fff; }
.model-card.compare-selected .model-check { background:var(--accent2); border-color:var(--accent2); color:#000; }

/* ── compare toggle ── */
.compare-toggle {
  display:flex; align-items:center; gap:10px;
  padding:10px 14px;
  background: var(--surface2);
  border:1px solid var(--border);
  border-radius:10px;
  cursor:pointer;
}
.compare-toggle input { accent-color:var(--accent2); width:16px;height:16px; cursor:pointer; }
.compare-toggle label { font-size:13px; cursor:pointer; flex:1; }
.compare-badge {
  font-family:'DM Mono',monospace; font-size:10px;
  padding:2px 8px; border-radius:10px;
  background:rgba(0,229,195,0.1); color:var(--accent2);
}

/* ── opacity ── */
.slider-row { display:flex; align-items:center; gap:10px; }
.slider-row label { font-size:12px; color:var(--muted); white-space:nowrap; }
.slider-row input[type=range] { flex:1; accent-color:var(--accent); }
.slider-val { font-family:'DM Mono',monospace; font-size:12px; color:var(--accent); min-width:32px; text-align:right; }

/* ── run button ── */
.run-btn {
  width:100%; padding:14px;
  background: linear-gradient(135deg, var(--accent), #9d6fff);
  border:none; border-radius:10px;
  font-family:'Syne',sans-serif; font-weight:700; font-size:15px;
  color:#fff; cursor:pointer;
  transition: all .2s;
  position:relative; overflow:hidden;
}
.run-btn::after {
  content:'';
  position:absolute; inset:0;
  background: linear-gradient(135deg, transparent 40%, rgba(255,255,255,0.1));
  opacity:0; transition:.2s;
}
.run-btn:hover::after { opacity:1; }
.run-btn:hover { transform:translateY(-2px); box-shadow:0 8px 24px rgba(124,109,255,0.4); }
.run-btn:disabled { opacity:.4; cursor:not-allowed; transform:none; box-shadow:none; }
.run-btn.loading {
  background: var(--surface2);
  color:var(--muted);
}

/* ── main area ── */
.main {
  display:flex; flex-direction:column;
  overflow:hidden;
}

/* ── results ── */
.results-area {
  flex:1; padding:24px;
  overflow-y:auto;
  display:flex; flex-direction:column; gap:20px;
}

.empty-state {
  flex:1; display:flex; flex-direction:column;
  align-items:center; justify-content:center;
  gap:14px; color:var(--muted);
}
.empty-glyph {
  font-size:56px; opacity:.3;
  filter: grayscale(1);
}
.empty-state p { font-size:14px; line-height:1.6; text-align:center; max-width:300px; }

/* ── single result ── */
.result-block {
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:14px;
  overflow:hidden;
  animation: fadeUp .35s ease both;
}
@keyframes fadeUp {
  from { opacity:0; transform:translateY(12px); }
  to   { opacity:1; transform:translateY(0); }
}

.result-header {
  padding:14px 18px;
  display:flex; align-items:center; justify-content:space-between;
  border-bottom:1px solid var(--border);
  background: linear-gradient(90deg, rgba(124,109,255,0.05), transparent);
}
.result-title {
  font-family:'Syne',sans-serif; font-weight:700; font-size:15px;
}
.result-meta {
  display:flex; gap:10px; align-items:center;
}
.meta-chip {
  font-family:'DM Mono',monospace; font-size:11px;
  padding:3px 10px; border-radius:10px;
  background:rgba(255,255,255,0.05);
  border:1px solid var(--border);
  color:var(--muted);
}
.meta-chip.time { color:var(--accent2); border-color:rgba(0,229,195,0.2); background:rgba(0,229,195,0.05); }

.result-images {
  display:flex;
}
.img-panel {
  flex:1; position:relative;
  cursor: col-resize;
}
.img-panel + .img-panel {
  border-left:1px solid var(--border);
}
.img-panel img {
  width:100%; display:block;
  max-height:400px; object-fit:contain;
  background:#050508;
}
.img-label {
  position:absolute; top:10px; left:10px;
  font-family:'DM Mono',monospace; font-size:10px;
  padding:3px 9px; border-radius:6px;
  background:rgba(0,0,0,0.7); backdrop-filter:blur(4px);
  border:1px solid var(--border); color:#aaa;
  text-transform:uppercase; letter-spacing:1px;
}

/* ── view toggle ── */
.view-tabs {
  display:flex; gap:1px;
  padding:10px 18px;
  border-bottom:1px solid var(--border);
}
.vtab {
  padding:5px 14px; border-radius:6px;
  font-size:12px; cursor:pointer;
  color:var(--muted); transition:.15s;
  font-family:'DM Mono',monospace;
}
.vtab.active { background:rgba(124,109,255,0.15); color:var(--accent); }

/* ── class legend ── */
.class-legend {
  padding:14px 18px;
  border-top:1px solid var(--border);
  display:flex; flex-wrap:wrap; gap:8px;
}
.cls-chip {
  display:flex; align-items:center; gap:6px;
  padding:4px 10px; border-radius:20px;
  background:rgba(255,255,255,0.04);
  border:1px solid var(--border);
  font-size:12px;
  transition:.15s;
}
.cls-chip:hover { background:rgba(255,255,255,0.07); }
.cls-dot { width:10px;height:10px; border-radius:50%; flex-shrink:0; }
.cls-name { color:var(--text); }
.cls-pct { color:var(--muted); font-family:'DM Mono',monospace; font-size:11px; }

/* ── compare result ── */
.compare-grid {
  display:grid; grid-template-columns:1fr 1fr; gap:16px;
}

/* ── progress ── */
.progress-bar {
  height:3px; width:100%;
  background:var(--surface2);
  position:relative; overflow:hidden;
  display:none;
}
.progress-bar.active { display:block; }
.progress-bar::after {
  content:'';
  position:absolute; top:0; left:-60%;
  width:60%; height:100%;
  background: linear-gradient(90deg, transparent, var(--accent), var(--accent2), transparent);
  animation: slide 1.2s linear infinite;
}
@keyframes slide { to { left:110%; } }

/* ── toast ── */
.toast {
  position:fixed; bottom:24px; right:24px;
  background:var(--surface); border:1px solid var(--border);
  border-radius:10px; padding:12px 18px;
  font-size:13px; z-index:1000;
  transform:translateY(60px); opacity:0;
  transition:.3s; max-width:320px;
}
.toast.show { transform:translateY(0); opacity:1; }
.toast.error { border-color:rgba(255,79,109,0.4); color:var(--danger); }
.toast.success { border-color:rgba(0,229,195,0.3); color:var(--accent2); }

/* scrollbar */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.08); border-radius:3px; }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-mark">⬡</div>
    SegLab
  </div>
  <div class="header-right">
    <span>Semantic Segmentation Explorer</span>
    <span class="device-badge" id="deviceBadge">—</span>
  </div>
</header>

<div class="layout">

  <!-- ── Sidebar ── -->
  <aside class="sidebar">

    <!-- Upload -->
    <div>
      <div class="section-label">Input Image</div>
      <div class="upload-zone" id="dropZone">
        <input type="file" id="fileInput" accept="image/*">
        <div class="upload-icon">🖼</div>
        <p><strong>Drop image here</strong><br>or click to browse</p>
        <img id="previewThumb" class="preview-thumb" alt="preview">
      </div>
    </div>

    <!-- Dataset -->
    <div>
      <div class="section-label">Dataset</div>
      <div class="dataset-tabs">
        <div class="tab active" data-ds="ade20k" onclick="setDataset('ade20k')">ADE20K<br><span style="font-size:10px;opacity:.6">150 cls</span></div>
        <div class="tab" data-ds="cityscapes" onclick="setDataset('cityscapes')">Cityscapes<br><span style="font-size:10px;opacity:.6">19 cls</span></div>
      </div>
    </div>

    <!-- Models -->
    <div>
      <div class="section-label">Model</div>
      <div class="model-grid" id="modelGrid">
        <div class="model-card selected" data-model="segformer" onclick="selectModel(this)">
          <div>
            <div class="model-name">SegFormer</div>
            <div class="model-desc">MiT-B1 · Transformer</div>
          </div>
          <div class="model-check">✓</div>
        </div>
        <div class="model-card" data-model="fcn" onclick="selectModel(this)">
          <div>
            <div class="model-name">FCN</div>
            <div class="model-desc">R50-D8 · Fully Conv</div>
          </div>
          <div class="model-check"></div>
        </div>
        <div class="model-card" data-model="deeplabv3" onclick="selectModel(this)">
          <div>
            <div class="model-name">DeepLabV3</div>
            <div class="model-desc">R50-D8 · ASPP</div>
          </div>
          <div class="model-check"></div>
        </div>
      </div>
    </div>

    <!-- Compare mode -->
    <div>
      <div class="section-label">Comparison</div>
      <div class="compare-toggle">
        <input type="checkbox" id="compareToggle" onchange="toggleCompare()">
        <label for="compareToggle">Compare two models</label>
        <span class="compare-badge">BETA</span>
      </div>
      <div id="compareModelGrid" class="model-grid" style="margin-top:10px;display:none">
        <div class="model-card compare-selected" data-model2="fcn" onclick="selectModel2(this)">
          <div>
            <div class="model-name">FCN</div>
            <div class="model-desc">R50-D8 · Fully Conv</div>
          </div>
          <div class="model-check">✓</div>
        </div>
        <div class="model-card" data-model2="segformer" onclick="selectModel2(this)">
          <div>
            <div class="model-name">SegFormer</div>
            <div class="model-desc">MiT-B1 · Transformer</div>
          </div>
          <div class="model-check"></div>
        </div>
        <div class="model-card" data-model2="deeplabv3" onclick="selectModel2(this)">
          <div>
            <div class="model-name">DeepLabV3</div>
            <div class="model-desc">R50-D8 · ASPP</div>
          </div>
          <div class="model-check"></div>
        </div>
      </div>
    </div>

    <!-- Opacity -->
    <div>
      <div class="section-label">Overlay</div>
      <div class="slider-row">
        <label>Opacity</label>
        <input type="range" id="opacitySlider" min="0" max="100" value="55" oninput="updateOpacity()">
        <span class="slider-val" id="opacityVal">55%</span>
      </div>
    </div>

    <!-- Run -->
    <button class="run-btn" id="runBtn" onclick="runSegmentation()">
      ▶ Run Segmentation
    </button>

  </aside>

  <!-- ── Main ── -->
  <div class="main">
    <div class="progress-bar" id="progressBar"></div>

    <div class="results-area" id="resultsArea">
      <div class="empty-state" id="emptyState">
        <div class="empty-glyph">◈</div>
        <p>Upload an image and choose a model to begin semantic segmentation.</p>
      </div>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let selectedDataset  = 'ade20k';
let selectedModel    = 'segformer';
let selectedModel2   = 'fcn';
let compareMode      = false;
let uploadedImageB64 = null;
let uploadedImageEl  = null;  // original image element for display

// ── Init ───────────────────────────────────────────────────────────────────
fetch('/api/info').then(r=>r.json()).then(d => {
  document.getElementById('deviceBadge').textContent = d.device;
});

// ── Upload ─────────────────────────────────────────────────────────────────
const dropZone  = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const thumb     = document.getElementById('previewThumb');

fileInput.addEventListener('change', e => handleFile(e.target.files[0]));

dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag');
  handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) return showToast('Please upload an image file.', 'error');
  const reader = new FileReader();
  reader.onload = e => {
    uploadedImageB64 = e.target.result.split(',')[1];
    thumb.src = e.target.result;
    thumb.style.display = 'block';
    dropZone.querySelector('p').innerHTML = '<strong>' + file.name + '</strong><br>Ready to segment';
  };
  reader.readAsDataURL(file);
}

// ── Dataset / Model selection ──────────────────────────────────────────────
function setDataset(ds) {
  selectedDataset = ds;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.ds === ds));
}

function selectModel(card) {
  document.querySelectorAll('#modelGrid .model-card').forEach(c => {
    c.classList.remove('selected');
    c.querySelector('.model-check').textContent = '';
  });
  card.classList.add('selected');
  card.querySelector('.model-check').textContent = '✓';
  selectedModel = card.dataset.model;
}

function selectModel2(card) {
  document.querySelectorAll('#compareModelGrid .model-card').forEach(c => {
    c.classList.remove('compare-selected');
    c.querySelector('.model-check').textContent = '';
  });
  card.classList.add('compare-selected');
  card.querySelector('.model-check').textContent = '✓';
  selectedModel2 = card.dataset.model2;
}

function toggleCompare() {
  compareMode = document.getElementById('compareToggle').checked;
  document.getElementById('compareModelGrid').style.display = compareMode ? 'flex' : 'none';
  document.getElementById('compareModelGrid').style.flexDirection = 'column';
}

function updateOpacity() {
  const v = document.getElementById('opacitySlider').value;
  document.getElementById('opacityVal').textContent = v + '%';
}

// ── Run ────────────────────────────────────────────────────────────────────
async function runSegmentation() {
  if (!uploadedImageB64) return showToast('Please upload an image first.', 'error');
  
  const btn = document.getElementById('runBtn');
  const bar = document.getElementById('progressBar');
  const opacity = document.getElementById('opacitySlider').value / 100;

  btn.disabled = true;
  btn.classList.add('loading');
  btn.textContent = compareMode ? 'Running comparison…' : 'Running inference…';
  bar.classList.add('active');

  try {
    const emptyState = document.getElementById('emptyState');
    if (emptyState) emptyState.remove();

    if (compareMode) {
      const [r1, r2] = await Promise.all([
        callInfer(selectedDataset, selectedModel,  opacity),
        callInfer(selectedDataset, selectedModel2, opacity),
      ]);
      renderCompare(r1, r2);
    } else {
      const r = await callInfer(selectedDataset, selectedModel, opacity);
      renderSingle(r);
    }
  } catch(e) {
    showToast('Error: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.classList.remove('loading');
    btn.textContent = '▶ Run Segmentation';
    bar.classList.remove('active');
  }
}

async function callInfer(dataset, modelKey, opacity) {
  const resp = await fetch('/api/segment', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ image: uploadedImageB64, dataset, model: modelKey, opacity }),
  });
  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.error || resp.statusText);
  }
  return resp.json();
}

// ── Render single ──────────────────────────────────────────────────────────
function renderSingle(data) {
  const area = document.getElementById('resultsArea');
  const block = document.createElement('div');
  block.className = 'result-block';
  
  const dsLabel = data.dataset === 'ade20k' ? 'ADE20K · 150 cls' : 'Cityscapes · 19 cls';
  
  block.innerHTML = `
    <div class="result-header">
      <div class="result-title">${data.model}</div>
      <div class="result-meta">
        <span class="meta-chip">${dsLabel}</span>
        <span class="meta-chip time">${data.elapsed} ms</span>
      </div>
    </div>
    <div class="view-tabs">
      <span class="vtab active" onclick="setView(this,'blended','${block.id||'b'+Date.now()}')">Overlay</span>
      <span class="vtab" onclick="setView(this,'mask','${block.id||'b'+Date.now()}')">Mask only</span>
      <span class="vtab" onclick="setView(this,'split','${block.id||'b'+Date.now()}')">Side by side</span>
    </div>
    <div class="result-images" id="imgArea_${Date.now()}">
      ${imgPanel(data.image, 'overlay')}
    </div>
    <div class="class-legend">${classChips(data.classes)}</div>
  `;
  
  // Store data reference
  block._data = data;
  
  // Setup view tabs
  const tabs  = block.querySelectorAll('.vtab');
  const imgArea = block.querySelector('.result-images');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t=>t.classList.remove('active'));
      tab.classList.add('active');
      const view = tab.textContent.toLowerCase();
      if (view === 'overlay') {
        imgArea.innerHTML = imgPanel(data.image, 'overlay');
      } else if (view === 'mask only') {
        imgArea.innerHTML = imgPanel(data.mask, 'mask');
      } else {
        imgArea.innerHTML = imgPanel('data:image/jpeg;base64,' + uploadedImageB64, 'original', true) +
                            imgPanel(data.image, 'overlay');
      }
    });
  });
  
  area.prepend(block);
  showToast('Done in ' + data.elapsed + ' ms', 'success');
}

// ── Render compare ─────────────────────────────────────────────────────────
function renderCompare(d1, d2) {
  const area  = document.getElementById('resultsArea');
  const block = document.createElement('div');
  block.className = 'result-block';
  
  const dsLabel = d1.dataset === 'ade20k' ? 'ADE20K · 150 cls' : 'Cityscapes · 19 cls';
  
  block.innerHTML = `
    <div class="result-header">
      <div class="result-title">Model Comparison</div>
      <div class="result-meta">
        <span class="meta-chip">${dsLabel}</span>
        <span class="meta-chip time">${Math.max(d1.elapsed,d2.elapsed)} ms</span>
      </div>
    </div>
    <div class="result-images">
      ${imgPanel(d1.image, d1.model, false, 'var(--accent)')}
      ${imgPanel(d2.image, d2.model, false, 'var(--accent2)')}
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0;border-top:1px solid var(--border)">
      <div class="class-legend" style="border-right:1px solid var(--border)">${classChips(d1.classes)}</div>
      <div class="class-legend">${classChips(d2.classes)}</div>
    </div>
  `;
  
  area.prepend(block);
  showToast('Comparison complete', 'success');
}

// ── Helpers ────────────────────────────────────────────────────────────────
function imgPanel(src, label, raw=false, accentColor=null) {
  const imgSrc = raw ? src : 'data:image/jpeg;base64,' + src;
  const accent = accentColor ? `border-top: 2px solid ${accentColor};` : '';
  return `<div class="img-panel" style="${accent}">
    <img src="${imgSrc}" alt="${label}">
    <span class="img-label">${label}</span>
  </div>`;
}

function classChips(classes) {
  return classes.map(c =>
    `<div class="cls-chip">
      <div class="cls-dot" style="background:${c.color}"></div>
      <span class="cls-name">${c.name}</span>
      <span class="cls-pct">${c.pct}%</span>
    </div>`
  ).join('');
}

// ── Toast ──────────────────────────────────────────────────────────────────
let toastTimer;
function showToast(msg, type='') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast show ' + type;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 3000);
}
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/api/info')
def api_info():
    return jsonify({'device': _current_device})


@app.route('/api/segment', methods=['POST'])
def api_segment():
    data     = request.get_json(force=True)
    dataset  = data.get('dataset', 'ade20k')
    model_k  = data.get('model', 'segformer')
    opacity  = float(data.get('opacity', 0.55))
    img_b64  = data.get('image', '')

    if not img_b64:
        return jsonify({'error': 'No image provided'}), 400
    if dataset not in MODELS:
        return jsonify({'error': f'Unknown dataset: {dataset}'}), 400
    if model_k not in MODELS[dataset]:
        return jsonify({'error': f'Unknown model: {model_k}'}), 400

    try:
        img_bytes = base64.b64decode(img_b64)
        img_pil   = Image.open(io.BytesIO(img_bytes))
        result    = run_inference(img_pil, dataset, model_k, opacity)
        return jsonify(result)
    except Exception as e:
        print(f'[error] {e}', flush=True)
        return jsonify({'error': str(e)}), 500


# ─── Entry ────────────────────────────────────────────────────────────────────
def main():
    global _current_device

    parser = argparse.ArgumentParser()
    parser.add_argument('--port',   type=int, default=5555)
    parser.add_argument('--host',   type=str, default='0.0.0.0')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    _current_device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[info] Using device: {_current_device}', flush=True)

    print(f'\n  → SegLab running at http://localhost:{args.port}\n', flush=True)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
