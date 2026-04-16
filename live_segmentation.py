#!/usr/bin/env python3
"""
Live Web-Based Camera Segmentation
===================================
Flask + SocketIO app that captures your local webcam in-browser,
sends frames to the server for inference on GPU,
and streams the segmented overlay back in real time.
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

from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit

from models import build_fcn, build_deeplabv3, build_segformer, inference_model
from mmseg.utils import get_classes, get_palette

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MMSEG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'mmsegmentation'))

MODELS = {
    'cityscapes': {
        'fcn': {
            'config': os.path.join(SCRIPT_DIR, 'configs/fcn_r50-d8_cityscapes.py'),
            'checkpoint': os.path.join(SCRIPT_DIR, 'weights/fcn_r50-d8_512x1024_40k_cityscapes.pth'),
            'name': 'FCN-R50-D8',
        },
        'segformer': {
            'config': os.path.join(SCRIPT_DIR, 'configs/segformer_mit-b1_cityscapes.py'),
            'checkpoint': os.path.join(SCRIPT_DIR, 'weights/segformer_mit-b1_8x1_1024x1024_160k_cityscapes_20211208_064213-655c7b3f.pth'),
            'name': 'SegFormer-MiT-B1',
        },
        'deeplabv3': {
            'config': os.path.join(MMSEG_DIR, 'configs/deeplabv3/deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024.py'),
            'checkpoint': os.path.join(SCRIPT_DIR, 'weights/deeplabv3_r50-d8_512x1024_40k_cityscapes_20200605_022449-acadc2f8.pth'),
            'name': 'DeepLabV3-R50-D8'
        }
    },
    'ade20k': {
        'fcn': {
            'config': os.path.join(MMSEG_DIR, 'configs/fcn/fcn_r50-d8_4xb4-80k_ade20k-512x512.py'),
            'checkpoint': os.path.join(SCRIPT_DIR, 'weights/fcn_r50-d8_512x512_80k_ade20k_20200614_144016-f8ac5082.pth'),
            'name': 'FCN-R50-D8'
        },
        'segformer': {
            'config': os.path.join(MMSEG_DIR, 'configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py'),
            'checkpoint': os.path.join(SCRIPT_DIR, 'weights/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth'),
            'name': 'SegFormer-MiT-B1'
        },
        'deeplabv3': {
            'config': os.path.join(MMSEG_DIR, 'configs/deeplabv3/deeplabv3_r50-d8_4xb4-80k_ade20k-512x512.py'),
            'checkpoint': os.path.join(SCRIPT_DIR, 'weights/deeplabv3_r50-d8_512x512_80k_ade20k_20200614_185028-0bb3f844.pth'),
            'name': 'DeepLabV3-R50-D8'
        }
    }
}

DATASET_INFO = {
    'ade20k': {
        'classes': get_classes('ade20k'),
        'palette': np.array(get_palette('ade20k'), dtype=np.uint8)
    },
    'cityscapes': {
        'classes': get_classes('cityscapes'),
        'palette': np.array(get_palette('cityscapes'), dtype=np.uint8)
    }
}

# ─── Flask app ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'semseg-live'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading',
                    max_http_buffer_size=16 * 1024 * 1024)

# Global state
model = None
current_device = 'cpu'
current_dataset = 'ade20k'


# ─── Helpers ─────────────────────────────────────────────────────────────────

def colorize(mask, palette):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    num_classes = len(palette)
    for c in range(num_classes):
        color[mask == c] = palette[c]
    return color


def top_classes(mask, k, palette, class_names):
    ids, counts = np.unique(mask, return_counts=True)
    total = mask.size
    keep = counts / total > 0.01
    ids, counts = ids[keep], counts[keep]
    order = np.argsort(-counts)[:k]
    
    res = []
    for i in order:
        cls_id = ids[i]
        color = palette[cls_id]
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        res.append({
            'name': class_names[cls_id],
            'pct': round(float(counts[i] / total * 100), 1),
            'color': hex_color
        })
    return res


def load_selected_model(dataset, model_key):
    global model, current_dataset
    info = MODELS[dataset][model_key]
    print(f"Loading {info['name']} on {current_device}...", flush=True)
    num_classes = len(DATASET_INFO[dataset]['classes'])
    if model_key == 'fcn':
        model = build_fcn(num_classes, info['checkpoint'], device=current_device)
    elif model_key == 'segformer':
        model = build_segformer(num_classes, info['checkpoint'], device=current_device)
    elif model_key == 'deeplabv3':
        model = build_deeplabv3(num_classes, info['checkpoint'], device=current_device)
    current_dataset = dataset
    print(f"Model {info['name']} loaded.", flush=True)


# ─── Routes ──────────────────────────────────────────────────────────────────

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Live Segmentation — MMSegmentation</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Inter', sans-serif;
    background: #0a0a0f;
    color: #e0e0e0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  header {
    width: 100%;
    padding: 16px 32px;
    background: linear-gradient(135deg, #12121a 0%, #1a1a2e 100%);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  header h1 {
    font-size: 18px;
    font-weight: 600;
    background: linear-gradient(90deg, #6ee7b7, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .status {
    font-size: 13px;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 500;
  }
  .status.connected    { background: rgba(34,197,94,0.15); color: #4ade80; }
  .status.disconnected { background: rgba(239,68,68,0.15); color: #f87171; }
  .status.loading      { background: rgba(250,204,21,0.15); color: #facc15; }

  .main {
    flex: 1;
    display: flex;
    gap: 20px;
    padding: 24px;
    max-width: 1200px;
    width: 100%;
  }

  .feed-container {
    flex: 1;
    position: relative;
    background: #111118;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
  }

  .feed-container canvas,
  .feed-container video {
    width: 100%;
    display: block;
    border-radius: 12px;
  }
  
  video#webcam { 
      display: none; 
  }

  .fps-badge {
    position: absolute;
    top: 12px;
    right: 12px;
    background: rgba(0,0,0,0.65);
    backdrop-filter: blur(6px);
    padding: 5px 12px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    color: #4ade80;
    font-variant-numeric: tabular-nums;
  }

  .sidebar {
    width: 260px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .card {
    background: #111118;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 16px;
  }

  .card h3 {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #888;
    margin-bottom: 12px;
  }

  .class-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
  }

  .class-item .dot {
    width: 14px; height: 14px;
    border-radius: 50%;
    flex-shrink: 0;
    box-shadow: 0px 0px 4px rgba(0,0,0,0.5);
  }

  .class-item .name {
    flex: 1;
    font-size: 13px;
    font-weight: 500;
  }

  .class-item .pct {
    font-size: 13px;
    font-variant-numeric: tabular-nums;
    color: #aaa;
  }

  .bar-bg {
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    margin-top: 3px;
    width: 100%;
  }

  .bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s ease;
  }

  .controls {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .controls label {
    font-size: 12px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  
  .controls select {
    width: 100%;
    padding: 8px;
    background: #1a1a2e;
    color: white;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 6px;
    font-family: inherit;
    font-size: 13px;
    cursor: pointer;
    font-weight: 500;
  }

  .controls input[type=range] {
    width: 100%;
    accent-color: #3b82f6;
  }

  .btn {
    width: 100%;
    padding: 10px;
    border: none;
    border-radius: 8px;
    font-family: inherit;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn-start {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: #fff;
  }

  .btn-stop {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: #fff;
  }

  .btn:hover { opacity: 0.85; transform: translateY(-1px); }

  .empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 400px;
    color: #555;
    font-size: 15px;
  }
</style>
</head>
<body>

<header>
  <h1>⬡ Live Segmentation Dashboard</h1>
  <span id="status" class="status disconnected">Disconnected</span>
</header>

<div class="main">
  <div class="feed-container">
    <video id="webcam" autoplay playsinline></video>
    <canvas id="output"></canvas>
    <div class="fps-badge" id="fps">— FPS</div>
    <div class="empty-state" id="empty">Click ▶ Start Feed to begin</div>
  </div>

  <div class="sidebar">
    <div class="card controls">
      <h3>Model Selection</h3>
      <select id="modelSelect" onchange="switchModel(this.value)">
          <optgroup label="ADE20K (150 Classes)">
              <option value="ade20k|segformer" selected>SegFormer ADE20K</option>
              <option value="ade20k|fcn">FCN ADE20K</option>
              <option value="ade20k|deeplabv3">DeepLabV3 ADE20K</option>
          </optgroup>
          <optgroup label="Cityscapes (19 Classes)">
              <option value="cityscapes|segformer">SegFormer Cityscapes</option>
              <option value="cityscapes|fcn">FCN Cityscapes</option>
              <option value="cityscapes|deeplabv3">DeepLabV3 Cityscapes</option>
          </optgroup>
      </select>
      
      <h3 style="margin-top: 10px;">Controls</h3>
      <button class="btn btn-start" id="startBtn" onclick="startStream()">▶ Start Feed</button>
      <button class="btn btn-stop"  id="stopBtn"  onclick="stopStream()" style="display:none">■ Stop Feed</button>
      <label style="margin-top: 5px;">Overlay opacity</label>
      <input type="range" id="opacity" min="0" max="100" value="50">
    </div>

    <div class="card">
      <h3>Detected Classes</h3>
      <div id="classList"><span style="color:#555;font-size:13px">Waiting for data…</span></div>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.5/socket.io.min.js"></script>
<script>
const video    = document.getElementById('webcam');
const canvas   = document.getElementById('output');
const ctx      = canvas.getContext('2d');
const fpsEl    = document.getElementById('fps');
const statusEl = document.getElementById('status');
const classEl  = document.getElementById('classList');
const emptyEl  = document.getElementById('empty');
const startBtn = document.getElementById('startBtn');
const stopBtn  = document.getElementById('stopBtn');
const opSlider = document.getElementById('opacity');
const modelSel = document.getElementById('modelSelect');

let socket = null;
let streaming = false;
let sendCanvas = document.createElement('canvas');
let sendCtx = sendCanvas.getContext('2d');
let lastSendTime = 0;
let waitingForResponse = false;
let modelSwitching = false;

// FPS tracking
let frameCount = 0;
let fpsTimer = performance.now();

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = 'status ' + cls;
}

function switchModel(val) {
    if(!socket) connectSocket();
    const parts = val.split('|');
    modelSwitching = true;
    setStatus('Loading model...', 'loading');
    modelSel.disabled = true;
    socket.emit('switch_model', { dataset: parts[0], model: parts[1] });
}

function connectSocket() {
  if (socket) return;
  socket = io();

  socket.on('connect', () => {
      if(!modelSwitching) {
          setStatus(streaming ? 'Streaming' : 'Connected', 'connected');
      }
  });
  
  socket.on('disconnect', () => setStatus('Disconnected', 'disconnected'));

  socket.on('model_switched', () => {
     modelSwitching = false;
     modelSel.disabled = false;
     setStatus(streaming ? 'Streaming' : 'Connected', 'connected');
  });

  socket.on('result', (data) => {
    waitingForResponse = false;
    
    // Draw the returned overlay
    const img = new window.Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      emptyEl.style.display = 'none';
      frameCount++;
    };
    img.src = 'data:image/jpeg;base64,' + data.image;

    // Update class list
    if (data.classes) {
      classEl.innerHTML = data.classes.map(c => {
        return `<div class="class-item">
          <div class="dot" style="background:${c.color}"></div>
          <span class="name">${c.name}</span>
          <span class="pct">${c.pct}%</span>
        </div>
        <div class="bar-bg"><div class="bar-fill" style="width:${c.pct}%;background:${c.color}"></div></div>`;
      }).join('');
    }
  });
}

async function startStream() {
  connectSocket();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } }
    });
    video.srcObject = stream;
    await video.play();

    sendCanvas.width = 640;
    sendCanvas.height = 480;

    streaming = true;
    startBtn.style.display = 'none';
    stopBtn.style.display = 'block';
    emptyEl.style.display = 'none';
    if (!modelSwitching) setStatus('Streaming', 'connected');

    // FPS counter
    setInterval(() => {
      const now = performance.now();
      const elapsed = (now - fpsTimer) / 1000;
      const fps = frameCount / elapsed;
      fpsEl.textContent = fps.toFixed(1) + ' FPS';
      frameCount = 0;
      fpsTimer = now;
    }, 1000);

    sendFrame();
  } catch (e) {
    alert('Camera error: ' + e.message);
  }
}

function stopStream() {
  streaming = false;
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }
  startBtn.style.display = 'block';
  stopBtn.style.display = 'none';
  if (!modelSwitching) setStatus('Stopped', 'disconnected');
}

function sendFrame() {
  if (!streaming) return;

  if (!waitingForResponse && !modelSwitching) {
    // Send un-mirrored frame because we flip it on the backend
    sendCtx.drawImage(video, 0, 0, 640, 480);
    const dataUrl = sendCanvas.toDataURL('image/jpeg', 0.8);
    const b64 = dataUrl.split(',')[1];

    waitingForResponse = true;
    socket.emit('frame', {
      image: b64,
      opacity: opSlider.value / 100
    });
  }

  requestAnimationFrame(sendFrame);
}
</script>

</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@socketio.on('switch_model')
def handle_switch(data):
    try:
        ds = data['dataset']
        mod = data['model']
        load_selected_model(ds, mod)
        emit('model_switched')
    except Exception as e:
        print(f"Model switch error: {e}", flush=True)

@socketio.on('frame')
def handle_frame(data):
    global model, current_dataset
    if model is None:
        return
        
    try:
        # Decode incoming JPEG
        img_bytes = base64.b64decode(data['image'])
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        frame = np.array(img_pil)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Invert the video horizontally (creates mirror effect)
        frame_bgr = cv2.flip(frame_bgr, 1)

        opacity = float(data.get('opacity', 0.5))

        # Run inference
        pred = inference_model(model, frame_bgr)

        h, w = frame_bgr.shape[:2]
        if pred.shape != (h, w):
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

        palette = DATASET_INFO[current_dataset]['palette']
        class_names = DATASET_INFO[current_dataset]['classes']

        # Build overlay
        seg_color = colorize(pred, palette)  # RGB
        
        # We need the base frame in RGB for final blending and sending to client
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        blended = (frame_rgb * (1 - opacity) + seg_color * opacity).astype(np.uint8)

        # Encode result as JPEG
        blended_pil = Image.fromarray(blended)
        buf = io.BytesIO()
        blended_pil.save(buf, format='JPEG', quality=80)
        result_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Top classes
        classes = top_classes(pred, 6, palette, class_names)

        emit('result', {'image': result_b64, 'classes': classes})

    except Exception as e:
        print(f'Frame error: {e}', flush=True)


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    global current_device

    parser = argparse.ArgumentParser(description='Live web segmentation')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    current_device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    load_selected_model('ade20k', 'segformer')
    
    print(f'\n  → Server running. Open http://localhost:{args.port} in your browser\n')

    socketio.run(app, host=args.host, port=args.port,
                 allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
