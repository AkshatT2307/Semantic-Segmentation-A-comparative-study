#!/usr/bin/env python3
"""
MMSegmentation Inference Script for Cityscapes
===============================================
Runs pretrained FCN-R50-D8 and SegFormer-MiT-B1 models on Cityscapes
validation set. Supports full mIoU evaluation, single-image inference,
and visualization output.

Usage:
    # Evaluate FCN on Cityscapes val
    python mmseg_inference.py --model fcn --eval

    # Evaluate SegFormer on Cityscapes val
    python mmseg_inference.py --model segformer --eval

    # Evaluate both models
    python mmseg_inference.py --model all --eval

    # Single image inference with visualization
    python mmseg_inference.py --model fcn --image path/to/image.png

    # Full eval with visualizations saved
    python mmseg_inference.py --model fcn --eval --vis-count 20
"""

import argparse
import glob
import json
import os
import sys
import time
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Suppress mmcv-lite warnings
warnings.filterwarnings('ignore', message='.*mmcv-lite.*')
warnings.filterwarnings('ignore', message='.*MultiScaleDeformableAttention.*')

from mmseg.apis import init_model, inference_model
from mmseg.utils import get_classes, get_palette


# ─── Cityscapes labelId → trainId mapping ────────────────────────────────────
# Original 34 Cityscapes label IDs mapped to 19 training classes + 255 (ignore)
LABEL_ID_TO_TRAIN_ID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0,    # road
    8: 1,    # sidewalk
    9: 255, 10: 255,
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    14: 255, 15: 255, 16: 255,
    17: 5,   # pole
    18: 255,
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    29: 255, 30: 255,
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
    -1: 255, 255: 255,
}

# Build a numpy LUT for fast remapping (covers 0..255)
_REMAP_LUT = np.full(256, 255, dtype=np.uint8)
for lid, tid in LABEL_ID_TO_TRAIN_ID.items():
    if 0 <= lid < 256:
        _REMAP_LUT[lid] = tid

CITYSCAPES_CLASSES = get_classes('cityscapes')
CITYSCAPES_PALETTE = get_palette('cityscapes')


# ─── Model registry ─────────────────────────────────────────────────────────

MODELS = {
    'fcn': {
        'config': 'configs/fcn_r50-d8_cityscapes.py',
        'checkpoint': 'weights/fcn_r50-d8_512x1024_40k_cityscapes.pth',
        'name': 'FCN-R50-D8',
    },
    'segformer': {
        'config': 'configs/segformer_mit-b1_cityscapes.py',
        'checkpoint': 'weights/segformer_mit-b1_8x1_1024x1024_160k_cityscapes_20211208_064213-655c7b3f.pth',
        'name': 'SegFormer-MiT-B1',
    },
}


# ─── Evaluation helpers ─────────────────────────────────────────────────────

def remap_label_ids(label_img):
    """Convert Cityscapes labelIds to trainIds using LUT."""
    return _REMAP_LUT[label_img]


def compute_iou_per_class(pred, gt, num_classes=19, ignore_index=255):
    """Compute per-class IoU between prediction and ground truth."""
    assert pred.shape == gt.shape
    valid = gt != ignore_index
    pred_v = pred[valid]
    gt_v = gt[valid]

    intersection = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)

    for c in range(num_classes):
        pred_c = pred_v == c
        gt_c = gt_v == c
        intersection[c] = np.logical_and(pred_c, gt_c).sum()
        union[c] = np.logical_or(pred_c, gt_c).sum()

    return intersection, union


def collect_cityscapes_val_pairs(data_root):
    """Collect (image_path, label_path) pairs for Cityscapes val set."""
    img_dir = os.path.join(data_root, 'leftImg8bit', 'val')
    lbl_dir = os.path.join(data_root, 'gtFine', 'val')

    pairs = []
    for city in sorted(os.listdir(img_dir)):
        city_img_dir = os.path.join(img_dir, city)
        city_lbl_dir = os.path.join(lbl_dir, city)
        if not os.path.isdir(city_img_dir):
            continue
        for img_file in sorted(os.listdir(city_img_dir)):
            if not img_file.endswith('_leftImg8bit.png'):
                continue
            prefix = img_file.replace('_leftImg8bit.png', '')
            lbl_file = f'{prefix}_gtFine_labelIds.png'
            lbl_path = os.path.join(city_lbl_dir, lbl_file)
            if os.path.exists(lbl_path):
                pairs.append((os.path.join(city_img_dir, img_file), lbl_path))
    return pairs


# ─── Visualization ───────────────────────────────────────────────────────────

def colorize_mask(mask, palette, num_classes=19):
    """Convert a class-index mask to an RGB color image."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(num_classes):
        color[mask == c] = palette[c]
    return color


def save_visualization(img_path, pred_mask, gt_mask, save_path, palette,
                       opacity=0.5):
    """Save a side-by-side visualization: image, pred overlay, GT overlay."""
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Resize masks to image size if needed
    if pred_mask.shape != (h, w):
        pred_mask = cv2.resize(pred_mask, (w, h),
                               interpolation=cv2.INTER_NEAREST)
    if gt_mask.shape != (h, w):
        gt_mask = cv2.resize(gt_mask, (w, h),
                             interpolation=cv2.INTER_NEAREST)

    pred_color = colorize_mask(pred_mask, palette)
    gt_color = colorize_mask(gt_mask, palette)

    # Overlay
    pred_overlay = (img_rgb * (1 - opacity) + pred_color * opacity).astype(
        np.uint8)
    gt_overlay = (img_rgb * (1 - opacity) + gt_color * opacity).astype(
        np.uint8)

    # Stack: Original | GT | Prediction
    canvas = np.concatenate([img_rgb, gt_overlay, pred_overlay], axis=1)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, 'Input', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(canvas, 'Ground Truth', (w + 10, 30), font, 1,
                (255, 255, 255), 2)
    cv2.putText(canvas, 'Prediction', (2 * w + 10, 30), font, 1,
                (255, 255, 255), 2)

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, canvas_bgr)


def save_class_legend(palette, classes, save_path):
    """Save a color legend mapping class indices to names."""
    n = len(classes)
    cell_h, cell_w = 30, 300
    canvas = np.zeros((n * cell_h, cell_w, 3), dtype=np.uint8)
    for i, (name, color) in enumerate(zip(classes, palette)):
        y = i * cell_h
        canvas[y:y + cell_h, :60] = color
        cv2.putText(canvas, f'{i}: {name}', (70, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, canvas_bgr)


# ─── Main routines ───────────────────────────────────────────────────────────

def evaluate_model(model_key, data_root, device, vis_count=0, vis_dir=None):
    """Run full mIoU evaluation on Cityscapes val set."""
    info = MODELS[model_key]
    print(f'\n{"="*70}')
    print(f'  Evaluating: {info["name"]}')
    print(f'  Config:     {info["config"]}')
    print(f'  Checkpoint: {info["checkpoint"]}')
    print(f'  Device:     {device}')
    print(f'{"="*70}\n')

    # Init model
    model = init_model(info['config'], info['checkpoint'], device=device)

    # Collect val pairs
    pairs = collect_cityscapes_val_pairs(data_root)
    print(f'Found {len(pairs)} validation image-label pairs')
    if len(pairs) == 0:
        print('ERROR: No validation pairs found. Check data_root.')
        return None

    # Evaluation
    total_intersection = np.zeros(19, dtype=np.int64)
    total_union = np.zeros(19, dtype=np.int64)
    total_correct = 0
    total_valid = 0

    # Select visualization indices
    vis_indices = set()
    if vis_count > 0:
        step = max(1, len(pairs) // vis_count)
        vis_indices = set(range(0, len(pairs), step)[:vis_count])
        if vis_dir:
            model_vis_dir = os.path.join(vis_dir, model_key)
            os.makedirs(model_vis_dir, exist_ok=True)
            save_class_legend(CITYSCAPES_PALETTE, CITYSCAPES_CLASSES,
                              os.path.join(model_vis_dir, 'legend.png'))

    t_start = time.time()
    for idx, (img_path, lbl_path) in enumerate(tqdm(pairs,
                                                     desc=f'{info["name"]}')):
        # Run inference
        result = inference_model(model, img_path)
        pred = result.pred_sem_seg.data.cpu().numpy().squeeze()  # (H, W)

        # Load GT and remap labelIds → trainIds
        gt_raw = np.array(Image.open(lbl_path), dtype=np.uint8)
        gt = remap_label_ids(gt_raw)

        # Resize pred to gt size if needed
        if pred.shape != gt.shape:
            pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        # Accumulate IoU
        inter, union = compute_iou_per_class(pred, gt)
        total_intersection += inter
        total_union += union

        # Pixel accuracy
        valid = gt != 255
        total_correct += (pred[valid] == gt[valid]).sum()
        total_valid += valid.sum()

        # Save visualization
        if idx in vis_indices and vis_dir:
            vis_path = os.path.join(model_vis_dir,
                                    f'{os.path.basename(img_path)}')
            save_visualization(img_path, pred, gt, vis_path,
                               CITYSCAPES_PALETTE)

    elapsed = time.time() - t_start

    # Compute metrics
    with np.errstate(divide='ignore', invalid='ignore'):
        iou_per_class = total_intersection / np.maximum(total_union, 1)
    miou = iou_per_class.mean()
    pixel_acc = total_correct / max(total_valid, 1)

    # Print results
    print(f'\n{"─"*70}')
    print(f'  Results: {info["name"]}')
    print(f'{"─"*70}')
    print(f'  mIoU:           {miou * 100:.2f}%')
    print(f'  Pixel Accuracy: {pixel_acc * 100:.2f}%')
    print(f'  Inference Time: {elapsed:.1f}s ({elapsed/len(pairs):.2f}s/img)')
    print(f'\n  Per-class IoU:')
    for c in range(19):
        bar = '█' * int(iou_per_class[c] * 30)
        print(f'    {c:2d} {CITYSCAPES_CLASSES[c]:20s}  '
              f'{iou_per_class[c]*100:5.1f}%  {bar}')
    print(f'{"─"*70}\n')

    # Save results to JSON
    results = {
        'model': info['name'],
        'checkpoint': info['checkpoint'],
        'mIoU': float(miou),
        'pixel_accuracy': float(pixel_acc),
        'num_images': len(pairs),
        'inference_time_s': float(elapsed),
        'per_class_iou': {
            CITYSCAPES_CLASSES[c]: float(iou_per_class[c])
            for c in range(19)
        }
    }
    return results


def infer_single_image(model_key, img_path, device, out_dir='outputs'):
    """Run inference on a single image and save visualization."""
    info = MODELS[model_key]
    print(f'Loading {info["name"]}...')
    model = init_model(info['config'], info['checkpoint'], device=device)

    print(f'Running inference on: {img_path}')
    result = inference_model(model, img_path)
    pred = result.pred_sem_seg.data.cpu().numpy().squeeze()

    os.makedirs(out_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(img_path))[0]

    # Color mask
    pred_color = colorize_mask(pred, CITYSCAPES_PALETTE)
    pred_path = os.path.join(out_dir, f'{basename}_{model_key}_pred.png')
    cv2.imwrite(pred_path, cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

    # Overlay
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    if pred_color.shape[:2] != (h, w):
        pred_color = cv2.resize(pred_color, (w, h),
                                interpolation=cv2.INTER_NEAREST)
    overlay = (img_rgb * 0.5 + pred_color * 0.5).astype(np.uint8)
    overlay_path = os.path.join(out_dir, f'{basename}_{model_key}_overlay.png')
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f'Saved: {pred_path}')
    print(f'Saved: {overlay_path}')

    # Legend
    save_class_legend(CITYSCAPES_PALETTE, CITYSCAPES_CLASSES,
                      os.path.join(out_dir, 'legend.png'))


def main():
    parser = argparse.ArgumentParser(
        description='MMSeg inference on Cityscapes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--model', type=str, default='all',
                        choices=['fcn', 'segformer', 'all'],
                        help='Model to run (default: all)')
    parser.add_argument('--eval', action='store_true',
                        help='Run full mIoU evaluation on val set')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image for inference')
    parser.add_argument('--data-root', type=str, default='data/cityscapes',
                        help='Path to cityscapes data root')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: auto)')
    parser.add_argument('--vis-count', type=int, default=10,
                        help='Number of visualizations to save during eval')
    parser.add_argument('--vis-dir', type=str, default='results/mmseg_vis',
                        help='Directory to save visualizations')
    parser.add_argument('--out-dir', type=str, default='results/mmseg_eval',
                        help='Directory to save evaluation results')
    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    models_to_run = list(MODELS.keys()) if args.model == 'all' else [args.model]

    # Single image mode
    if args.image:
        for m in models_to_run:
            infer_single_image(m, args.image, args.device, args.out_dir)
        return

    # Evaluation mode
    if args.eval:
        all_results = {}
        for m in models_to_run:
            results = evaluate_model(m, args.data_root, args.device,
                                     vis_count=args.vis_count,
                                     vis_dir=args.vis_dir)
            if results:
                all_results[m] = results

        # Save combined results
        os.makedirs(args.out_dir, exist_ok=True)
        results_path = os.path.join(args.out_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\nResults saved to: {results_path}')

        # Print comparison table
        if len(all_results) > 1:
            print(f'\n{"="*50}')
            print(f'  Model Comparison')
            print(f'{"="*50}')
            print(f'  {"Model":<25s} {"mIoU":>8s} {"Pixel Acc":>10s}')
            print(f'  {"─"*45}')
            for m, r in all_results.items():
                print(f'  {r["model"]:<25s} '
                      f'{r["mIoU"]*100:7.2f}% '
                      f'{r["pixel_accuracy"]*100:9.2f}%')
            print()
        return

    parser.print_help()
    print('\nError: Specify --eval or --image <path>')


if __name__ == '__main__':
    main()
