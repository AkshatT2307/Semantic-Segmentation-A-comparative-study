"""
Visualization utilities for segmentation maps.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


def save_segmentation_maps(images, targets, preds, save_dir, prefix="vis", max_samples=10):
    """
    Save side-by-side visualizations of image / ground truth / prediction.

    Args:
        images:  list of tensors (C, H, W) in [0, 1].
        targets: list of tensors (H, W) integer class ids.
        preds:   list of tensors (H, W) integer class ids.
        save_dir: directory to save PNGs.
        prefix:  filename prefix.
        max_samples: maximum number of samples to save.
    """
    os.makedirs(save_dir, exist_ok=True)
    n = min(len(images), max_samples)

    for i in range(n):
        img_np = images[i].permute(1, 2, 0).numpy()  # (H, W, 3)
        tgt_np = targets[i].numpy().astype(np.float32)
        pred_np = preds[i].numpy().astype(np.float32)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(np.clip(img_np, 0, 1))
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(tgt_np, cmap="tab20", interpolation="nearest")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred_np, cmap="tab20", interpolation="nearest")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{prefix}_{i:04d}.png"), dpi=100)
        plt.close(fig)
