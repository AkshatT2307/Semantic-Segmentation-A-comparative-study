import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def save_segmentation_maps(images, targets, preds, save_dir, prefix="vis", max_samples=10):
    """
    Saves a side-by-side visualization of (Original, Ground Truth, Prediction).
    
    Args:
        images (torch.Tensor): (B, 3, H, W) float tensor normalized to [0, 1].
        targets (torch.Tensor): (B, H, W) integer ground truth mask tensor.
        preds (torch.Tensor): (B, H, W) integer/float predicted mask tensor.
        save_dir (str): output directory.
        prefix (str): string to prepend to the filename.
        max_samples (int): maximum number of samples to process from the batch.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    B = min(len(images), max_samples)
    
    for i in range(B):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        # Clip to [0, 1] just in case
        img = np.clip(img, 0, 1)
        
        target = targets[i].cpu().numpy()
        pred = preds[i].cpu().numpy()
        
        # Extract the true unmapped multi-class indices present
        target_unique = [int(c) for c in np.unique(target) if c != 255]
        
        # Prevent ignored edge indices from mathematically stretching the visible colormap
        target[target == 255] = 0
        pred[pred == 255] = 0

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Use a harsh categorical colormap like 'tab20' so every integer leaps in color hue!
        axes[1].imshow(target, cmap='tab20', interpolation='nearest', vmin=0, vmax=20) 
        axes[1].set_title(f"Ground Truth Mask\nIDs: {target_unique}")
        axes[1].axis('off')

        axes[2].imshow(pred, cmap='tab20', interpolation='nearest', vmin=0, vmax=20)
        axes[2].set_title("Predicted Mask")
        axes[2].axis('off')

        save_path = os.path.join(save_dir, f"{prefix}_{i}.png")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
