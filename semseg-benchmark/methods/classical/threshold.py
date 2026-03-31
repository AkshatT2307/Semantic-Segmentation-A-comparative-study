import torch
import torch.nn as nn
import numpy as np
import cv2

class ThresholdSegmentation(nn.Module):
    """
    A classical threshold-based segmentation algorithm wrapped in a PyTorch module.
    It takes an RGB image batch, converts it to grayscale, and applies global thresholding or Otsu.
    """
    def __init__(self, method='otsu', global_thresh=127):
        """
        Args:
            method (str): 'global' or 'otsu'.
            global_thresh (int): Fixed threshold value if method is 'global'.
        """
        super(ThresholdSegmentation, self).__init__()
        self.method = method
        self.global_thresh = global_thresh

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Output from dataloader, expected shape (B, 3, H, W) normalized to [0,1].
            
        Returns:
            torch.Tensor: Segmentation mask of shape (B, H, W).
        """
        B, C, H, W = x.shape
        masks = []
        
        # We process each item in the batch
        for i in range(B):
            img = x[i]
            
            # Convert from format (C, H, W) in [0, 1] to numpy (H, W, C) in [0, 255]
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # Convert to grayscale
            if img_np.shape[-1] == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np.squeeze()

            if self.method == 'otsu':
                # Otsu's thresholding automatically computes the ideal threshold from the histogram
                ret, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif self.method == 'global':
                # Global thresholding uses the hardcoded global_thresh value
                ret, binary_mask = cv2.threshold(gray, self.global_thresh, 255, cv2.THRESH_BINARY)
            else:
                # Fallback simple threshold
                binary_mask = (gray > self.global_thresh).astype(np.uint8) * 255
            
            # Convert the 255 peak mask to a binary class label mask (0 for background, 1 for foreground)
            binary_mask = (binary_mask / 255).astype(np.int64)
            masks.append(torch.from_numpy(binary_mask))

        # Stack into shape (B, H, W) and move to the device of x
        batch_masks = torch.stack(masks).to(x.device)
        return batch_masks

