import torch
import torch.nn as nn
import numpy as np
from skimage.segmentation import slic

class RegionSegmentation(nn.Module):
    """
    Region segmentation utilizing SLIC superpixels.
    """
    def __init__(self, n_segments=100, compactness=10.0, sigma=1.0):
        super(RegionSegmentation, self).__init__()
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma

    def forward(self, x):
        B, C, H, W = x.shape
        masks = []
        for i in range(B):
            img = x[i]
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # Superpixel prediction
            segments = slic(img_np, n_segments=self.n_segments, compactness=self.compactness, sigma=self.sigma, channel_axis=-1, start_label=1)
            masks.append(torch.from_numpy(segments.astype(np.int64)))

        batch_masks = torch.stack(masks).to(x.device)
        return batch_masks
