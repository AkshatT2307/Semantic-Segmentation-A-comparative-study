import torch
import torch.nn as nn
import numpy as np
from skimage.segmentation import felzenszwalb

class GraphCutSegmentation(nn.Module):
    """
    Graph-based image segmentation using Felzenszwalb's efficient graph algorithm.
    It produces arbitrary integer region clusters based on edges and graph MSTs.
    """
    def __init__(self, scale=100.0, sigma=0.5, min_size=50):
        super(GraphCutSegmentation, self).__init__()
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size

    def forward(self, x):
        """
        x: (B, 3, H, W) normalized to [0,1].
        """
        B, C, H, W = x.shape
        masks = []
        for i in range(B):
            img = x[i]
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # Predict clusters using felzenszwalb
            segments = felzenszwalb(img_np, scale=self.scale, sigma=self.sigma, min_size=self.min_size)
            
            masks.append(torch.from_numpy(segments.astype(np.int64)))
            
        batch_masks = torch.stack(masks).to(x.device)
        return batch_masks
