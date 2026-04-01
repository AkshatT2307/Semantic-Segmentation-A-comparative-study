import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import MiniBatchKMeans

class KMeansSegmentation(nn.Module):
    def __init__(self, n_clusters=5, random_state=42):
        super(KMeansSegmentation, self).__init__()
        self.n_clusters = n_clusters
        self.random_state = random_state

    def forward(self, x):
        B, C, H, W = x.shape
        masks = []
        for i in range(B):
            img = x[i]
            img_np = img.permute(1, 2, 0).cpu().numpy() # [0, 1] HxWxC
            
            pixels = img_np.reshape(-1, C)
            
            kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init="auto")
            labels = kmeans.fit_predict(pixels)
            
            mask = labels.reshape(H, W)
            masks.append(torch.from_numpy(mask.astype(np.int64)))
            
        batch_masks = torch.stack(masks).to(x.device)
        return batch_masks
