import torch
import torch.nn as nn
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import concurrent.futures
import os

class SVMSegmentation(nn.Module):
    """
    Self-training SVM segmentation: KMeans generates pseudo-labels on a pixel
    subsample, then a LinearSVC refines the decision boundaries and predicts
    for the full image.  Produces multi-class cluster masks (same convention
    as KMeansSegmentation / GMMSegmentation).

    LinearSVC is used instead of kernel SVC for computational tractability on
    full-resolution images (O(n) vs O(n^2) scaling).
    """

    def __init__(self, n_clusters=5, sample_fraction=0.1, random_state=42):
        super(SVMSegmentation, self).__init__()
        self.n_clusters = n_clusters
        self.sample_fraction = sample_fraction
        self.random_state = random_state
        self.max_workers = os.cpu_count()

    def _process_single(self, img_tensor):
        C, H, W = img_tensor.shape
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # HxWxC [0, 1]
        pixels = img_np.reshape(-1, C)

        # Standardize features for SVM
        scaler = StandardScaler()
        pixels_scaled = scaler.fit_transform(pixels)

        n_pixels = pixels_scaled.shape[0]
        n_sample = max(int(n_pixels * self.sample_fraction), self.n_clusters * 10)
        n_sample = min(n_sample, n_pixels)

        rng = np.random.RandomState(self.random_state)
        sample_idx = rng.choice(n_pixels, size=n_sample, replace=False)
        sample_pixels = pixels_scaled[sample_idx]

        # Phase 1: KMeans pseudo-labels on the subsample
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init='auto',
        )
        pseudo_labels = kmeans.fit_predict(sample_pixels)

        # Phase 2: Train LinearSVC on pseudo-labeled subsample, predict full image
        svm = LinearSVC(random_state=self.random_state, max_iter=1000, dual='auto')
        svm.fit(sample_pixels, pseudo_labels)
        labels = svm.predict(pixels_scaled)

        mask = labels.reshape(H, W)
        return torch.from_numpy(mask.astype(np.int64))

    def forward(self, x):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            masks = list(executor.map(self._process_single, [x[i] for i in range(x.size(0))]))

        batch_masks = torch.stack(masks).to(x.device)
        return batch_masks
