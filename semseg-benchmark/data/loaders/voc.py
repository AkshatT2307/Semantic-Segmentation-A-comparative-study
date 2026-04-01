import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation
import torchvision.transforms.functional as TF
import numpy as np

class VOCDataset(Dataset):
    """
    PyTorch Dataset wrapper for Pascal VOC 2012 Semantic Segmentation.
    Automatically downloads the dataset if it's not present.
    """
    def __init__(self, root, split='train', transforms=None):
        self.root = root
        
        # Check if downloaded to avoid massive 2GB MD5 checksum delays on evaluation launches
        import os
        is_downloaded = os.path.exists(os.path.join(self.root, 'VOCdevkit'))

        # VOC splits are 'train', 'val', or 'trainval'.
        # For evaluation, 'val' is standard. We map 'val' to 'val'.
        self.voc_ds = VOCSegmentation(
            root=self.root, 
            year='2012', 
            image_set=split, 
            download=not is_downloaded
        )
        self.transforms = transforms

    def __len__(self):
        return len(self.voc_ds)

    def __getitem__(self, idx):
        img_pil, mask_pil = self.voc_ds[idx]

        # By default, VOCSegmentation returns PIL images.
        if self.transforms:
            # Assumes Albumentations or custom dict transforms
            img_np = np.array(img_pil)
            mask_np = np.array(mask_pil)
            transformed = self.transforms(image=img_np, mask=mask_np)
            img = transformed['image']
            mask = transformed['mask']
        else:
            # Resize PIL images to (256, 256) first to guarantee unified batch sizes
            from PIL import Image
            img_pil = img_pil.resize((256, 256), resample=Image.BILINEAR)
            mask_pil = mask_pil.resize((256, 256), resample=Image.NEAREST)

            # Baseline normalization to (C, H, W) in [0.0, 1.0]
            img = TF.to_tensor(img_pil)
            mask = torch.from_numpy(np.array(mask_pil, dtype=np.int64))

        return {'img': img, 'mask': mask, 'img_path': f"voc_{idx}"}
