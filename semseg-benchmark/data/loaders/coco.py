import os
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path

class CocoStuffDataset(Dataset):
    """
    PyTorch Dataset for loading COCO-Stuff semantic segmentation data.
    """
    def __init__(self, root, split='train', transforms=None):
        self.root = Path(root)
        self.split = split
        self.transforms = transforms

        image_dir = self.root / 'images' / f'{split}2017'
        mask_dir = self.root / 'annotations' / f'{split}2017'
        
        self.images = sorted(list(image_dir.glob('*.jpg')))
        self.masks = sorted(list(mask_dir.glob('*.png')))
        
        if not self.images:
            print(f"Warning: No images found for split '{split}' in {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        mask_path = str(self.masks[idx])

        # Load image (RGB)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask (semantic segmentation labels are pixel values)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transforms:
            # Applies albumentation-style transformations that expect dict inputs
            transformed = self.transforms(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        else:
            # Baseline default scaling and normalization to standard PyTorch tensor shapes
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return {'img': img, 'mask': mask, 'img_path': img_path}
