import os
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path

class AerialDataset(Dataset):
    """
    PyTorch Dataset for loading Semantic Drone Dataset (aerial semantic segmentation).
    """
    def __init__(self, root, split='train', transforms=None):
        self.root = root
        
        # Auto-resolve repository data folder if 'semantic_drone_dataset' not in current root
        if not os.path.exists(os.path.join(self.root, 'semantic_drone_dataset')):
            repo_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data'))
            if os.path.exists(os.path.join(repo_data_dir, 'semantic_drone_dataset')):
                self.root = repo_data_dir

        self.root = Path(self.root) / 'semantic_drone_dataset' / 'semantic_drone_dataset'
        self.split = split
        self.transforms = transforms

        image_dir = self.root / 'original_images'
        mask_dir = self.root / 'label_images_semantic'
        
        self.images = sorted(list(image_dir.glob('*.jpg')))
        self.masks = sorted(list(mask_dir.glob('*.png')))
        
        if not self.images:
            print(f"Warning: No images found in {image_dir}")

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
            # Resize via cv2
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return {'img': img, 'mask': mask, 'img_path': os.path.basename(img_path)}
