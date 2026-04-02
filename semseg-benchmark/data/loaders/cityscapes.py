import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import numpy as np

class CityscapesDataset(Dataset):
    """
    PyTorch Dataset wrapper for Custom Cityscapes Semantic Segmentation.
    Expects data to be placed in `root/cityscapes/split/img` and `root/cityscapes/split/label`.
    """
    def __init__(self, root, split='val', transforms=None):
        self.root = root
        
        # Auto-resolve repository data folder if 'Cityscapes' not in current root
        if not os.path.exists(os.path.join(self.root, 'Cityscapes')):
            repo_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data'))
            if os.path.exists(os.path.join(repo_data_dir, 'Cityscapes')):
                self.root = repo_data_dir

        self.split = split
        self.transforms = transforms
        
        self.img_dir = os.path.join(self.root, 'Cityscapes', self.split, 'images')
        self.label_dir = os.path.join(self.root, 'Cityscapes', self.split, 'labels')
        
        self.images = sorted(glob.glob(os.path.join(self.img_dir, '*/*.png')))
        self.masks = sorted(glob.glob(os.path.join(self.label_dir, '*/*_gtFine_labelIds.png')))
        
        if len(self.images) == 0:
            print(f"Warning: No images found in {self.img_dir}")

        # 19-class Cityscapes mapping table
        valid_classes = {
            7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 
            19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 
            26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
        }
        self.id_to_trainid = np.ones(256, dtype=np.uint8) * 255
        for k, v in valid_classes.items():
            self.id_to_trainid[k] = v

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        img_pil = Image.open(img_path).convert('RGB')
        mask_pil = Image.open(mask_path).convert('L')
        
        # Apply 19-class transformation on numpy array lookup
        mask_np = np.array(mask_pil)
        mask_np = self.id_to_trainid[mask_np]

        if self.transforms:
            img_np = np.array(img_pil)
            transformed = self.transforms(image=img_np, mask=mask_np)
            img = transformed['image']
            mask = transformed['mask']
        else:
            # Recreate PIL Image mapping 19 classes back for safe nearest-neighbor resizing 
            mask_pil = Image.fromarray(mask_np)
            
            # Resize PIL images to (256, 256) first to guarantee unified batch sizes
            img_pil = img_pil.resize((256, 256), resample=Image.BILINEAR)
            mask_pil = mask_pil.resize((256, 256), resample=Image.NEAREST)

            # Baseline normalization to (C, H, W) in [0.0, 1.0]
            img = TF.to_tensor(img_pil)
            mask = torch.from_numpy(np.array(mask_pil, dtype=np.int64))

        return {'img': img, 'mask': mask, 'img_path': f"cityscapes_{idx}"}
