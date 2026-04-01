import torch
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
import torchvision.transforms.functional as TF
import numpy as np

class CityscapesDataset(Dataset):
    """
    PyTorch Dataset wrapper for Cityscapes Semantic Segmentation.
    Expects data to be placed in `root/gtFine` and `root/leftImg8bit`.
    """
    def __init__(self, root, split='val', transforms=None):
        self.root = root
        
        # Cityscapes splits are 'train', 'val', 'test'.
        self.cityscapes_ds = Cityscapes(
            root=self.root, 
            split=split,
            mode='fine',
            target_type='semantic'
        )
        self.transforms = transforms

    def __len__(self):
        return len(self.cityscapes_ds)

    def __getitem__(self, idx):
        img_pil, mask_pil = self.cityscapes_ds[idx]

        if self.transforms:
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

        return {'img': img, 'mask': mask, 'img_path': f"cityscapes_{idx}"}
