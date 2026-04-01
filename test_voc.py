import os
import sys
sys.path.append('./semseg-benchmark') 
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.loaders.voc import VOCDataset
import warnings
warnings.filterwarnings('ignore')

print(f"[{os.getpid()}] Starting VOC Verification...")

# Load standard validation split identical to your run script
ds = VOCDataset(root='./data', split='val')

# Plot the first 5 images arrays
fig, axes = plt.subplots(1, 5, figsize=(20, 5))

for i in range(5):
    target = ds[i]['mask']
    original_classes = torch.unique(target).cpu().numpy().tolist()
    
    # Strip boundary ignore index of 255 for the unique set evaluation strictly
    valid_classes = [c for c in original_classes if c != 255]
    
    print(f"VOC Image {i} Native Unique Values: {original_classes}")
    
    # Visualize explicitly enforcing tab20 with bounds mapping integers strictly
    target_np = target.cpu().numpy()
    target_np[target_np == 255] = 0
    
    axes[i].imshow(target_np, cmap='tab20', vmin=0, vmax=20)
    axes[i].set_title(f"Image {i}\nActive Semantic ID: {valid_classes}")
    axes[i].axis('off')

plt.tight_layout()
os.makedirs('./results', exist_ok=True)
plt.savefig('./results/test_voc_masks.png')
print("Saved array to ./results/test_voc_masks.png")
