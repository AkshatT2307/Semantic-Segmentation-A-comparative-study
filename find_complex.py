import torch
import random
import sys
import os
sys.path.insert(0, os.path.abspath('semseg-benchmark'))
from data.loaders.voc import VOCDataset
from tqdm import tqdm

def find_complex():
    print("Loading VOC Validation Dataset...")
    ds = VOCDataset(root='./data', split='val')
    
    complex_indices = []
    
    print("Scanning for images with >= 4 unique classes (Background + 3 Objects)...")
    for i in tqdm(range(len(ds))):
        target = ds[i]['mask']
        # Extract unique classes and ignore boundary 255
        valids = [c.item() for c in torch.unique(target) if c.item() != 255]
        
        if len(valids) >= 4:
            complex_indices.append(i)
            
    print(f"\nFound {len(complex_indices)} highly complex images!")
    
    # Pick 5 random images securely using a set seed
    random.seed(42)
    selected_5_indices = random.sample(complex_indices, 5)
    
    print(f"Randomly chosen 5 indices for visualization (Seed 42): {selected_5_indices}")
    
if __name__ == "__main__":
    find_complex()
