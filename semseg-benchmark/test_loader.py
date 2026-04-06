import torch
print(torch.__version__)
from evaluation.visualizer import *
from data.loaders.cityscapes import CityscapesDataset
from PIL import Image

ds = CityscapesDataset(root='./data', split='val')
if len(ds) > 0:
    item = ds[0]
    mask = item['mask']
    print(torch.unique(mask))
else:
    print("No data found")
