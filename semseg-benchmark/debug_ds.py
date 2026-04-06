import torch
import sys

# add path to allow imports
sys.path.append('/opt/watchdog/users/cerussite/ImageSegmentation/SemSeg/semseg-benchmark')

from data.loaders.cityscapes import CityscapesDataset

ds = CityscapesDataset(root='./data', split='val')
if len(ds) == 0:
    print("Dataset is empty. Cannot debug.")
else:
    for i in range(min(5, len(ds))):
        mask = ds[i]['mask']
        print(f"Sample {i} unique IDs:", torch.unique(mask).tolist())
