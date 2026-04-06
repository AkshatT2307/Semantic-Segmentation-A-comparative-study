import sys
import glob
from PIL import Image
import numpy as np

masks = glob.glob('/opt/watchdog/users/cerussite/ImageSegmentation/SemSeg/semseg-benchmark/data/Cityscapes/val/labels/*/*_gtFine_labelIds.png')
if len(masks) == 0:
    print("No masks found!")
    sys.exit(0)

# check the first mask
m = Image.open(masks[0])
print("Original format mode:", m.mode)
print("Unique IDs originally:", np.unique(np.array(m)))

m_L = m.convert('L')
print("Unique IDs after L:", np.unique(np.array(m_L)))

