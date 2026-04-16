import torch
from torch.utils.data import DataLoader
import sys
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.join(SCRIPT_DIR, 'semseg-benchmark')

# Add paths
sys.path.insert(0, BENCHMARK_DIR)
sys.path.insert(0, SCRIPT_DIR)

from convseg_net import convseg_s
from data.loaders.cityscapes import CityscapesDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test ConvSeg-Net on Cityscapes")
    parser.add_argument('--cpu', action='store_true', help='Force CPU even when CUDA is available')
    return parser.parse_args()


args = parse_args()

print("Initializing dataset...")
try:
    dataset = CityscapesDataset(root=os.path.join(BENCHMARK_DIR, 'data'), split='val')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Initializing ConvSeg-Net (Small)...")
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Using device: {device}")
    model = convseg_s(num_classes=19).to(device)
    model.eval()

    print("Fetching a batch...")
    batch = next(iter(loader))
    img = batch['img'].to(device)
    mask = batch['mask'].to(device)

    print(f"Input image shape: {img.shape}")
    print(f"Target mask shape: {mask.shape}")

    print("Running forward pass...")
    with torch.no_grad():
        out = model(img)

    print(f"Output shape: {out.shape}")
    assert out.shape == (1, 19, img.shape[-2], img.shape[-1]), f"Output shape mismatch! Expected (1, 19, {img.shape[-2]}, {img.shape[-1]}), got {out.shape}"
    print("Success! The ConvSeg-Net successfully processed a Cityscapes batch.")
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
