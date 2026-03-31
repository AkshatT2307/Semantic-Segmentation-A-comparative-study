import argparse
import os
import torch
import numpy as np
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("eval_reports.log", mode='a'),
        logging.StreamHandler()
    ]
)

from data.loaders.coco import CocoStuffDataset
from data.loaders.voc import VOCDataset
from methods.classical.threshold import ThresholdSegmentation
from evaluation.iou import compute_iou
from evaluation.pixel_acc import compute_pixel_accuracy

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate threshold segmentation algorithms.")
    parser.add_argument("--dataset", type=str, default="voc", choices=["coco", "voc"], help="Select Dataset wrapper.")
    parser.add_argument("--data-root", type=str, default="./data", help="Path to dataset root directory (Default: ./data).")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Dataset split to evaluate on.")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--method", type=str, default="otsu", choices=["otsu", "global"], help="Thresholding method.")
    parser.add_argument("--global-thresh", type=int, default=127, help="Global threshold value (used iff method=global).")
    
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Executing Threshold Evaluation ({args.method}) tracking on {device}...")

    # Load dataset
    if args.dataset == 'coco':
        if not os.path.exists(args.data_root):
            logging.error(f"Dataset directory {args.data_root} doesn't exist.")
            return
        dataset = CocoStuffDataset(root=args.data_root, split=args.split)
    elif args.dataset == 'voc':
        logging.info(f"Initializing Pascal VOC loader inside {args.data_root} ...")
        dataset = VOCDataset(root=args.data_root, split=args.split)
        
    if len(dataset) == 0:
        logging.error("Dataset empty. Check your data root path.")
        return
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize classical model
    model = ThresholdSegmentation(method=args.method, global_thresh=args.global_thresh).to(device)

    total_iou = 0.0
    total_acc = 0.0
    num_samples = 0

    logging.info(f"Starting Evaluation Iteration for {args.dataset}...")
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            img = batch['img'].to(device)           # (B, 3, H, W)
            target = batch['mask'].to(device)       # (B, H, W) [classes depend on dataset]
            
            # Predict
            pred = model(img)                       # (B, H, W) binary (0 or 1)
            
            # Binarize Target: Assume class '0' is background, ignores are '255', >0 is foreground.
            target_bin = ((target > 0) & (target != 255)).long()
            
            # Since target has ignore index 255, we mask them out from metric computation
            target_bin[target == 255] = 255

            _, miou = compute_iou(pred, target_bin, num_classes=2, ignore_index=255)
            acc = compute_pixel_accuracy(pred, target_bin, ignore_index=255)

            if not torch.isnan(miou):
                total_iou += miou.item()
                total_acc += acc.item()
                num_samples += 1

    if num_samples > 0:
        final_miou = total_iou / num_samples
        final_acc = total_acc / num_samples
        logging.info("\n=== Evaluation Results ===")
        logging.info(f"Algorithm:       Threshold ({args.method})")
        logging.info(f"Dataset:         {args.dataset.upper()} ({args.split})")
        logging.info(f"mIoU (Binary):   {final_miou:.4f}")
        logging.info(f"Pixel Accuracy:  {final_acc:.4f}")
    else:
        logging.warning("Completed loop, but no valid samples were evaluated. Please check dataset masks.")

if __name__ == "__main__":
    main()
