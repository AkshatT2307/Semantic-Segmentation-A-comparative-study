import argparse
import os
import torch
import numpy as np
import logging
from tqdm import tqdm
import random
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
from data.loaders.cityscapes import CityscapesDataset
from methods.classical.threshold import ThresholdSegmentation
from methods.classical.graph_cut import GraphCutSegmentation
from methods.classical.region import RegionSegmentation
from methods.classical.edge import EdgeSegmentation
from methods.ml.kmeans import KMeansSegmentation
from utils.visualize import save_segmentation_maps
from evaluation.iou import compute_iou
from evaluation.pixel_acc import compute_pixel_accuracy
from evaluation.mappings import map_clusters_to_classes

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate threshold segmentation algorithms.")
    parser.add_argument("--dataset", type=str, default="voc", choices=["coco", "voc", "cityscapes"], help="Select Dataset wrapper.")
    parser.add_argument("--data-root", type=str, default="./data", help="Path to dataset root directory (Default: ./data).")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Dataset split to evaluate on.")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--method", type=str, default="otsu", choices=["otsu", "global", "graph_cut", "region", "kmeans", "edge"], help="Segmentation method.")
    parser.add_argument("--global-thresh", type=int, default=127, help="Global threshold value (used iff method=global).")
    parser.add_argument("--visualize", action="store_true", help="Save visualization maps.")
    parser.add_argument("--vis-count", type=int, default=10, help="Number of random samples to visualize.")
    parser.add_argument("--vis-seed", type=int, default=42, help="Seed for bounding deterministic visualizations randomly scattering over dataset.")
    parser.add_argument("--vis-complex", action="store_true", help="Forces the visualizer array to specifically isolate images with >= 4 unique semantic classes!")
    
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Executing Evaluation ({args.method}) tracking on {device}...")

    # Load dataset
    if args.dataset == 'coco':
        if not os.path.exists(args.data_root):
            logging.error(f"Dataset directory {args.data_root} doesn't exist.")
            return
        dataset = CocoStuffDataset(root=args.data_root, split=args.split)
    elif args.dataset == 'voc':
        logging.info(f"Initializing Pascal VOC loader inside {args.data_root} ...")
        dataset = VOCDataset(root=args.data_root, split=args.split)
    elif args.dataset == 'cityscapes':
        logging.info(f"Initializing Cityscapes loader inside {args.data_root} ...")
        dataset = CityscapesDataset(root=args.data_root, split=args.split)
        
    if len(dataset) == 0:
        logging.error("Dataset empty. Check your data root path.")
        return
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize classical model
    if args.method in ['otsu', 'global']:
        model = ThresholdSegmentation(method=args.method, global_thresh=args.global_thresh).to(device)
    elif args.method == 'graph_cut':
        model = GraphCutSegmentation().to(device)
    elif args.method == 'region':
        model = RegionSegmentation().to(device)
    elif args.method == 'kmeans':
        model = KMeansSegmentation().to(device)
    elif args.method == 'edge':
        model = EdgeSegmentation().to(device)

    vis_target_indices = set()
    if args.visualize:
        random.seed(args.vis_seed)
        if args.vis_complex:
            complex_cache_path = os.path.join(args.data_root, f"{args.dataset}_{args.split}_complex_indices.npy")
            if os.path.exists(complex_cache_path):
                logging.info(f"Loading cached complex indices from {complex_cache_path}...")
                complex_idx = np.load(complex_cache_path).tolist()
            else:
                logging.info("Scanning dataset purely isolating highly complex semantic scenes (>= 4 IDs)...")
                complex_idx = []
                for i in tqdm(range(len(dataset)), desc="Complexity Scan"):
                    target_mask = dataset[i]['mask']
                    valids = [c.item() for c in torch.unique(target_mask) if c.item() != 255]
                    if len(valids) >= 4:
                        complex_idx.append(i)
                np.save(complex_cache_path, complex_idx)
                logging.info(f"Saved complex indices cache to {complex_cache_path}!")
                
            vis_target_indices = set(random.sample(complex_idx, min(args.vis_count, len(complex_idx))))
            logging.info(f"Locked {len(vis_target_indices)} deterministic complex samples dynamically mapped!")
        else:
            vis_target_indices = set(random.sample(range(len(dataset)), min(args.vis_count, len(dataset))))

    total_iou = 0.0
    total_acc = 0.0
    num_samples = 0
    
    vis_images = []
    vis_targets = []
    vis_preds = []

    global_idx = 0
    logging.info(f"Starting Evaluation Iteration for {args.dataset}...")
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            img = batch['img'].to(device)           # (B, 3, H, W)
            target = batch['mask'].to(device)       # (B, H, W) [classes depend on dataset]
            
            # Predict
            pred = model(img)                       # (B, H, W) binary (0 or 1)
            
            if args.method in ['otsu', 'global', 'edge']:
                # Binarize Target for binary models: Assume class '0' is background, >=1 is foreground.
                target_eval = ((target > 0) & (target != 255)).long()
                target_eval[target == 255] = 255
                pred_eval = pred
                eval_classes = 2
            elif args.method in ['graph_cut', 'region', 'kmeans']:
                # Keep multi-class and dynamically map clusters to semantic classes
                target_eval = target
                pred_eval = map_clusters_to_classes(pred, target, ignore_index=255)
                # Ensure IoU handles up to max semantic class values
                eval_classes = 256
            
            # Append final mapped evaluations for visualization securely
            if args.visualize:
                for b in range(pred_eval.shape[0]):
                    if (global_idx + b) in vis_target_indices:
                        vis_images.append(img[b].cpu())
                        vis_targets.append(target_eval[b].cpu())
                        vis_preds.append(pred_eval[b].cpu())
                        
            # Advance iteration index bounds dynamically over the batch dimension natively
            global_idx += img.shape[0]

            _, miou = compute_iou(pred_eval, target_eval, num_classes=eval_classes, ignore_index=255)
            acc = compute_pixel_accuracy(pred_eval, target_eval, ignore_index=255)

            if not torch.isnan(miou):
                total_iou += miou.item()
                total_acc += acc.item()
                num_samples += 1

    if num_samples > 0:
        final_miou = total_iou / num_samples
        final_acc = total_acc / num_samples
        logging.info("\n=== Evaluation Results ===")
        logging.info(f"Algorithm:       {args.method.upper()}")
        logging.info(f"Dataset:         {args.dataset.upper()} ({args.split})")
        logging.info(f"mIoU:            {final_miou:.4f}")
        logging.info(f"Pixel Accuracy:  {final_acc:.4f}")
    else:
        logging.warning("Completed loop, but no valid samples were evaluated. Please check dataset masks.")
        
    if args.visualize and len(vis_images) > 0:
        logging.info("Saving visualization maps...")
        
        save_dir = os.path.join(".", "results", f"{args.dataset}_{args.method}")
        save_segmentation_maps(vis_images, vis_targets, vis_preds, save_dir, prefix="vis", max_samples=len(vis_images))
        logging.info(f"Visualizations saved to {save_dir}")

if __name__ == "__main__":
    main()
