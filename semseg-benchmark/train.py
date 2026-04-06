import argparse
import os
import sys
import json

# Add methods/deep to sys.path to resolve 'ultralytics' imports
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'methods/deep'))


import torch
import torch.nn as nn
from tqdm import tqdm
import logging
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from methods.deep.unet import UNet
from methods.deep.segformer import SegFormer
from methods.deep.mask_rcnn import MaskRCNN
from training.losses import CrossEntropyDiceLoss
from evaluation.iou import compute_iou
from data.loaders.coco import CocoStuffDataset
from data.loaders.cityscapes import CityscapesDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate and Train deep segmentation algorithms.")
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "segformer", "segnet", "mask_rcnn"], help="Select model to train.")
    parser.add_argument("--encoder", type=str, default="resnet34", help="Encoder backbone for the model (e.g. resnet34, resnet50, efficientnet-b4). Pretrained ImageNet weights are downloaded automatically.")
    parser.add_argument("--dataset", type=str, default="cityscapes", choices=["coco", "cityscapes"], help="Select Dataset wrapper.")
    parser.add_argument("--data-root", type=str, default="./data", help="Path to dataset root directory (Default: ./data).")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="L2 regularization (weight decay).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on.")
    
    return parser.parse_args()

def train(model, train_loader, val_loader, criterion, optimizer, args, num_classes):
    device = torch.device(args.device)
    model = model.to(device)
    
    out_dir = f"runs/{args.model}_{args.dataset}"
    weights_dir = os.path.join(out_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    # Training history for curves and logs
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_miou": [],
    }
    
    best_miou = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_total = 0
        
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        for batch in tqdm(train_loader, desc="Training"):
            inputs, target = batch['img'].to(device), batch['mask'].to(device)
            predict = model(inputs)
            
            optimizer.zero_grad()
            loss = criterion(predict, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            n_total += inputs.size(0)
            
        train_loss /= n_total
        
        model.eval()
        val_loss, total_iou, num_samples = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                inputs, target = batch['img'].to(device), batch['mask'].to(device)
                predict = model(inputs)
                loss = criterion(predict, target)
                
                pred_class = predict.argmax(dim=1)
                
                _, miou = compute_iou(pred_class, target, num_classes=num_classes, ignore_index=255)
                
                val_loss += loss.item() * inputs.size(0)
                if not torch.isnan(miou):
                    total_iou += miou.item() * inputs.size(0)
                    num_samples += inputs.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_miou = total_iou / num_samples if num_samples > 0 else 0.0
        
        logging.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")
        
        # Record metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)
        
        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), os.path.join(weights_dir, "best.pt"))
            logging.info(f"Saved best model with mIoU: {best_miou:.4f}")
        
        # Save last-epoch checkpoint (model + optimizer + epoch for resuming)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_miou": best_miou,
        }, os.path.join(weights_dir, "last.pt"))
        
        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(weights_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_miou": best_miou,
            }, ckpt_path)
            logging.info(f"Saved periodic checkpoint: {ckpt_path}")
        
        # Save training log after every epoch (overwrite with full history)
        with open(os.path.join(out_dir, "training_log.json"), "w") as f:
            json.dump(history, f, indent=2)
    
    # ---- Save training curves ----
    epochs_range = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(epochs_range, history["train_loss"], label="Train Loss")
    ax1.plot(epochs_range, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{args.model.upper()} — Loss")
    ax1.legend()
    ax1.grid(True)
    
    # mIoU curve
    ax2.plot(epochs_range, history["val_miou"], label="Val mIoU", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mIoU")
    ax2.set_title(f"{args.model.upper()} — Validation mIoU")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    logging.info(f"Training curves saved to {out_dir}/training_curves.png")
    logging.info(f"Training log saved to {out_dir}/training_log.json")
    logging.info(f"Best mIoU: {best_miou:.4f}")

def main():
    args = get_args()
        
    logging.info(f"Loading {args.dataset} dataset from {args.data_root}...")
    
    if args.dataset == 'coco':
        trainset = CocoStuffDataset(root=args.data_root, split='train')
        valset = CocoStuffDataset(root=args.data_root, split='val')
        num_classes = 171 # COCO-Stuff typically has 171
    elif args.dataset == 'cityscapes':
        trainset = CityscapesDataset(root=args.data_root, split='train')
        valset = CityscapesDataset(root=args.data_root, split='val')
        num_classes = 19
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=valset, batch_size=args.batch_size, shuffle=False)
        
    logging.info(f"Initializing {args.model} (encoder={args.encoder}, pretrained=imagenet) for {num_classes} classes...")
    if args.model == "unet":
        model = UNet(n_channels=3, n_classes=num_classes, encoder_name=args.encoder, encoder_weights='imagenet')
    elif args.model == "segformer":
        model = SegFormer(n_channels=3, n_classes=num_classes, freeze_encoder=True, encoder_name=args.encoder, encoder_weights='imagenet')
    elif args.model == "mask_rcnn":
        model = MaskRCNN(n_channels=3, n_classes=num_classes, freeze_encoder=False, encoder_name=args.encoder, encoder_weights='imagenet')
    elif args.model == "segnet":
        raise NotImplementedError("SegNet model not yet implemented.")
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    criterion = CrossEntropyDiceLoss(ignore_index=255)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    logging.info(f"Starting training on {args.device} for {args.epochs} epochs...")
    train(model, train_loader, val_loader, criterion, optimizer, args, num_classes)

if __name__ == "__main__":
    main()
