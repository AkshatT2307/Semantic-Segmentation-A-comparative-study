"""
validate_all.py — Smoke-test every method (classical / ML / deep) end-to-end.

Tests:
  1. Model instantiation
  2. Forward pass (inference) with a synthetic batch
  3. For deep models: backward pass (training step) + checkpoint save/load
  4. For deep models: verify checkpoint contains the right keys and can be
     reloaded to produce the same predictions

Run:
    python validate_all.py
"""

import os
import sys
import json
import traceback
import tempfile

import torch
import torch.nn as nn
import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from training.losses import CrossEntropyDiceLoss
from evaluation.iou import compute_iou
from evaluation.pixel_acc import compute_pixel_accuracy

# ── helpers ──────────────────────────────────────────────────────────────────
PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"

results = []

def log_result(name, passed, msg=""):
    status = PASS if passed else FAIL
    detail = f" — {msg}" if msg else ""
    print(f"  {status}  {name}{detail}")
    results.append((name, passed, msg))


def make_batch(B=2, C=3, H=64, W=64):
    """Return a synthetic RGB batch in [0, 1]."""
    return torch.rand(B, C, H, W)


def make_target(B=2, H=64, W=64, num_classes=19):
    """Return a synthetic integer mask with some ignore pixels."""
    target = torch.randint(0, num_classes, (B, H, W))
    # Sprinkle ~5 % ignore pixels
    ignore_mask = torch.rand(B, H, W) < 0.05
    target[ignore_mask] = 255
    return target


# ── 1. Classical methods ─────────────────────────────────────────────────────
def test_classical():
    print("\n═══ Classical Methods ═══")
    from methods.classical.threshold import ThresholdSegmentation
    from methods.classical.graph_cut import GraphCutSegmentation
    from methods.classical.region import RegionSegmentation
    from methods.classical.edge import EdgeSegmentation

    x = make_batch()

    for name, cls, kwargs in [
        ("Otsu Threshold",   ThresholdSegmentation, {"method": "otsu"}),
        ("Global Threshold", ThresholdSegmentation, {"method": "global", "global_thresh": 127}),
        ("Graph Cut",        GraphCutSegmentation,  {}),
        ("Region (SLIC)",    RegionSegmentation,    {}),
        ("Edge (Canny)",     EdgeSegmentation,      {}),
    ]:
        try:
            model = cls(**kwargs)
            out = model(x)
            assert out.shape == (x.size(0), x.size(2), x.size(3)), f"Expected (B,H,W), got {out.shape}"
            assert out.dtype == torch.int64, f"Expected int64, got {out.dtype}"
            log_result(f"{name} — forward", True, f"output {out.shape}")
        except Exception as e:
            log_result(f"{name} — forward", False, str(e))


# ── 2. ML methods ────────────────────────────────────────────────────────────
def test_ml():
    print("\n═══ ML Methods ═══")
    from methods.ml.kmeans import KMeansSegmentation
    from methods.ml.gmm import GMMSegmentation
    from methods.ml.svm import SVMSegmentation

    x = make_batch()

    for name, cls, kwargs in [
        ("KMeans",  KMeansSegmentation, {}),
        ("GMM",     GMMSegmentation,    {}),
        ("SVM",     SVMSegmentation,    {}),
    ]:
        try:
            model = cls(**kwargs)
            out = model(x)
            assert out.shape == (x.size(0), x.size(2), x.size(3)), f"Expected (B,H,W), got {out.shape}"
            assert out.dtype == torch.int64, f"Expected int64, got {out.dtype}"
            log_result(f"{name} — forward", True, f"output {out.shape}")
        except Exception as e:
            log_result(f"{name} — forward", False, str(e))


# ── 3. Deep methods ──────────────────────────────────────────────────────────
def test_deep():
    print("\n═══ Deep Methods ═══")
    from methods.deep.unet import UNet
    from methods.deep.segformer import SegFormer
    from methods.deep.mask_rcnn import MaskRCNN  # actually FPN alias

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  (device: {device})")

    num_classes = 19
    B, C, H, W = 2, 3, 64, 64
    x = make_batch(B, C, H, W).to(device)
    target = make_target(B, H, W, num_classes).to(device)

    criterion = CrossEntropyDiceLoss(ignore_index=255)

    for name, factory, kwargs in [
        ("UNet",      UNet,      {"n_channels": C, "n_classes": num_classes, "encoder_name": "resnet34", "encoder_weights": "imagenet"}),
        ("SegFormer", SegFormer, {"n_channels": C, "n_classes": num_classes, "freeze_encoder": True, "encoder_name": "resnet34", "encoder_weights": "imagenet"}),
        ("FPN/MaskRCNN", MaskRCNN, {"n_channels": C, "n_classes": num_classes, "freeze_encoder": False, "encoder_name": "resnet34", "encoder_weights": "imagenet"}),
    ]:
        # ---- Instantiation ----
        try:
            model = factory(**kwargs).to(device)
            log_result(f"{name} — instantiation", True)
        except Exception as e:
            log_result(f"{name} — instantiation", False, str(e))
            continue

        # ---- Forward (inference) ----
        try:
            model.eval()
            with torch.no_grad():
                out = model(x)
            assert out.shape == (B, num_classes, H, W), f"Expected (B,C,H,W), got {out.shape}"
            log_result(f"{name} — inference forward", True, f"output {out.shape}")
        except Exception as e:
            log_result(f"{name} — inference forward", False, str(e))

        # ---- Backward (training step) ----
        try:
            model.train()
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
            out = model(x)
            loss = criterion(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            log_result(f"{name} — training step", True, f"loss={loss.item():.4f}")
        except Exception as e:
            log_result(f"{name} — training step", False, str(e))

        # ---- IoU / pixel-acc on dummy ----
        try:
            model.eval()
            with torch.no_grad():
                out = model(x)
            pred_class = out.argmax(dim=1)
            _, miou = compute_iou(pred_class, target, num_classes=num_classes, ignore_index=255)
            acc = compute_pixel_accuracy(pred_class, target, ignore_index=255)
            log_result(f"{name} — metrics", True, f"mIoU={miou.item():.4f}  acc={acc.item():.4f}")
        except Exception as e:
            log_result(f"{name} — metrics", False, str(e))

        # ---- Checkpoint save & reload ----
        try:
            tmp = tempfile.mkdtemp()
            best_path = os.path.join(tmp, "best.pt")
            last_path = os.path.join(tmp, "last.pt")

            # Save best (state_dict only)
            torch.save(model.state_dict(), best_path)
            assert os.path.isfile(best_path), "best.pt not created"
            assert os.path.getsize(best_path) > 0, "best.pt is empty"

            # Save last (full checkpoint)
            torch.save({
                "epoch": 0,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_miou": 0.0,
            }, last_path)
            assert os.path.isfile(last_path), "last.pt not created"
            assert os.path.getsize(last_path) > 0, "last.pt is empty"

            log_result(f"{name} — checkpoint save", True,
                       f"best={os.path.getsize(best_path)//1024}KB  last={os.path.getsize(last_path)//1024}KB")

            # ---- Reload & verify ----
            ckpt = torch.load(last_path, map_location=device, weights_only=False)
            assert "epoch" in ckpt, "Missing 'epoch' key"
            assert "model_state_dict" in ckpt, "Missing 'model_state_dict' key"
            assert "optimizer_state_dict" in ckpt, "Missing 'optimizer_state_dict' key"
            assert "best_miou" in ckpt, "Missing 'best_miou' key"

            # Load into fresh model
            model2 = factory(**kwargs).to(device)
            model2.load_state_dict(ckpt["model_state_dict"])
            model2.eval()
            with torch.no_grad():
                out2 = model2(x)
            # Check numeric consistency
            model.eval()
            with torch.no_grad():
                out1 = model(x)
            assert torch.allclose(out1, out2, atol=1e-5), "Reloaded model outputs differ!"
            log_result(f"{name} — checkpoint reload & verify", True, "outputs match ✓")

            # Cleanup
            os.remove(best_path)
            os.remove(last_path)
            os.rmdir(tmp)

        except Exception as e:
            log_result(f"{name} — checkpoint save/reload", False, str(e))


# ── 4. Loss function ─────────────────────────────────────────────────────────
def test_loss():
    print("\n═══ Loss Function ═══")
    try:
        criterion = CrossEntropyDiceLoss(ignore_index=255)
        logits = torch.randn(2, 19, 64, 64, requires_grad=True)  # (B, C, H, W)
        target = make_target(2, 64, 64, 19)
        loss = criterion(logits, target)
        assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
        assert not torch.isnan(loss), "Loss is NaN"
        loss.backward()
        log_result("CrossEntropyDiceLoss — forward+backward", True, f"loss={loss.item():.4f}")
    except Exception as e:
        log_result("CrossEntropyDiceLoss — forward+backward", False, str(e))


# ── 5. Training log / curves save ────────────────────────────────────────────
def test_training_log():
    print("\n═══ Training Log / Curves ═══")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        tmp = tempfile.mkdtemp()
        history = {
            "train_loss": [0.9, 0.7, 0.5],
            "val_loss": [1.0, 0.8, 0.6],
            "val_miou": [0.1, 0.2, 0.3],
        }
        log_path = os.path.join(tmp, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)
        assert os.path.isfile(log_path), "training_log.json not created"
        log_result("training_log.json save", True)

        curve_path = os.path.join(tmp, "training_curves.png")
        epochs_range = range(1, 4)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(epochs_range, history["train_loss"], label="Train Loss")
        ax1.plot(epochs_range, history["val_loss"], label="Val Loss")
        ax1.legend()
        ax2.plot(epochs_range, history["val_miou"], label="Val mIoU", color="green")
        ax2.legend()
        fig.savefig(curve_path, dpi=150)
        plt.close(fig)
        assert os.path.isfile(curve_path)
        assert os.path.getsize(curve_path) > 0
        log_result("training_curves.png save", True)

        # cleanup
        os.remove(log_path)
        os.remove(curve_path)
        os.rmdir(tmp)
    except Exception as e:
        log_result("Training log/curves", False, str(e))


# ── 6. Data loader (Cityscapes) ──────────────────────────────────────────────
def test_dataloader():
    print("\n═══ Data Loader ═══")
    from data.loaders.cityscapes import CityscapesDataset
    try:
        ds = CityscapesDataset(root="./data", split="val")
        if len(ds) == 0:
            log_result("Cityscapes val loader", False, "Dataset is empty — check paths")
            return
        sample = ds[0]
        img = sample['img']
        mask = sample['mask']
        assert img.dim() == 3 and img.shape[0] == 3, f"Expected (3,H,W), got {img.shape}"
        assert mask.dim() == 2, f"Expected (H,W), got {mask.shape}"
        # Check value ranges
        assert img.min() >= 0.0 and img.max() <= 1.0, f"Image range [{img.min()}, {img.max()}], expected [0,1]"
        valid_mask = mask[mask != 255]
        if valid_mask.numel() > 0:
            assert valid_mask.min() >= 0 and valid_mask.max() < 19, f"Mask range [{valid_mask.min()}, {valid_mask.max()}], expected [0,19)"
        log_result(f"Cityscapes val loader", True, f"{len(ds)} samples, img={img.shape}, mask={mask.shape}")
    except Exception as e:
        log_result("Cityscapes val loader", False, str(e))


# ── 7. Full mini-training loop (1 epoch, 1 batch) ───────────────────────────
def test_mini_train():
    print("\n═══ Mini Training Loop (1 epoch, 1 batch) ═══")
    from methods.deep.unet import UNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 19
    B, C, H, W = 2, 3, 64, 64

    try:
        model = UNet(n_channels=C, n_classes=num_classes, encoder_name="resnet34", encoder_weights="imagenet").to(device)
        criterion = CrossEntropyDiceLoss(ignore_index=255)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = make_batch(B, C, H, W).to(device)
        target = make_target(B, H, W, num_classes).to(device)

        # Training step
        model.train()
        out = model(x)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_out = model(x)
        pred_class = val_out.argmax(dim=1)
        _, miou = compute_iou(pred_class, target, num_classes=num_classes, ignore_index=255)
        acc = compute_pixel_accuracy(pred_class, target, ignore_index=255)

        # Save checkpoint
        tmp = tempfile.mkdtemp()
        weights_dir = os.path.join(tmp, "weights")
        os.makedirs(weights_dir)

        torch.save(model.state_dict(), os.path.join(weights_dir, "best.pt"))
        torch.save({
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_miou": miou.item(),
        }, os.path.join(weights_dir, "last.pt"))

        history = {"train_loss": [loss.item()], "val_loss": [loss.item()], "val_miou": [miou.item()]}
        with open(os.path.join(tmp, "training_log.json"), "w") as f:
            json.dump(history, f, indent=2)

        # Verify all saved files
        assert os.path.isfile(os.path.join(weights_dir, "best.pt"))
        assert os.path.isfile(os.path.join(weights_dir, "last.pt"))
        assert os.path.isfile(os.path.join(tmp, "training_log.json"))

        log_result("Mini train loop (UNet)", True,
                   f"loss={loss.item():.4f} mIoU={miou.item():.4f} acc={acc.item():.4f}")

        # Cleanup
        import shutil
        shutil.rmtree(tmp)

    except Exception as e:
        log_result("Mini train loop (UNet)", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  semseg-benchmark — Full Validation Suite")
    print("=" * 60)

    test_classical()
    test_ml()
    test_loss()
    test_deep()
    test_training_log()
    test_dataloader()
    test_mini_train()

    # ── Summary ──
    total = len(results)
    passed = sum(1 for _, p, _ in results if p)
    failed = sum(1 for _, p, _ in results if not p)

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("\nFailed tests:")
        for name, p, msg in results:
            if not p:
                print(f"  ✗ {name}: {msg}")
        sys.exit(1)
    else:
        print("\n  All tests passed! 🎉")
        sys.exit(0)
