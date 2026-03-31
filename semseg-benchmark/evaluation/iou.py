import torch

def compute_iou(pred, target, num_classes=2, ignore_index=255):
    """
    Computes Intersection over Union (IoU) per class and mean IoU.
    
    Args:
        pred: Tensor of shape (B, H, W) containing integer class predictions.
        target: Tensor of shape (B, H, W) containing integer ground truth classes.
        num_classes: Number of classes.
        ignore_index: Target index to ignore (typically 255).
        
    Returns:
        ious: Tensor of IoU for each class.
        miou: Mean IoU over all valid classes.
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Filter out ignored indices
    valid_mask = target != ignore_index
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    ious = torch.zeros(num_classes)
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious[cls] = float('nan')  # Ignore if class is not present in both
        else:
            ious[cls] = intersection / union
            
    # Compute mean IoU ignoring NaNs (classes that were not present in union)
    valid_ious = ious[~torch.isnan(ious)]
    miou = valid_ious.mean() if len(valid_ious) > 0 else torch.tensor(0.0)
    
    return ious, miou
