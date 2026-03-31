import torch

def compute_pixel_accuracy(pred, target, ignore_index=255):
    """
    Computes overall Pixel Accuracy.
    
    Args:
        pred: Tensor of shape (B, H, W) containing integer class predictions.
        target: Tensor of shape (B, H, W) containing integer ground truth classes.
        ignore_index: Target index to ignore (typically 255).
        
    Returns:
        accuracy: Float containing overall pixel accuracy.
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Filter out ignored indices
    valid_mask = target != ignore_index
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    total_valid_pixels = target.numel()
    
    if total_valid_pixels == 0:
        return torch.tensor(0.0)
        
    correct_pixels = (pred == target).sum().float()
    accuracy = correct_pixels / total_valid_pixels
    
    return accuracy
