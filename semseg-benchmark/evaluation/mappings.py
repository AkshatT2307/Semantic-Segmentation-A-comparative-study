import torch

def map_clusters_to_classes(pred, target, ignore_index=255):
    """
    Implements Majority Voting (Cluster-to-Class Assignment) for evaluating unsupervised
    segmentation techniques.
    
    For each arbitrary predicted cluster region in 'pred', this function finds the most 
    frequent ground truth semantic class inside 'target' for that same region, 
    and recasts the entire prediction cluster to that semantic class.
    
    Args:
        pred: (B, H, W) arbitrary generated mapping clusters (e.g. from KMeans or GraphCut).
        target: (B, H, W) semantic ground truth mapping index classes.
        ignore_index: Integer index to ignore mapping against (usually 255).
        
    Returns:
        mapped_preds: (B, H, W) containing proper semantic class integers matching the targets.
    """
    mapped_preds = torch.zeros_like(pred)
    
    for b in range(pred.size(0)):
        unique_clusters = torch.unique(pred[b])
        for c in unique_clusters:
            # Mask out background/ignore_index from ground truth before mode analysis
            valid_mask = (pred[b] == c) & (target[b] != ignore_index)
            
            if valid_mask.sum() > 0:
                # Find the most frequent valid target class overlapping this cluster
                target_pixels = target[b][valid_mask]
                mode_val = torch.mode(target_pixels).values
                mapped_preds[b][pred[b] == c] = mode_val
            else:
                # If a cluster exists purely in the ignore region, map it to ignore
                mapped_preds[b][pred[b] == c] = ignore_index
                
    return mapped_preds
