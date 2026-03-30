"""Spatial ops: patch extraction via F.unfold, activation map construction."""

import torch
import torch.nn.functional as F


def extract_patches(feature_map: torch.Tensor, patch_size: int = 5, stride: int = 1) -> torch.Tensor:
    """Extract all patches from a 2D feature map.
    Args: feature_map (1, C, H, W), patch_size, stride
    Returns: (n_patches, C * patch_size^2)
    """
    patches = F.unfold(feature_map, kernel_size=patch_size, stride=stride)
    return patches[0].T


def build_activation_map(positions: torch.Tensor, activations: torch.Tensor,
                         n_features: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """Scatter neuron activations into a spatial feature map.
    Multiple neurons at same position → top-k activations kept.
    Args: positions (n, 2) of (row, col), activations (n,), n_features, grid dims
    Returns: (1, n_features, grid_h, grid_w)
    """
    device = activations.device
    amap = torch.zeros(n_features, grid_h * grid_w, device=device)

    flat_pos = positions[:, 0] * grid_w + positions[:, 1]

    # Sort by position, then by activation descending
    sort_key = flat_pos.float() * 1000 - activations
    order = sort_key.argsort()
    sorted_pos = flat_pos[order]
    sorted_acts = activations[order]

    # Compute rank within each position
    pos_change = torch.cat([torch.tensor([True], device=device), sorted_pos[1:] != sorted_pos[:-1]])
    cumcount = torch.zeros(len(sorted_pos), dtype=torch.long, device=device)
    count = 0
    for i in range(len(cumcount)):
        if pos_change[i]:
            count = 0
        cumcount[i] = count
        count += 1

    # Only keep top n_features per position
    keep = cumcount < n_features
    kept_pos = sorted_pos[keep]
    kept_acts = sorted_acts[keep]
    kept_feat = cumcount[keep]

    amap[kept_feat.long(), kept_pos.long()] = kept_acts
    return amap.view(1, n_features, grid_h, grid_w)
