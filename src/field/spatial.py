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

    # Compute rank within each position — vectorized
    # pos_change marks where a new group starts
    pos_change = torch.cat([torch.tensor([True], device=device), sorted_pos[1:] != sorted_pos[:-1]])
    # cumcount: within each group of same position, count 0, 1, 2, ...
    # Trick: cumsum of ones, minus cumsum that resets at boundaries
    arange = torch.arange(len(sorted_pos), device=device)
    # group_starts_idx[i] = index of the first element in this group
    group_cumsum = pos_change.long().cumsum(0)  # 1-indexed group id
    # scatter the index of first occurrence per group
    n_groups = int(group_cumsum.max())
    first_idx = torch.zeros(n_groups + 1, dtype=torch.long, device=device)
    # Reverse iterate so smallest index wins
    rev = torch.arange(len(sorted_pos) - 1, -1, -1, device=device)
    first_idx[group_cumsum[rev]] = rev
    cumcount = arange - first_idx[group_cumsum]

    # Only keep top n_features per position
    keep = cumcount < n_features
    kept_pos = sorted_pos[keep]
    kept_acts = sorted_acts[keep]
    kept_feat = cumcount[keep]

    amap[kept_feat.long(), kept_pos.long()] = kept_acts
    return amap.view(1, n_features, grid_h, grid_w)


def extract_patches_batch(feature_maps: torch.Tensor, patch_size: int = 5,
                          stride: int = 1) -> torch.Tensor:
    """Batch patch extraction: (B, C, H, W) -> (B, n_patches, patch_dim).

    One CUDA kernel for the entire batch.
    """
    # F.unfold: (B, C, H, W) -> (B, C*k*k, n_patches)
    patches = F.unfold(feature_maps, kernel_size=patch_size, stride=stride)
    return patches.permute(0, 2, 1)  # (B, n_patches, patch_dim)


def build_activation_map_batch(positions: torch.Tensor, activations: torch.Tensor,
                               grid_h: int, grid_w: int) -> torch.Tensor:
    """Batch scatter: take max activation per position per image.

    Args:
        positions: (n_neurons, 2) of (row, col) — shared across batch
        activations: (B, n_neurons)
        grid_h, grid_w: output grid dimensions

    Returns:
        (B, 1, grid_h, grid_w) — one feature channel (max per position)
    """
    B = activations.shape[0]
    device = activations.device
    n_positions = grid_h * grid_w

    flat_pos = positions[:, 0] * grid_w + positions[:, 1]  # (n_neurons,)
    flat_pos = flat_pos.clamp(0, n_positions - 1)

    # Expand flat_pos for batch: (B, n_neurons)
    flat_pos_batch = flat_pos.unsqueeze(0).expand(B, -1)

    # Scatter max: (B, n_positions)
    output = torch.zeros(B, n_positions, device=device)
    output.scatter_reduce_(1, flat_pos_batch, activations, reduce='amax', include_self=True)

    return output.view(B, 1, grid_h, grid_w)
