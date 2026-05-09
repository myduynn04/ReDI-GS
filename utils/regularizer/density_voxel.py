# ============================================================
# [CRSGaussian Density-aware Stage A] Voxel binning density
# File: utils/regularizer/density_voxel.py  (TẠO MỚI)
# Mục đích: Count Gaussians per voxel → normalized density per-Gaussian [0,1]
# Được gọi từ: anchor_dropout_mask() khi density_method="voxel"
# ============================================================

import torch


@torch.no_grad()
def compute_voxel_density(xyz: torch.Tensor, voxel_size: float = None) -> torch.Tensor:
    """[CRSGaussian Density-aware] Voxel binning density per Gaussian.

    Voxel_size auto-chosen: target ~5-10 Gaussians per voxel at median density.
    Density = count of Gaussians trong cùng voxel, normalized max → 1.

    Args:
        xyz: (N, 3) Gaussian positions (CUDA tensor).
        voxel_size: scalar float. None → auto-compute từ scene extent.

    Returns:
        density: (N,) float tensor trong [0, 1]. 1 = densest.
    """
    N = xyz.shape[0]
    if N == 0:
        return torch.zeros(0, device=xyz.device)

    # Auto voxel size: chia scene extent theo cube-root N × 1.5 factor
    # Heuristic: target ~5-10 Gaussians per voxel tại median density
    if voxel_size is None:
        scene_extent = (xyz.max(0).values - xyz.min(0).values).mean().item()
        voxel_size = (scene_extent / (N ** (1.0 / 3.0))) * 1.5

    # Normalize xyz → voxel indices (long)
    xyz_min = xyz.min(0).values
    voxel_idx = ((xyz - xyz_min) / voxel_size).floor().long()   # (N, 3)

    # Hash 3D voxel index → 1D key (tránh dict, dùng torch.unique)
    max_dim = voxel_idx.max(0).values + 1
    hash_key = (voxel_idx[:, 0] * max_dim[1] * max_dim[2]
                + voxel_idx[:, 1] * max_dim[2]
                + voxel_idx[:, 2])

    # Count per voxel via unique + inverse_idx broadcast
    _unique_keys, inverse_idx, counts = torch.unique(
        hash_key, return_inverse=True, return_counts=True
    )

    # Density raw per Gaussian = count trong voxel của nó
    density_raw = counts[inverse_idx].float()              # (N,)

    # Normalize [0, 1] — max = densest voxel
    density = density_raw / density_raw.max().clamp(min=1.0)
    return density
