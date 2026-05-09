# ============================================================
# [CRSGaussian Density-aware Stage A] Covariance overlap density
# File: utils/regularizer/density_covariance.py  (TẠO MỚI)
# Mục đích: Shape-aware density — sum Bhattacharyya overlap với k spatial
#           neighbors. Gaussian overlap nhiều neighbors → density cao.
# Được gọi từ: anchor_dropout_mask() khi density_method="covariance"
#
# Không dùng pytorch3d (không available trong codebase). Dùng torch.cdist
# + topk giống anchor_dropout_mask hiện tại để tìm k-NN.
# ============================================================

import torch


@torch.no_grad()
def compute_covariance_density(xyz: torch.Tensor,
                                scaling: torch.Tensor,
                                rotation: torch.Tensor,
                                k: int = 10,
                                batch_size: int = 10000) -> torch.Tensor:
    """[CRSGaussian Density-aware] Covariance overlap density per Gaussian.

    Bhattacharyya overlap giữa 2 Gaussian (μ_i, Σ_i) và (μ_j, Σ_j):
        D_B = (1/8) * Δμ^T * Σ_mean^{-1} * Δμ + (1/2) * ln(|Σ_mean|/sqrt(|Σ_i||Σ_j|))
        overlap = exp(-D_B)

    Density per Gaussian = sum overlap với k nearest neighbors (spatial).

    Args:
        xyz:      (N, 3) positions.
        scaling:  (N, 3) pre-activation scale (sẽ exp).
        rotation: (N, 4) quaternion.
        k:        số spatial neighbors.
        batch_size: batch Bhattacharyya compute tránh OOM.

    Returns:
        density: (N,) float tensor [0, 1]. 1 = overlap cao nhất.
    """
    from utils.general_utils import build_rotation

    N = xyz.shape[0]
    device = xyz.device
    if N == 0:
        return torch.zeros(0, device=device)

    # ── Step 1: k spatial neighbors per Gaussian ──
    # Dùng torch.cdist + topk (same pattern như anchor_dropout_mask).
    # Batch cdist nếu N lớn để tránh OOM (N × N float matrix).
    effective_k = min(k + 1, N)  # +1 vì sẽ skip self
    if N > 20000:
        # Batched knn để tránh OOM
        nn_chunks = []
        for b_start in range(0, N, batch_size):
            b_end = min(b_start + batch_size, N)
            d = torch.cdist(xyz[b_start:b_end], xyz)           # (B, N)
            _, idx = torch.topk(d, k=effective_k, largest=False)
            nn_chunks.append(idx[:, 1:])                        # skip self (idx 0)
        neighbor_idx = torch.cat(nn_chunks, dim=0)              # (N, k)
    else:
        dists = torch.cdist(xyz, xyz)                           # (N, N)
        _, idx = torch.topk(dists, k=effective_k, largest=False)
        neighbor_idx = idx[:, 1:]                               # (N, k)

    # ── Step 2: build full 3D covariance per Gaussian ──
    # Σ = R * diag(s^2) * R^T  where s = exp(scaling)
    scale_activated = torch.exp(scaling)                        # (N, 3)
    R = build_rotation(rotation)                                # (N, 3, 3)
    S = torch.zeros(N, 3, 3, device=device)
    S[:, 0, 0] = scale_activated[:, 0] ** 2
    S[:, 1, 1] = scale_activated[:, 1] ** 2
    S[:, 2, 2] = scale_activated[:, 2] ** 2
    cov = R @ S @ R.transpose(-1, -2)                           # (N, 3, 3)

    # Per-Gaussian log-det (precompute để khỏi tính lại trong loop)
    eye3 = 1e-8 * torch.eye(3, device=device)
    log_det = torch.logdet(cov + eye3)                          # (N,)

    # ── Step 3: pairwise Bhattacharyya batched ──
    density_raw = torch.zeros(N, device=device)

    for b_start in range(0, N, batch_size):
        b_end = min(b_start + batch_size, N)
        B = b_end - b_start

        xyz_i = xyz[b_start:b_end]                              # (B, 3)
        cov_i = cov[b_start:b_end]                              # (B, 3, 3)
        log_det_i = log_det[b_start:b_end]                      # (B,)
        nbr_idx = neighbor_idx[b_start:b_end]                   # (B, k)

        xyz_j     = xyz[nbr_idx]                                # (B, k, 3)
        cov_j     = cov[nbr_idx]                                # (B, k, 3, 3)
        log_det_j = log_det[nbr_idx]                            # (B, k)

        # Mean covariance: Σ_mean = (Σ_i + Σ_j) / 2
        cov_mean = (cov_i.unsqueeze(1) + cov_j) / 2.0           # (B, k, 3, 3)

        # Mahalanobis-like: Δμ^T Σ_mean^{-1} Δμ
        diff = (xyz_i.unsqueeze(1) - xyz_j).unsqueeze(-1)       # (B, k, 3, 1)
        cov_mean_inv = torch.inverse(cov_mean + eye3)           # (B, k, 3, 3)
        mahal = (diff.transpose(-1, -2) @ cov_mean_inv @ diff).squeeze(-1).squeeze(-1)  # (B, k)

        # Log-det ratio
        log_det_mean = torch.logdet(cov_mean + eye3)            # (B, k)
        log_det_geo = 0.5 * (log_det_i.unsqueeze(1) + log_det_j)  # (B, k)

        # Bhattacharyya distance
        DB = 0.125 * mahal + 0.5 * (log_det_mean - log_det_geo)  # (B, k)
        DB = torch.clamp(DB, min=0.0)                            # numerical floor

        # Overlap = exp(-DB), sum over k neighbors
        overlap = torch.exp(-DB).sum(dim=1)                      # (B,)
        density_raw[b_start:b_end] = overlap

    # Normalize [0, 1]
    density = density_raw / density_raw.max().clamp(min=1e-6)
    return density
