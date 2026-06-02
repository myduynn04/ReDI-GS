# ============================================================
# [CRSGaussian Plug-in A2.2] DropAnSH adapted cho splatfacto
# File: crsgaussian_plugin/dropansh.py  (KEEP LOCAL — upload server)
#
# Port từ utils/regularizer/dropansh.py của CRSGaussian, adapt naming:
#   - CRSGaussian:        gaussians.get_xyz, _features_rest, _opacity
#   - splatfacto: gauss_params["means"], ["features_rest"], ["opacities"]
#
# 2 cơ chế:
#   (1) Anchor dropout — drop cluster (anchor + k-NN) → ép Gaussian không phụ
#       thuộc vùng cụ thể (KHÁC B1 dropout đều)
#   (2) SH degree dropout — zero _features_rest above lmax, progressive
#       schedule lmax 0→1→2 → coarse-to-fine appearance
#
# Implementation note cho splatfacto:
#   - Anchor dropout: tạm thời set opacity = -1e5 (logit) → sigmoid ≈ 0 → không
#     render. Snapshot + restore sau forward.
#   - SH dropout: zero gauss_params["features_rest"][drop_mask, num_keep:]
#     in-place, snapshot + restore.
# ============================================================
"""[CRSGaussian Plug-in A2.2] DropAnSH cho Nerfstudio splatfacto."""

from typing import Optional, Tuple

import torch


def anchor_dropout_keep_mask(
    means: torch.Tensor,
    iteration: int,
    total_iter: int = 10000,
    k: int = 10,
    pa_max: float = 0.02,
    batch_threshold: int = 200_000_000,
) -> torch.Tensor:
    """[CRSGaussian Plug-in A2.2] Anchor dropout — return keep_mask (N,).

    Linear ramp pa = pa_max * min(1, iter/total) → num_anchors.
    Chọn anchor uniform random, drop anchor + k-NN.

    Args:
        means: (N, 3) Gaussian positions (= splatfacto gauss_params["means"]).
        iteration: current train step.
        total_iter: tổng iter training cho linear ramp.
        k: số nearest neighbors / anchor.
        pa_max: anchor sample rate max (default 0.02 = 2%).
        batch_threshold: batch cdist nếu num_anchors × N quá lớn.

    Returns:
        keep_mask: (N,) bool. True = keep (render), False = drop.
    """
    N = means.shape[0]
    device = means.device

    if pa_max <= 0.0:
        return torch.ones(N, device=device, dtype=torch.bool)

    pa = pa_max * min(1.0, iteration / max(total_iter, 1))
    num_anchors = max(1, int(pa * N))
    num_anchors = min(num_anchors, N)

    anchor_idx = torch.randperm(N, device=device)[:num_anchors]
    anchor_pos = means[anchor_idx]

    effective_k = min(k + 1, N)
    if num_anchors * N > batch_threshold:
        batch_size = max(1, batch_threshold // N)
        nn_chunks = []
        for i in range(0, num_anchors, batch_size):
            d = torch.cdist(anchor_pos[i:i + batch_size], means)
            _, idx = torch.topk(d, k=effective_k, largest=False)
            nn_chunks.append(idx)
        nn_idx = torch.cat(nn_chunks, dim=0)
    else:
        dists = torch.cdist(anchor_pos, means)
        _, nn_idx = torch.topk(dists, k=effective_k, largest=False)

    drop_set = nn_idx.flatten().unique()
    keep_mask = torch.ones(N, device=device, dtype=torch.bool)
    keep_mask[drop_set] = False
    return keep_mask


def sh_degree_dropout_inplace(
    features_rest: torch.Tensor,
    iteration: int,
    p_sh: float = 0.2,
    schedule: Tuple[int, int, int] = (2000, 4000, 6000),
) -> Optional[Tuple]:
    """[CRSGaussian Plug-in A2.2] SH degree dropout — modify features_rest IN-PLACE.

    Progressive schedule:
        iter < schedule[0]:              lmax=0  (zero all rest)
        schedule[0] ≤ iter < schedule[1]: lmax=1  (keep 3 first)
        schedule[1] ≤ iter < schedule[2]: lmax=2  (keep 8 first)
        iter ≥ schedule[2]:              no dropout (return None)

    Args:
        features_rest: (N, M, 3) splatfacto gauss_params["features_rest"].
        iteration: current train step.
        p_sh: per-Gaussian probability of dropout (default 0.2).
        schedule: 3 iter checkpoints.

    Returns:
        snapshot: None (no-op) hoặc (drop_mask, num_keep_rest, saved_tensor)
                  để restore sau forward.
    """
    if p_sh <= 0.0 or iteration >= schedule[2]:
        return None

    if iteration < schedule[0]:
        lmax = 0
    elif iteration < schedule[1]:
        lmax = 1
    else:
        lmax = 2

    N = features_rest.shape[0]
    device = features_rest.device

    drop_mask = torch.rand(N, device=device) < p_sh
    if not drop_mask.any():
        return None

    # (lmax+1)^2 - 1: lmax=0→0, lmax=1→3, lmax=2→8
    num_keep_rest = max(0, (lmax + 1) ** 2 - 1)
    rest_shape = features_rest.shape[1]

    if num_keep_rest >= rest_shape:
        return None

    saved = features_rest.data[drop_mask, num_keep_rest:, :].clone()
    features_rest.data[drop_mask, num_keep_rest:, :] = 0.0
    return (drop_mask, num_keep_rest, saved)


def restore_sh_dropout(features_rest: torch.Tensor, snapshot: Optional[Tuple]):
    """[CRSGaussian Plug-in A2.2] Restore features_rest sau forward."""
    if snapshot is None:
        return
    drop_mask, num_keep_rest, saved = snapshot
    features_rest.data[drop_mask, num_keep_rest:, :] = saved
