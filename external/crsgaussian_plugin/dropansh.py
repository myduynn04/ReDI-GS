# ============================================================
# [CRSGaussian Path A B3] DropAnSH — anchor + SH degree dropout
# File: crsgaussian_plugin/dropansh.py  (KEEP LOCAL — upload server)
#
# Port từ Phase A nerfstudio_plugin/dropansh.py, adapt cho CoR-GS GaussianModel:
#   - means    → gaussians._xyz
#   - features_rest → gaussians._features_rest
#   - opacity  → gaussians._opacity (logit space)
#
# Apply trong CrsGaussianModel.get_outputs trước khi gọi render:
#   1. Snapshot opacity → set drop_mask vị trí về -1e5 (sigmoid ≈ 0)
#   2. Snapshot features_rest → zero rest above lmax
#   3. Render
#   4. Restore opacity + features_rest (in-place)
# ============================================================
"""[Path A B3] DropAnSH — anchor + SH degree dropout."""

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
    """[Path A B3] Anchor dropout — return keep_mask (N,).

    Linear ramp pa = pa_max * min(1, iter/total) → num_anchors.
    Chọn anchor uniform random, drop anchor + k-NN.
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
    """[Path A B3] SH degree dropout — modify features_rest IN-PLACE.

    Progressive schedule:
        iter < schedule[0]:              lmax=0  (zero all rest)
        schedule[0] ≤ iter < schedule[1]: lmax=1  (keep 3 first)
        schedule[1] ≤ iter < schedule[2]: lmax=2  (keep 8 first)
        iter ≥ schedule[2]:              no dropout (return None)
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

    num_keep_rest = max(0, (lmax + 1) ** 2 - 1)
    rest_shape = features_rest.shape[1]

    if num_keep_rest >= rest_shape:
        return None

    saved = features_rest.data[drop_mask, num_keep_rest:, :].clone()
    features_rest.data[drop_mask, num_keep_rest:, :] = 0.0
    return (drop_mask, num_keep_rest, saved)


def restore_sh_dropout(features_rest: torch.Tensor, snapshot: Optional[Tuple]):
    """[Path A B3] Restore features_rest sau render."""
    if snapshot is None:
        return
    drop_mask, num_keep_rest, saved = snapshot
    features_rest.data[drop_mask, num_keep_rest:, :] = saved


def apply_anchor_dropout_to_opacity(
    opacity_logit: torch.Tensor,
    keep_mask: torch.Tensor,
) -> torch.Tensor:
    """[Path A B3] Apply anchor dropout bằng cách set logit về -1e5 (sigmoid≈0).

    Returns: snapshot tensor để restore sau render.
    """
    snapshot = opacity_logit.data.clone()
    drop_mask = ~keep_mask
    opacity_logit.data[drop_mask] = -1e5
    return snapshot


def restore_opacity(opacity_logit: torch.Tensor, snapshot: torch.Tensor):
    """Restore opacity logit từ snapshot."""
    opacity_logit.data.copy_(snapshot)


__all__ = [
    "anchor_dropout_keep_mask",
    "sh_degree_dropout_inplace",
    "restore_sh_dropout",
    "apply_anchor_dropout_to_opacity",
    "restore_opacity",
]
