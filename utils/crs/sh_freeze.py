# ============================================================
# [CRSGaussian Phase 8c] CRS-modulated SH freeze
# File: utils/crs/sh_freeze.py (NEW)
# Mục đích: Per-Gaussian SH freeze thay thế global freeze_sh_after.
#           CRS thấp (untrustworthy) → zero _features_rest gradient
#                                   → SH coef không update → Gaussian
#                                     phải tự cải thiện CRS qua geometry/opacity.
#           CRS cao → SH continue learning bình thường.
# Được gọi từ: train.py SAU loss.backward(), TRƯỚC optimizer.step().
# ============================================================

import torch


def apply_crs_modulated_sh_freeze(
    gaussians,
    iter: int,
    freeze_start: int = 1000,
    tau_freeze: float = 0.5,
):
    """Zero out _features_rest gradient cho Gaussians có CRS thấp.

    Mechanism: thay vì global freeze SH (Track A1 freeze_sh_after = freeze ALL
    Gaussians sau iter X), apply selective per-Gaussian. Lý do:
    - High CRS Gaussian (geometry/color đáng tin) → SH tiếp tục optimize chi tiết.
    - Low CRS Gaussian (untrustworthy) → SH freeze → tránh memorize sai.

    Idempotent: gọi mỗi iter sau backward không gây side effect (chỉ zero grad,
    không flag persistent state). Khác freeze_sh() set lr=0 vĩnh viễn.

    Args:
        gaussians: GaussianModel với .get_crs property + ._features_rest.grad.
        iter: current iteration. Skip nếu iter ≤ freeze_start.
        freeze_start: iter bắt đầu apply (default 1000 ≈ T_warmup CRS).
        tau_freeze: CRS threshold (< tau → freeze). Default 0.5.
            Ablate: {0.3, 0.5, 0.7}.
    """
    # Skip iter sớm — CRS chưa stable, freeze theo CRS noise → hại.
    if iter <= freeze_start:
        return

    # Skip nếu chưa backward (grad=None ở iter cuối hoặc init phase).
    if gaussians._features_rest.grad is None:
        return

    crs = gaussians.get_crs.detach().squeeze(-1)  # (N,) ∈ [0, 1]

    # Defensive shape check — CRS shape phải match feature_rest dim 0.
    if crs.shape[0] != gaussians._features_rest.shape[0]:
        return

    freeze_mask = crs < tau_freeze  # (N,) bool — Gaussians cần freeze
    if freeze_mask.any():
        # _features_rest shape [N, K_rest, 3] → mask broadcast lên dim 0.
        gaussians._features_rest.grad[freeze_mask] = 0
