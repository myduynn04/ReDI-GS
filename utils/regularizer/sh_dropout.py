# ============================================================
# [CRSGaussian Track B] Per-Gaussian Dropout Regularization
# File: utils/regularizer/sh_dropout.py  (TẠO MỚI)
# Mục đích:
#   Tính per-Gaussian keep mask cho dropout regularization.
#   Target culprit đã identify ở Track A: higher-order SH (_features_rest)
#   overfit trong sparse-view setting. Dropout buộc model không phụ thuộc
#   vào tập Gaussian cụ thể nào → giảm memorization.
#
# 3 modes (master switch trong arguments):
#   "uniform" — B1: constant drop prob cho mọi Gaussian (reproduce Co-Adapt)
#   "sh_norm" — B3: drop theo ||_features_rest|| (high = suspect view-dep overfit)
#   "hybrid"  — B4: kết hợp CRS (position signal) + sh_norm (color signal)
#
# Tham khảo:
#   - Co-Adaptation-of-3DGS/gaussian_renderer/__init__.py:40-45 (Bernoulli mask)
#   - Co-Adaptation-of-3DGS/gaussian_renderer/__init__.py:98-108 (index tensors)
#
# Được gọi từ:
#   gaussian_renderer/__init__.py — trong render() khi pipe.use_dropout=True
#   và disable_dropout=False và iter >= dropout_start_iter.
#
# Eval / pseudo / GUI path đều pass disable_dropout=True → skip hoàn toàn.
# ============================================================

import torch


def compute_dropout_mask(
    gaussians,
    mode: str = "uniform",
    base: float = 0.1,
    w_crs: float = 0.0,
    w_sh: float = 0.0,
    max_drop: float = 0.6,
    quantile: float = 0.95,
):
    """[CRSGaussian Track B] Compute per-Gaussian keep mask.

    Args:
        gaussians: GaussianModel (CRSGaussian version, có get_crs property).
        mode: "uniform" | "sh_norm" | "hybrid".
        base: drop prob sàn cho tất cả Gaussian (constant floor).
        w_crs: trọng số cho (1 - CRS) — Gaussian CRS thấp drop nhiều hơn.
        w_sh: trọng số cho ||_features_rest|| normalized — Gaussian có
              SH-rest lớn drop nhiều hơn (suspect view-dep memorization).
        max_drop: cap drop prob để luôn còn ≥ (1-max_drop) cơ hội render
                  → tránh trường hợp drop hết mọi Gaussian.
        quantile: percentile dùng làm reference cho sh_norm normalize.
                  p95 robust với outlier (1 Gaussian lỗi lớn không skew toàn bộ).

    Returns:
        keep_mask: (N,) bool tensor. True = giữ Gaussian, False = drop.
        drop_prob: (N,) float tensor. Diagnostic để log TB/console.
    """
    N = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device

    # Luôn có sàn = base (có thể 0 cho B0' hoặc >0 cho B1/B3/B4)
    drop_prob = torch.full((N,), base, device=device)

    if mode == "uniform":
        # B1: chỉ dùng sàn, không thêm signal. Reproduce Co-Adapt.
        pass

    elif mode in ("sh_norm", "hybrid"):
        if w_crs > 0:
            # CRS component (position signal): Gaussian có CRS thấp = bị nghi
            # là floater → drop nhiều hơn. CRS đã trong [0,1] qua sigmoid.
            # (1 - CRS) ∈ [0,1]: CRS=0.9 → contribution 0.1*w_crs;
            # CRS=0.2 → contribution 0.8*w_crs.
            crs = gaussians.get_crs.squeeze(-1)  # (N,) in [0,1]
            drop_prob = drop_prob + w_crs * (1.0 - crs)

        if w_sh > 0:
            # SH-norm component (color signal): Gaussian có ||_features_rest||
            # cao → đã học nhiều correction view-dependent → suspect memorize
            # 3 training views. Drop để buộc phân tán responsibility.
            # shape (N, K, 3) với K = (max_sh_degree+1)²-1. flatten(1) → (N, K*3).
            sh_rest = gaussians._features_rest.flatten(1)
            sh_norm = sh_rest.norm(dim=1)  # (N,)

            # Robust normalize dùng quantile thay vì max:
            # - max dễ bị outlier 1 Gaussian skew → toàn bộ /max ≈ 0
            # - p95 bỏ qua top 5% cực trị → normalize ổn định.
            # clamp(min=1e-6) tránh div-by-zero khi sh_norm toàn 0 (iter đầu).
            sh_ref = sh_norm.quantile(quantile).clamp(min=1e-6)
            sh_normalized = (sh_norm / sh_ref).clamp(0.0, 1.0)
            drop_prob = drop_prob + w_sh * sh_normalized

    else:
        raise ValueError(f"[Track B] Unknown dropout mode: {mode!r}")

    # Cap để luôn giữ ≥ (1-max_drop) Gaussians/iter → tránh collapse rendering.
    # max_drop=0.6 nghĩa là drop prob tối đa 60%, luôn còn ≥40% để render.
    drop_prob = drop_prob.clamp(0.0, max_drop)

    # Bernoulli: rand > p → giữ với xác suất (1-p). True = keep, False = drop.
    keep_mask = torch.rand_like(drop_prob) > drop_prob

    return keep_mask, drop_prob


def log_dropout_stats(drop_prob, keep_mask, tb_writer, iteration, tag="dropout"):
    """[CRSGaussian Track B] Log drop rate distribution — diagnostic only.

    Được gọi từ train.py mỗi 500 iter khi use_dropout=True.
    drop_prob: (N,) float tensor từ compute_dropout_mask.
    keep_mask: (N,) bool tensor từ compute_dropout_mask.
    """
    if tb_writer is None:
        return
    actual_drop_rate = 1.0 - keep_mask.float().mean().item()
    tb_writer.add_scalar(f'{tag}/actual_drop_rate', actual_drop_rate, iteration)
    tb_writer.add_scalar(f'{tag}/prob_mean', drop_prob.mean().item(), iteration)
    tb_writer.add_scalar(f'{tag}/prob_p50', drop_prob.quantile(0.50).item(), iteration)
    tb_writer.add_scalar(f'{tag}/prob_p95', drop_prob.quantile(0.95).item(), iteration)
