# ============================================================
# [CRSGaussian Phase 15] Anisotropy / shape regularizer.
# File: utils/regularizer/shape_reg.py  (TẠO MỚI)
# Mục đích: penalty liên tục lên anisotropy Gaussian (gap thật: A3
#   chỉ có lfcf_diffscale isotropify LÚC densify — verified lfcf.py:8,
#   148,224 — KHÔNG loss liên tục cho Gaussian đang tồn tại).
#
# 2 formulation (Phase-15 3-arm, user "test cả 3"):
#   mode="blunt"       (A): L = mean(log s_max − log s_min)
#                        → phạt RATIO mọi Gaussian. log-space (ratio A3
#                        median≈344, max≈1e8 — raw sẽ nổ). = CONTROL
#                        falsify Q4-diagnostic (Q4: 77% high-aniso là
#                        legitimate-flat → blunt predicted hại).
#   mode="smax_excess" (B): L = mean(relu(log s_max − τ)),
#                        τ = median(log s_max) + k·MAD  (self-calibrate
#                        per-call, detached → KHÔNG hằng số bịa).
#                        → CHỈ phạt s_max phình bất thường vs scene-
#                        normal (≈2-21% stretch/extreme Q4), KHÔNG đụng
#                        s_min-collapsed (77% surface-flat HỢP LỆ). Form
#                        data Q4 CHỈ vào.
#
# Được gọi từ: train.py (gated opt.use_shape_reg, sau
#   opt.shape_reg_start_iter). Default OFF → A3 byte-identical.
# Verified-from-code: gaussian_model.get_scaling=exp(_scaling) (N,3,
#   differentiable; log(get_scaling) → grad chảy vào _scaling). KHÔNG
#   đụng production khác; standalone module (Quy tắc 12).
# ============================================================
"""[CRSGaussian Phase 15] Anisotropy shape regularizer (blunt | smax_excess)."""

import torch

EPS = 1e-8


def compute_shape_reg_loss(gaussians, mode="blunt", smax_k=2.0):
    """Differentiable anisotropy penalty.

    Args:
        gaussians: GaussianModel (get_scaling = exp(_scaling), (N,3)).
        mode: "blunt" (A, ratio) | "smax_excess" (B, targeted).
        smax_k: B-only — τ = median(log_smax) + smax_k·MAD(log_smax).
    Returns:
        scalar tensor (có grad) hoặc None nếu N<10.
    """
    s = gaussians.get_scaling                       # (N,3) >0, differentiable
    if s.shape[0] < 10:
        return None
    log_s = torch.log(s + EPS)                      # grad → _scaling
    log_smax = log_s.max(dim=1).values              # (N,)
    log_smin = log_s.min(dim=1).values              # (N,)

    if mode == "blunt":
        # A: phạt log-ratio mọi Gaussian (control falsify Q4).
        return (log_smax - log_smin).mean()

    if mode == "smax_excess":
        # B: chỉ phạt s_max vượt scene-normal (median+k·MAD), detach
        # thống kê (threshold không nhận grad — chỉ relu-excess nhận).
        with torch.no_grad():
            med = log_smax.median()
            mad = (log_smax - med).abs().median()
            tau = med + smax_k * mad
        return torch.relu(log_smax - tau).mean()

    raise ValueError(f"shape_reg mode không hợp lệ: {mode} "
                     f"(blunt | smax_excess)")
