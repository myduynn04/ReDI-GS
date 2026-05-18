# ============================================================
# [CRSGaussian Phase 13.2.5] GDAGS coherence weight.
# File: utils/densify/gdags.py  (TẠO MỚI)
# Mục đích: Gradient-Direction-Aware Density Control (ICLR 2026)
#           coherence-weighted asymmetric clone/split threshold.
# Verified EXACT từ GDAGS/scene/gaussian_model.py:526-527 (KHÔNG đoán):
#     consistency = (grads + 1e-8) / (grads_abs + 1e-8)
#     weight      = 0.8 + 25 * (1 - consistency) ** 15
#     → clone dùng grads/weight (coherent ⇒ weight≈0.8 ⇒ threshold thấp
#       ⇒ clone NHIỀU); split dùng grads*weight (conflict ⇒ weight lớn
#       ⇒ split policy theo độ mâu thuẫn).
# Được gọi từ: scene/gaussian_model.py densify_and_prune STANDARD path,
#              CHỈ khi self.use_gdags=True (gate mirror absdensify).
# KHÔNG đụng LFCF path. Default OFF → A3 byte-identical.
#
# Note (verify-from-code): GDAGS:527 KHÔNG clamp consistency. Theo bất
# đẳng thức tam giác grads=‖Σ∇‖ ≤ Σ‖∇‖=grads_abs ⇒ consistency ≤ 1 by
# construction (eps 1e-8 ảnh hưởng không đáng kể). Mirror EXACT source
# (no clamp) để A/B trung thực với method gốc — KHÔNG tự thêm guard
# không có trong GDAGS (tránh deviate không cơ sở).
# ============================================================
"""[CRSGaussian Phase 13.2.5] GDAGS coherence weight (verified GDAGS:526-527)."""

import torch

GDAGS_W0 = 0.8
GDAGS_K = 25.0
GDAGS_POW = 15


def compute_gdags_weight(grads, grads_abs):
    """EXACT mirror GDAGS/scene/gaussian_model.py:526-527.

    Args:
        grads:     (N,1) = xyz_gradient_accum / denom      (= ‖Σ∇‖ standard)
        grads_abs: (N,1) = xyz_gradient_accum_abs / denom  (= Σ‖∇‖ abs-grad)
    Returns:
        weight: (N,1) — clone threshold ÷weight, split threshold ×weight.
    """
    consistency = (grads + 1e-8) / (grads_abs + 1e-8)
    weight = GDAGS_W0 + GDAGS_K * torch.pow(1.0 - consistency, GDAGS_POW)
    return weight
