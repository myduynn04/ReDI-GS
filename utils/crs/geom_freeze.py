# ============================================================
# [CRSGaussian Phase 26 A1] CRS-modulated geometric freeze
# File: utils/crs/geom_freeze.py (NEW)
# Mục đích: Per-Gaussian freeze cho _xyz/_scaling/_rotation, đối xứng với
#           apply_crs_modulated_sh_freeze (sh_freeze.py) vốn chỉ freeze SH.
#           V thấp (hình học đang trôi dạt/méo bất thường) → zero grad vị
#           trí+hình dạng → Gaussian ngừng di chuyển/biến dạng, buộc phải
#           "chứng minh" ổn định trở lại (V tăng) trước khi được học tiếp.
#           Mục tiêu: chặn hiện tượng 1 Gaussian méo dùng SH bậc cao để
#           "diễn" 2 màu cạnh nhau thay vì tách thành 2 Gaussian.
# Được gọi từ: train.py SAU loss.backward(), TRƯỚC optimizer.step()
#              (cùng vị trí apply_crs_modulated_sh_freeze).
# ============================================================

import torch


def apply_crs_modulated_geom_freeze(
    gaussians,
    iter: int,
    freeze_start: int = 1000,
    tau_freeze: float = 0.5,
):
    """Zero out _xyz/_scaling/_rotation gradient cho Gaussians có V_stability thấp.

    Idempotent, cùng cơ chế apply_crs_modulated_sh_freeze: chỉ zero grad tại
    iter hiện tại, không set lr=0 vĩnh viễn — Gaussian có thể "thoát" freeze
    ở iter sau nếu V phục hồi.

    Args:
        gaussians: GaussianModel với .get_v_stability property (thêm ở
            gaussian_model.py hoặc truyền trực tiếp qua compute_V_stability)
            + ._xyz/._scaling/._rotation .grad.
        iter: current iteration. Skip nếu iter ≤ freeze_start.
        freeze_start: iter bắt đầu apply. Default 1000 (khớp T_warmup).
        tau_freeze: V threshold (< tau → freeze). Default 0.5.
    """
    if iter <= freeze_start:
        return

    if gaussians._xyz.grad is None:
        return

    from utils.crs.v_stability import compute_V_stability
    V = compute_V_stability(gaussians).detach().squeeze(-1)  # (N,) ∈ [0, 1]

    if V.shape[0] != gaussians._xyz.shape[0]:
        return

    freeze_mask = V < tau_freeze  # (N,) bool
    if not freeze_mask.any():
        return

    gaussians._xyz.grad[freeze_mask] = 0
    if gaussians._scaling.grad is not None:
        gaussians._scaling.grad[freeze_mask] = 0
    if gaussians._rotation.grad is not None:
        gaussians._rotation.grad[freeze_mask] = 0
