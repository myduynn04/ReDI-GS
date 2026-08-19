# ============================================================
# [CRSGaussian Phase 26 A1] Geometric stability (V) signal
# File: utils/crs/v_stability.py (NEW)
# Mục đích: Track EMA mean + variance của _xyz, _scaling, _rotation per
#           Gaussian → phát hiện Gaussian đang "trôi dạt" hình học (vị trí
#           hoặc hình dạng dao động mạnh qua các iteration), khác S_stability
#           (sh_stability.py) chỉ theo dõi drift màu (_features_rest).
#           Cao variance = hình học đang bất ổn (có thể đang méo để bù đắp
#           cho SH, thay vì tách thành 2 Gaussian riêng) → V thấp.
# Được gọi từ: train.py mỗi crs_update_interval iter, SAU optimizer.step()
#              (cùng nhịp với update_sh_stability).
# ============================================================

import torch


@torch.no_grad()
def update_v_stability(gaussians, beta: float = 0.95):
    """Update EMA mean và variance của _xyz, _scaling, _rotation per Gaussian.

    Cùng công thức Welford-like EMA với update_sh_stability (sh_stability.py),
    áp dụng riêng cho 3 tensor hình học thay vì _features_rest.

    Args:
        gaussians: GaussianModel với _xyz [N,3], _scaling [N,3], _rotation [N,4].
        beta: EMA decay factor. Default 0.95 (khớp sh_stability_ema_beta default).

    Side effects:
        gaussians._v_ema_mean_xyz / _v_ema_var_xyz       (N, 3)
        gaussians._v_ema_mean_scaling / _v_ema_var_scaling (N, 3)
        gaussians._v_ema_mean_rotation / _v_ema_var_rotation (N, 4)
    """
    xyz = gaussians._xyz.detach()
    scaling = gaussians._scaling.detach()
    rotation = gaussians._rotation.detach()

    # Re-init khi chưa có buffer hoặc N thay đổi (densify/prune giữa các update).
    # Cùng pattern shape-mismatch-detect như sh_stability.py — không đụng
    # gaussian_model.py prune/densify code paths (Quy tắc 12: module tách biệt).
    need_init = (
        not hasattr(gaussians, "_v_ema_mean_xyz")
        or gaussians._v_ema_mean_xyz.shape != xyz.shape
    )
    if need_init:
        gaussians._v_ema_mean_xyz = xyz.clone()
        gaussians._v_ema_var_xyz = torch.zeros_like(xyz)
        gaussians._v_ema_mean_scaling = scaling.clone()
        gaussians._v_ema_var_scaling = torch.zeros_like(scaling)
        gaussians._v_ema_mean_rotation = rotation.clone()
        gaussians._v_ema_var_rotation = torch.zeros_like(rotation)
        return

    for name, x in (("xyz", xyz), ("scaling", scaling), ("rotation", rotation)):
        mean_attr = f"_v_ema_mean_{name}"
        var_attr = f"_v_ema_var_{name}"
        mean_old = getattr(gaussians, mean_attr)
        # delta tính từ mean cũ (trước update) — phản ánh "x vs estimate cũ".
        delta = x - mean_old
        setattr(gaussians, mean_attr, beta * mean_old + (1 - beta) * x)
        var_old = getattr(gaussians, var_attr)
        setattr(gaussians, var_attr, beta * var_old + (1 - beta) * delta.pow(2))


@torch.no_grad()
def compute_V_stability(gaussians, scale_xyz: float = 1.0, scale_shape: float = 1.0):
    """Compute V_stability per Gaussian từ EMA variance của xyz + scaling + rotation.

    Mapping logic (cùng dạng compute_S_stability trong sh_stability.py):
        std_xyz   = sqrt(sum(var_xyz))     — độ "nhảy" vị trí
        std_shape = sqrt(sum(var_scaling) + sum(var_rotation)) — độ "méo" hình dạng
        V = 1 - sigmoid(scale_xyz * log1p(std_xyz) + scale_shape * log1p(std_shape))

    std=0 (chưa đủ EMA data, hoặc hình học tuyệt đối ổn định) → V=0.5 (neutral).
    std lớn (đang trôi dạt/méo) → V thấp (untrustworthy).

    Args:
        gaussians: GaussianModel.
        scale_xyz: multiplier cho phần vị trí trước sigmoid.
        scale_shape: multiplier cho phần scale+rotation trước sigmoid.

    Returns:
        V: (N, 1) tensor float32 GPU, range [0, 1].
    """
    if not hasattr(gaussians, "_v_ema_var_xyz"):
        N = gaussians.get_xyz.shape[0]
        return torch.full((N, 1), 0.5, device=gaussians.get_xyz.device)

    var_xyz_total = gaussians._v_ema_var_xyz.sum(dim=1)           # (N,)
    var_shape_total = (
        gaussians._v_ema_var_scaling.sum(dim=1)
        + gaussians._v_ema_var_rotation.sum(dim=1)
    )                                                              # (N,)

    std_xyz = torch.sqrt(var_xyz_total + 1e-8)
    std_shape = torch.sqrt(var_shape_total + 1e-8)

    V = 1.0 - torch.sigmoid(scale_xyz * torch.log1p(std_xyz)
                             + scale_shape * torch.log1p(std_shape))
    return V.unsqueeze(-1).clamp(0.0, 1.0)  # (N, 1)
