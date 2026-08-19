# ============================================================
# [CRSGaussian Phase 26 A2] CRS-weighted temporal parameter regularization
# File: utils/loss/temporal_reg.py (NEW)
# Mục đích: Toàn bộ loss hiện tại (L1, D-SSIM, depth Pearson, C1 normal) đều
#           là "loss 2D" — so sánh render vs GT tại pixel. Không có loss nào
#           tác động TRỰC TIẾP lên tham số 3D (μ, scale, rotation) để giữ nó
#           ổn định qua các iteration. Kết quả: optimizer có thể "học sai"
#           sau khi khởi tạo đúng — Gaussian trôi dạt/méo để hàm SH bù đắp
#           thay vì tách thành Gaussian riêng.
#           L_temporal phạt độ lệch giữa tham số hiện tại và EMA của chính
#           nó — một dạng "velocity penalty" trực tiếp trong không gian 3D
#           (không qua pixel). Weighted bởi CRS: Gaussian CRS cao (đã đáng
#           tin) bị phạt mạnh nếu di chuyển nhiều (nên ổn định); Gaussian
#           CRS thấp (đang cần tự sửa vị trí) được nới lỏng.
# Được gọi từ: train.py, cộng vào LossDict["loss_gs0"] TRƯỚC .backward(),
#              cùng nhịp các loss khác (mỗi iteration, không phải mỗi
#              crs_update_interval — cần buffer riêng, độc lập v_stability.py
#              của A1 để A2 dùng được mà không cần bật use_v_stability).
# ============================================================

import torch


@torch.no_grad()
def update_temporal_ema(gaussians, beta: float = 0.9):
    """EMA đơn giản (mean-only, không cần variance) cho _xyz/_scaling/_rotation.

    Tách biệt với utils/crs/v_stability.py (EMA mean+variance dùng để tính
    CRS signal V). Buffer riêng để L_temporal hoạt động độc lập không cần
    use_v_stability=True.

    Args:
        gaussians: GaussianModel.
        beta: EMA decay. Default 0.9 (nhanh hơn v_stability's 0.95 — muốn
            phản ứng nhanh với thay đổi gần đây để penalty không lag quá xa).

    Side effects:
        gaussians._temporal_ema_xyz / _temporal_ema_scaling / _temporal_ema_rotation
    """
    xyz = gaussians._xyz.detach()
    scaling = gaussians._scaling.detach()
    rotation = gaussians._rotation.detach()

    need_init = (
        not hasattr(gaussians, "_temporal_ema_xyz")
        or gaussians._temporal_ema_xyz.shape != xyz.shape
    )
    if need_init:
        gaussians._temporal_ema_xyz = xyz.clone()
        gaussians._temporal_ema_scaling = scaling.clone()
        gaussians._temporal_ema_rotation = rotation.clone()
        return

    gaussians._temporal_ema_xyz = beta * gaussians._temporal_ema_xyz + (1 - beta) * xyz
    gaussians._temporal_ema_scaling = beta * gaussians._temporal_ema_scaling + (1 - beta) * scaling
    gaussians._temporal_ema_rotation = beta * gaussians._temporal_ema_rotation + (1 - beta) * rotation


def compute_temporal_reg_loss(
    gaussians,
    lambda_xyz: float = 0.01,
    lambda_shape: float = 0.01,
    crs_weighted: bool = True,
):
    """[Phase 26 A2] L_temporal — phạt độ lệch tham số hiện tại vs EMA của nó.

    L = mean( w_i * [ λ_xyz·||μ_i - ema_μ_i||² + λ_shape·(||Δlog_s_i||² + ||Δq_i||²) ] )

    w_i = CRS_i nếu crs_weighted=True (Gaussian đáng tin bị phạt mạnh hơn
    khi di chuyển — ổn định hoá vùng đã học đúng), else w_i = 1 (uniform).

    QUAN TRỌNG: EMA buffer (_temporal_ema_*) không có .detach() applied lại
    ở đây vì update_temporal_ema() đã @torch.no_grad() — buffer là hằng số
    đối với autograd, gradient chỉ chảy qua nhánh _xyz/_scaling/_rotation
    hiện tại (đúng ý nghĩa "kéo tham số hiện tại về gần EMA").

    Args:
        gaussians: GaussianModel — cần _temporal_ema_* đã init qua
            update_temporal_ema() ít nhất 1 lần trước đó.
        lambda_xyz: trọng số phạt lệch vị trí.
        lambda_shape: trọng số phạt lệch scale+rotation.
        crs_weighted: True → weight theo CRS hiện tại (cần _crs_score).

    Returns:
        scalar tensor (có grad), hoặc None nếu chưa có EMA buffer (iter đầu).
    """
    if not hasattr(gaussians, "_temporal_ema_xyz"):
        return None

    N = gaussians._xyz.shape[0]
    if gaussians._temporal_ema_xyz.shape[0] != N:
        # Shape mismatch (densify/prune vừa chạy, EMA chưa kịp re-init ở
        # update tiếp theo) — skip 1 iter thay vì crash.
        return None

    d_xyz = gaussians._xyz - gaussians._temporal_ema_xyz               # (N,3)
    d_scaling = gaussians._scaling - gaussians._temporal_ema_scaling   # (N,3)
    d_rotation = gaussians._rotation - gaussians._temporal_ema_rotation  # (N,4)

    per_gauss = (
        lambda_xyz * d_xyz.pow(2).sum(dim=1)
        + lambda_shape * (d_scaling.pow(2).sum(dim=1) + d_rotation.pow(2).sum(dim=1))
    )  # (N,)

    if crs_weighted and hasattr(gaussians, "_crs_score"):
        w = gaussians.get_crs.detach().squeeze(-1)  # (N,) ∈ [0,1]
        loss = (w * per_gauss).sum() / w.sum().clamp(min=1e-6)
    else:
        loss = per_gauss.mean()

    return loss
