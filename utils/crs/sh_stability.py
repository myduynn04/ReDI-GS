# ============================================================
# [CRSGaussian Phase 8b] SH stability signal computation
# File: utils/crs/sh_stability.py (NEW)
# Mục đích: Track _features_rest variance per Gaussian qua EMA →
#           detect SH coefficient drift (memorizing colors vs stable).
#           Hi-variance = SH đang drift (overfit/memorize), low-variance = stable.
#           S_i = 1 - sigmoid(scale × log1p(sqrt(ema_var))) ∈ [0, 1].
# Được gọi từ: train.py mỗi crs_update_interval iter (sau optimizer step).
# ============================================================

import torch


@torch.no_grad()
def update_sh_stability(gaussians, beta: float = 0.95):
    """Update EMA mean và variance của _features_rest per Gaussian.

    Initializes _sh_ema_mean và _sh_ema_var attributes nếu chưa có.
    Sử dụng Welford-like EMA:
        delta     = x - mean_old
        mean_new  = β · mean_old + (1-β) · x
        var_new   = β · var_old  + (1-β) · delta²

    Args:
        gaussians: GaussianModel với _features_rest [N, K_rest, 3].
        beta: EMA decay factor. Default 0.95 (chậm, smooth qua nhiều iter).
              Cao hơn → smoother nhưng phản ứng chậm hơn.
              Ablate: {0.9, 0.95, 0.99}.

    Side effects:
        gaussians._sh_ema_mean: tensor cùng shape _features_rest, EMA mean.
        gaussians._sh_ema_var:  tensor cùng shape, EMA variance.
    """
    rest = gaussians._features_rest.detach()  # [N, K_rest, 3]

    # Init / re-init khi N thay đổi (densify/prune giữa các update).
    if (not hasattr(gaussians, "_sh_ema_mean")
            or gaussians._sh_ema_mean.shape != rest.shape):
        gaussians._sh_ema_mean = rest.clone()
        # Variance init = 0 (không có drift đã observed) → S đầu = sigmoid(0)=0.5
        # sau 1 lần map qua công thức compute_S_stability dưới.
        gaussians._sh_ema_var = torch.zeros_like(rest)
        return

    # delta tính từ MEAN OLD (trước khi update mean) để phản ánh "x vs estimate cũ".
    delta = rest - gaussians._sh_ema_mean
    gaussians._sh_ema_mean = beta * gaussians._sh_ema_mean + (1 - beta) * rest
    gaussians._sh_ema_var = beta * gaussians._sh_ema_var + (1 - beta) * delta.pow(2)


@torch.no_grad()
def compute_S_stability(gaussians, scale: float = 1.0):
    """Compute S_stability per Gaussian từ EMA variance.

    Mapping logic:
        total_var = sum across SH coefs × RGB channels  → scalar per Gaussian
        std       = sqrt(total_var)
        S         = 1 - sigmoid(scale × log1p(std))

    log1p(.) bóp std outlier về phổ rộng [0, ~10]. sigmoid về [0, 1].
    1 - sigmoid(.) flip: high std (drift) → low S (untrustworthy).
    Std=0 → log1p(0)=0 → sigmoid(0)=0.5 → S=0.5 (neutral).

    Args:
        gaussians: GaussianModel.
        scale: multiplier trước sigmoid. Higher = sharper bimodal split.
            Default 1.0. Ablate {0.5, 1.0, 2.0} nếu S range hẹp.

    Returns:
        S: (N, 1) tensor float32 GPU, range [0, 1].
            S ≈ 1: SH stable, không drift → trustworthy.
            S ≈ 0.5: chưa có data EMA hoặc std = 0.
            S ≈ 0: SH drifting nhanh → memorizing → untrustworthy.
    """
    if not hasattr(gaussians, "_sh_ema_var"):
        # Chưa init → neutral 0.5 cho mọi Gaussian
        N = gaussians.get_xyz.shape[0]
        return torch.full((N, 1), 0.5, device=gaussians.get_xyz.device)

    # Per-Gaussian total variance: sum across (K_rest × 3) coefficients.
    # Shape [N, K_rest, 3] → flatten → sum dim 1 → (N,)
    total_var = gaussians._sh_ema_var.flatten(start_dim=1).sum(dim=1)  # (N,)
    std = torch.sqrt(total_var + 1e-8)

    # Map std → [0, 1]: high std → low S.
    # log1p tame outlier (1 Gaussian drift mạnh không dominate sigmoid).
    S = 1.0 - torch.sigmoid(scale * torch.log1p(std))
    return S.unsqueeze(-1).clamp(0.0, 1.0)  # (N, 1)
