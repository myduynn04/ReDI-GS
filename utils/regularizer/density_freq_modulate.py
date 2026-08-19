# ============================================================
# [CRSGaussian Phase 26 C1] Density-as-frequency regularization modulation
# File: utils/regularizer/density_freq_modulate.py (NEW)
# Mục đích: DropAnSH hiện áp p_sh (SH dropout probability) và pa (anchor
#           sample rate) ĐỀU TAY cho mọi Gaussian — nghi ngờ đây là nguồn
#           gây underfit ở vùng chi tiết mịn (texture, biên sắc). Module này
#           dùng mật độ Gaussian cục bộ (đã có sẵn ở density_voxel.py /
#           density_covariance.py, hiện chỉ dùng để CHỌN anchor) làm proxy
#           cho "tần số không gian 3D": vùng mật độ cao + scale nhỏ = tần số
#           cao (chi tiết cần bảo toàn) → giảm cường độ regularization; vùng
#           mật độ thấp + scale lớn = tần số thấp (mặt phẳng trơn) → giữ
#           nguyên hoặc tăng cường độ.
# Được gọi từ: gaussian_renderer/__init__.py (thay p_sh scalar bằng per-
#              Gaussian probability array khi use_density_freq_modulate=True).
# ============================================================

import torch


@torch.no_grad()
def compute_frequency_signal(
    gaussians,
    method: str = "voxel",
    k: int = 10,
) -> torch.Tensor:
    """[Phase 26 C1] Ước lượng tần số không gian cục bộ per-Gaussian.

    freq_i cao = vùng dày đặc Gaussian nhỏ (chi tiết mịn, biên) → cần bảo toàn.
    freq_i thấp = vùng thưa Gaussian lớn (mặt phẳng trơn) → an toàn regularize mạnh.

    Kết hợp 2 tín hiệu đã có sẵn (density_voxel.py / density_covariance.py):
        density: mật độ Gaussian lân cận (đếm hoặc overlap).
        inv_scale: 1/mean(scale) — Gaussian nhỏ → tần số cao hơn Gaussian to.
    freq = normalize(density) * normalize(inv_scale), rồi renormalize [0,1].

    Args:
        gaussians: GaussianModel.
        method: "voxel" (rẻ, đếm theo ô lưới) | "covariance" (chính xác hơn,
            Bhattacharyya overlap, tốn hơn — dùng cho scene nhỏ/N thấp).
        k: số neighbor cho method="covariance".

    Returns:
        freq: (N,) float tensor [0, 1]. 1 = tần số cao nhất (bảo toàn).
    """
    xyz = gaussians.get_xyz
    N = xyz.shape[0]
    if N == 0:
        return torch.zeros(0, device=xyz.device)

    if method == "voxel":
        from utils.regularizer.density_voxel import compute_voxel_density
        density = compute_voxel_density(xyz)
    elif method == "covariance":
        from utils.regularizer.density_covariance import compute_covariance_density
        density = compute_covariance_density(
            xyz, gaussians._scaling, gaussians._rotation, k=k,
        )
    else:
        raise ValueError(f"[Phase 26 C1] method không hợp lệ: {method!r} "
                          f"(voxel | covariance)")

    # Gaussian scale nhỏ → tần số cao hơn (chi tiết mịn cần Gaussian nhỏ để vẽ).
    mean_scale = gaussians.get_scaling.mean(dim=1)  # (N,) đã activate (exp)
    inv_scale = 1.0 / (mean_scale + 1e-6)
    inv_scale_norm = inv_scale / inv_scale.quantile(0.95).clamp(min=1e-6)
    inv_scale_norm = inv_scale_norm.clamp(0.0, 1.0)

    freq = density * inv_scale_norm
    freq_ref = freq.quantile(0.95).clamp(min=1e-6)
    freq = (freq / freq_ref).clamp(0.0, 1.0)
    return freq


@torch.no_grad()
def modulate_probability(
    base_prob: float,
    freq: torch.Tensor,
    strength: float = 1.0,
    min_prob_ratio: float = 0.0,
) -> torch.Tensor:
    """[Phase 26 C1] Modulate base scalar probability thành per-Gaussian array.

    p_i = base_prob * (1 - strength * freq_i), clamp về
          [base_prob * min_prob_ratio, base_prob].
    freq_i cao (chi tiết) → p_i thấp (ít bị dropout/decay hơn).
    freq_i thấp (mặt phẳng) → p_i giữ gần base_prob.
    strength=0 → p_i = base_prob mọi nơi (byte-identical baseline).

    Args:
        base_prob: giá trị scalar gốc (vd dropansh_psh=0.2, dropansh_pa=0.02).
        freq: (N,) tensor [0,1] từ compute_frequency_signal.
        strength: mức độ giảm tại vùng tần số cao nhất. Default 1.0 (full range).
        min_prob_ratio: sàn tối thiểu theo tỉ lệ base_prob, tránh p_i=0 tuyệt
            đối (vẫn còn chút regularization ở vùng chi tiết). Default 0.0.

    Returns:
        p: (N,) float tensor, per-Gaussian probability.
    """
    p = base_prob * (1.0 - strength * freq)
    p = p.clamp(min=base_prob * min_prob_ratio, max=base_prob)
    return p
