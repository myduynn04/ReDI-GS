# ============================================================
# [CRSGaussian DropAnSH] Reproduce DropAnSH-GS regularization
# File: utils/regularizer/dropansh.py  (TẠO MỚI)
# Mục đích:
#   (1) Anchor-based dropout: drop cluster (anchor + k-NN) thay vì individual
#   (2) SH degree dropout: progressive schedule lmax 0→1→2 theo iter
#
# Khác với B1 dropout (utils/regularizer/sh_dropout.py):
#   - B1 dropout đều per-Gaussian → phân tán trách nhiệm
#   - Anchor dropout drop cluster 3D → ép Gaussian không phụ thuộc vùng cụ thể
#   - SH degree dropout coarse-to-fine → buộc model học DC trước high-order
#
# Paper ref: "Dropping Anchor and Spherical Harmonics for Sparse-view 3DGS"
#
# Được gọi từ: gaussian_renderer/__init__.py trong render() khi train.
# Combined với B1 dropout qua AND mask (xem renderer).
# ============================================================

import torch
from typing import Optional, Tuple


def anchor_dropout_mask(
    gaussians,
    iteration: int,
    total_iter: int = 10000,
    k: int = 10,
    pa_max: float = 0.02,
    crs_guided: bool = False,
    batch_threshold: int = 200_000_000,
    density_method: str = "uniform",
) -> torch.Tensor:
    """[CRSGaussian DropAnSH] Compute per-Gaussian keep mask — cluster drop.

    Args:
        gaussians: GaussianModel
        iteration: current iter (0-indexed OK).
        total_iter: tổng iter training → linear ramp pa = pa_max * iter/total.
        k: số nearest neighbors / anchor.
        pa_max: anchor sample rate tối đa (e.g., 0.02 = 2% Gaussians là anchor).
        crs_guided: True → chọn anchor weighted theo (1-CRS) (Phase 3).
        batch_threshold: nếu num_anchors × N vượt → batch cdist tránh OOM.
        density_method: [CRSGaussian Phase 2d Stage A] anchor sampling strategy.
            "uniform"    → random anchor (behavior cũ, baseline).
            "voxel"      → weighted bởi voxel density (chọn anchor ở vùng dày).
            "covariance" → weighted bởi sum Bhattacharyya overlap với k-NN.
            Cache kết quả vào gaussians._cached_density, invalidate khi
            densify_and_prune() chạy.

    Returns:
        keep_mask: (N,) bool tensor. True = keep, False = drop (anchor hoặc k-NN).
    """
    # ══════════════════════════════════════════════════════════════════
    # [THESIS §Reg (i) DropAnSH — Anchor-based cluster dropout]
    # Chọn ngẫu nhiên vài % Gaussian làm ANCHOR, bỏ mỗi anchor + k láng giềng
    # = xóa CẢ CỤM 3D khỏi render lần này. Trả về keep_mask (True=giữ).
    # Mục đích: chống "neighbor compensation" — bỏ cụm ép model học toàn cục.
    # ══════════════════════════════════════════════════════════════════
    N = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device

    # pa_max=0.0 → tắt anchor dropout hoàn toàn (dùng cho ablation "chỉ SH, không anchor")
    if pa_max <= 0.0:
        return torch.ones(N, device=device, dtype=torch.bool)

    # [THESIS Eq dropansh-anchor: pₐ(t) = pₐ^max · min(1, t/T)] — tỉ lệ anchor TĂNG DẦN theo iter.
    # Đầu train drop ít (chưa ổn định), cuối train drop tối đa pa_max (production 0.02 = 2%).
    pa = pa_max * min(1.0, iteration / max(total_iter, 1))
    num_anchors = max(1, int(pa * N))       # n_anchor = max(1, ⌊pₐ·N⌋)
    num_anchors = min(num_anchors, N)

    # ── Chọn anchor indices ──
    # Ưu tiên: crs_guided > density_method > uniform random.
    # ⚠️ [NOT USE] crs_guided + density voxel/covariance/crs: KHÔNG dùng trong
    #    production. Production luôn đi nhánh cuối (uniform).
    if crs_guided and hasattr(gaussians, '_crs_score'):
        # [NOT USE] Anchor weighted theo (1-CRS).
        crs = gaussians.get_crs.squeeze(-1)        # (N,)
        weights = (1.0 - crs).clamp(min=0.01)      # đảm bảo > 0 cho multinomial
        anchor_idx = torch.multinomial(weights, num_anchors, replacement=False)
    elif density_method in ("voxel", "covariance"):  # [NOT USE] density-aware
        # [CRSGaussian Phase 2d Stage A] Density-weighted anchor sampling (V1)
        # Compute OR reuse cached density. Cache hợp lệ khi cùng N.
        cached = getattr(gaussians, "_cached_density", None)
        if cached is None or cached.shape[0] != N:
            if density_method == "voxel":
                from utils.regularizer.density_voxel import compute_voxel_density
                cached = compute_voxel_density(gaussians.get_xyz)
            else:  # covariance
                from utils.regularizer.density_covariance import compute_covariance_density
                cached = compute_covariance_density(
                    gaussians.get_xyz,
                    gaussians._scaling,
                    gaussians._rotation,
                    k=k,
                )
            gaussians._cached_density = cached
            # Diag log mỗi lần compute lại (sau densify) — 1 dòng stats
            if iteration % 500 == 0 or iteration == 1000:
                d = cached
                print(f"[Density {density_method}] iter={iteration} N={N} "
                      f"min={d.min():.3f} q25={d.quantile(0.25):.3f} "
                      f"median={d.median():.3f} q75={d.quantile(0.75):.3f} "
                      f"max={d.max():.3f}")
        weights = (cached + 1e-6).to(device)
        anchor_idx = torch.multinomial(weights, num_anchors, replacement=False)
    elif density_method == "crs":  # [NOT USE] CRS-guided anchor
        # [CRSGaussian Phase 3α] Pure CRS-guided anchor selection.
        # CRS thấp (D_i sai, R_i sai) = floater candidate → prime anchor target.
        # CRS cao = surface đúng → protected (xác suất chọn anchor thấp).
        # get_crs đã activate sigmoid, range [0.08, 0.92] sau scale=5.0.
        crs = gaussians.get_crs.squeeze(-1)            # (N,) ∈ [0,1]
        weights = (1.0 - crs) + 1e-6                   # low CRS → high weight
        # Sanity log iter 1000 — verify CRS distribution có spread, không stuck 0.5
        if iteration == 1000:
            print(f"[CRS {density_method}] iter=1000 N={N} "
                  f"min={crs.min():.3f} q25={crs.quantile(0.25):.3f} "
                  f"median={crs.median():.3f} q75={crs.quantile(0.75):.3f} "
                  f"max={crs.max():.3f}")
        anchor_idx = torch.multinomial(weights, num_anchors, replacement=False)
    elif density_method == "crs_voxel":  # [NOT USE] CRS × density
        # [CRSGaussian Phase 3β] CRS × voxel density combined.
        # Dense AND low-CRS = floater cluster → prime target.
        # Dense + high-CRS = surface coherent → protected.
        # Sparse + low-CRS = isolated floater → ít chọn (đã handled bởi CRS pruning).
        cached = getattr(gaussians, "_cached_density", None)
        if cached is None or cached.shape[0] != N:
            from utils.regularizer.density_voxel import compute_voxel_density
            cached = compute_voxel_density(gaussians.get_xyz)
            gaussians._cached_density = cached
        crs = gaussians.get_crs.squeeze(-1)            # (N,)
        weights = cached * (1.0 - crs) + 1e-6
        if iteration == 1000:
            print(f"[CRS {density_method}] iter=1000 N={N} "
                  f"crs_med={crs.median():.3f} dens_med={cached.median():.3f} "
                  f"weight_med={weights.median():.4f}")
        anchor_idx = torch.multinomial(weights, num_anchors, replacement=False)
    else:
        # ⭐ PRODUCTION — chọn anchor NGẪU NHIÊN đều (uniform). Đây là nhánh bài dùng.
        anchor_idx = torch.randperm(N, device=device)[:num_anchors]

    positions = gaussians.get_xyz
    anchor_pos = positions[anchor_idx]  # (num_anchors, 3) — vị trí các anchor

    # ── Tìm k láng giềng gần nhất của mỗi anchor → gom thành CỤM để bỏ ──
    # k=10 production. Batch cdist nếu N × num_anchors quá lớn (tránh OOM).
    effective_k = min(k + 1, N)   # +1 vì chính anchor cũng nằm trong k-NN của nó
    if num_anchors * N > batch_threshold:
        # Batched cdist để tránh OOM khi N × num_anchors lớn
        batch_size = max(1, batch_threshold // N)
        nn_chunks = []
        for i in range(0, num_anchors, batch_size):
            d = torch.cdist(anchor_pos[i:i + batch_size], positions)  # (bs, N)
            _, idx = torch.topk(d, k=effective_k, largest=False)
            nn_chunks.append(idx)
        nn_idx = torch.cat(nn_chunks, dim=0)
    else:
        dists = torch.cdist(anchor_pos, positions)                     # (A, N)
        _, nn_idx = torch.topk(dists, k=effective_k, largest=False)   # (A, k+1)

    # Gộp tất cả (anchor + k-NN của mọi anchor) = tập cần BỎ (cả cụm)
    drop_set = nn_idx.flatten().unique()
    keep_mask = torch.ones(N, device=device, dtype=torch.bool)
    keep_mask[drop_set] = False   # False = bị drop (không tham gia render lần này)
    return keep_mask              # renderer dùng mask này để loại cụm khỏi forward


def sh_degree_dropout(
    gaussians,
    iteration: int,
    p_sh: float = 0.2,
    schedule: Tuple[int, int, int] = (2000, 4000, 6000),
    crs_modulated: bool = False,
) -> Optional[Tuple]:
    """[CRSGaussian DropAnSH] Zero SH rest coefficients above lmax in-place.

    Schedule (3 checkpoints):
        iter < schedule[0]:              lmax=0  (chỉ DC, zero tất cả rest)
        schedule[0] ≤ iter < schedule[1]: lmax=1
        schedule[1] ≤ iter < schedule[2]: lmax=2
        iter ≥ schedule[2]:              KHÔNG dropout (return None)

    IMPORTANT: modify _features_rest IN-PLACE → caller phải gọi
    restore_sh_dropout(gaussians, snapshot) sau forward pass để phục hồi.

    Args:
        p_sh: per-Gaussian base probability of SH dropout.
              p_sh=0 → skip (dùng để tắt SH-only, giữ anchor).
        crs_modulated: True → p_per = p_sh * (1-CRS) (Phase 3).

    Returns:
        snapshot_info: None (không dropout), hoặc
            (drop_mask: (N,) bool, num_keep_rest: int, saved: tensor)
    """
    # ══════════════════════════════════════════════════════════════════
    # [THESIS §Reg (i) DropAnSH — SH dropout coarse-to-fine]
    # Với xác suất p_sh mỗi Gaussian, ZERO tạm SH bậc > lmax lúc render.
    # lmax TĂNG dần theo iter → ép model học màu nền (DC) trước, chi tiết sau.
    # Mục đích: chống overfit tần-số-cao (SH bậc cao nhớ vẹt 3 view train).
    # ══════════════════════════════════════════════════════════════════
    # Skip nhanh nếu caller tắt hoặc đã qua giai đoạn dropout
    if p_sh <= 0.0:
        return None
    if iteration >= schedule[2]:   # sau iter 6000 → không dropout nữa (đã học đủ)
        return None

    # [THESIS Eq dropansh-sh] Lịch trình bậc SH tối đa được giữ (coarse→fine)
    if iteration < schedule[0]:    # < 2000: chỉ DC (zero HẾT bậc cao)
        lmax = 0
    elif iteration < schedule[1]:  # < 4000: giữ tới bậc 1
        lmax = 1
    else:                          # < 6000: giữ tới bậc 2
        lmax = 2

    N = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device

    # Xác suất drop mỗi Gaussian
    if crs_modulated and hasattr(gaussians, '_crs_score'):
        # [NOT USE] Phase 3 — p_per điều biến theo CRS. Production KHÔNG dùng.
        crs = gaussians.get_crs.squeeze(-1)
        p_per = (p_sh * (1.0 - crs)).clamp(0.0, 1.0)
    else:
        # ⭐ PRODUCTION — xác suất ĐỀU p_sh (0.2) cho mọi Gaussian
        p_per = torch.full((N,), p_sh, device=device)

    drop_mask = torch.rand(N, device=device) < p_per   # Gaussian nào trúng bị drop SH bậc cao
    if not drop_mask.any():
        return None

    # Số hệ số SH-rest được GIỮ ứng với bậc lmax: (lmax+1)² − 1
    # lmax=0 → giữ 0 (zero hết rest), lmax=1 → giữ 3, lmax=2 → giữ 8
    num_keep_rest = max(0, (lmax + 1) ** 2 - 1)
    rest_shape = gaussians._features_rest.shape[1]

    if num_keep_rest >= rest_shape:
        # Bậc SH của model ≤ lmax → không có bậc cao nào để zero
        return None

    # LƯU lại giá trị trước khi zero (để restore sau render) rồi ZERO in-place
    saved = gaussians._features_rest.data[drop_mask, num_keep_rest:, :].clone()   # snapshot
    gaussians._features_rest.data[drop_mask, num_keep_rest:, :] = 0.0             # tắt tạm SH bậc cao
    return (drop_mask, num_keep_rest, saved)   # trả snapshot để restore_sh_dropout() phục hồi


def restore_sh_dropout(gaussians, snapshot_info):
    """[CRSGaussian DropAnSH] Restore _features_rest sau forward pass.

    Gọi ngay SAU rasterizer call để phục hồi coefficients đã zero.
    Nếu snapshot_info=None → no-op (không có gì để restore).
    """
    if snapshot_info is None:
        return
    drop_mask, num_keep_rest, saved = snapshot_info
    gaussians._features_rest.data[drop_mask, num_keep_rest:, :] = saved
