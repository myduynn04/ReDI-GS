# ============================================================
# [CRSGaussian Phase 11 Step 3] R_feature — replace R_visible bằng
# DINO patch similarity giữa các views tại Gaussian projections.
# File: utils/crs/r_feature.py (NEW)
#
# Phase 8 R_visible: pairwise GT RGB similarity tại projected pixels.
#   Lighting-sensitive → contamination từ specular/highlight, gây
#   −0.111 dB alone (drift away từ true geometry signal).
# R_feature:        pairwise DINO patch similarity tại projected pixels.
#   Self-supervised → invariant với lighting, view-dependent BRDF.
#   Hypothesis: signal "có cùng object" thay vì "có cùng pixel color".
#
# Drop-in replacement: cùng signature output (N, 1) ∈ [0, 1] với
# compute_reprojection_consistency. Visibility filter giống nhau
# (>= min_visible_views → tính, else neutral 0.5).
#
# Camera convention (verified utils/depth + utils/crs):
#   - cam.world_view_transform.T = W2C row-major
#   - fx = fov2focal(cam.FoVx, W); cx = W/2.
# ============================================================

import torch
import torch.nn.functional as F

from utils.graphics_utils import fov2focal


@torch.no_grad()
def compute_R_feature(gaussians, train_cameras, dino_cache,
                       min_visible_views=2):
    """Per-Gaussian DINO feature consistency cross-view.

    Algorithm cho mỗi Gaussian i:
      1. Project xyz_i → pixel (u_k, v_k) trong mỗi cam k visible.
      2. Map (u_k, v_k) trong native res → patch idx trong (Hp, Wp) grid
         (DINO làm việc ở resolution 518×518 sau resize, cần re-scale).
      3. Index F_train[k][patch_idx] → feature (D,) cho mỗi cam k visible.
      4. Pairwise cosine similarity giữa các features:
            R_i = mean_{k1<k2} cos_sim(f_i^k1, f_i^k2) ∈ [-1, 1].
      5. Map [-1, 1] → [0, 1] qua (R + 1) / 2.

    Args:
        gaussians: GaussianModel — cần .get_xyz (N, 3) GPU.
        train_cameras: list training Camera objects.
        dino_cache: FeatureCache — cần .cache, .grid_shape, .feature_dim.
        min_visible_views: tối thiểu views Gaussian phải visible. < min →
            R = 0.5 (neutral, không đủ data).

    Returns:
        R: (N, 1) tensor float32 GPU ∈ [0, 1]. Drop-in cho R_visible.
    """
    xyz = gaussians.get_xyz.detach()
    N = xyz.shape[0]
    device = xyz.device

    # Filter cams có cả original_image (skip pseudo) VÀ cache hit.
    valid_cams = [
        c for c in train_cameras
        if hasattr(c, 'original_image') and c.uid in dino_cache.cache
    ]
    K = len(valid_cams)
    if K < min_visible_views:
        return torch.full((N, 1), 0.5, device=device)

    Hp, Wp = dino_cache.grid_shape           # (37, 37) typical
    D = dino_cache.feature_dim
    input_size = Hp * 14  # = dino.input_size (518)

    # Buffers: features per Gaussian per cam + valid mask.
    features = torch.zeros(N, K, D, device=device)
    valid_per_cam = torch.zeros(N, K, dtype=torch.bool, device=device)

    # Homogeneous coords — tính 1 lần, reuse.
    ones = torch.ones(N, 1, device=device, dtype=xyz.dtype)
    xyz_hom = torch.cat([xyz, ones], dim=1)  # (N, 4)

    for k, cam in enumerate(valid_cams):
        H = cam.image_height
        W = cam.image_width
        W2C = cam.world_view_transform.T     # (4, 4) GPU row-major
        pts_cam = (W2C @ xyz_hom.T).T        # (N, 4)
        depth = pts_cam[:, 2]                # (N,)

        fx = fov2focal(cam.FoVx, W)
        fy = fov2focal(cam.FoVy, H)
        cx, cy = W / 2.0, H / 2.0

        safe_depth = depth.clamp(min=1e-6)
        pixel_x = pts_cam[:, 0] / safe_depth * fx + cx
        pixel_y = pts_cam[:, 1] / safe_depth * fy + cy

        valid = (
            (depth > 0)
            & (pixel_x >= 0) & (pixel_x < W)
            & (pixel_y >= 0) & (pixel_y < H)
        )
        if valid.sum() == 0:
            continue

        # ── Map (u, v) ∈ native res → patch idx trong (Hp, Wp) ──
        # DINO làm việc ở resolution input_size (518). Sau resize từ (H, W)
        # → (input_size, input_size), pixel scale theo tỉ lệ:
        #   px_dino = pixel_x * input_size / W
        #   patch_x = px_dino // patch_size = pixel_x * input_size / W / 14
        # Tương tự cho y. Clamp về [0, Wp-1] / [0, Hp-1] sau div.
        scale_x = input_size / W / 14.0      # = Wp / W (vì Wp = input_size / 14)
        scale_y = input_size / H / 14.0
        valid_idx = torch.where(valid)[0]
        patch_x = (pixel_x[valid_idx] * scale_x).long().clamp(0, Wp - 1)
        patch_y = (pixel_y[valid_idx] * scale_y).long().clamp(0, Hp - 1)
        patch_flat = patch_y * Wp + patch_x  # (M,) ∈ [0, Np-1]

        F_cam = dino_cache.cache[cam.uid]    # (Np, D) GPU
        sampled = F_cam[patch_flat]           # (M, D)

        features[valid_idx, k] = sampled
        valid_per_cam[valid_idx, k] = True

    # ── Pairwise cosine sim cross-view ──
    R_sum = torch.zeros(N, device=device)
    pair_count = torch.zeros(N, device=device)
    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            both_valid = valid_per_cam[:, k1] & valid_per_cam[:, k2]
            if both_valid.sum() == 0:
                continue
            f1 = features[both_valid, k1]
            f2 = features[both_valid, k2]
            sim = F.cosine_similarity(f1, f2, dim=-1)  # (M,) ∈ [-1, 1]
            R_sum[both_valid] += sim
            pair_count[both_valid] += 1.0

    visible_count = valid_per_cam.sum(dim=1)
    enough_views = visible_count >= min_visible_views
    has_pairs = (pair_count > 0) & enough_views

    R = torch.full((N, 1), 0.5, device=device)
    if has_pairs.sum() > 0:
        # cos sim ∈ [-1, 1] → [0, 1]: (R + 1) / 2.
        R_avg = R_sum[has_pairs] / pair_count[has_pairs]
        R[has_pairs, 0] = ((R_avg + 1.0) / 2.0).clamp(0.0, 1.0)
    return R
