# ============================================================
# [CRSGaussian Phase 11 Step 1] Depth-based covisibility reweighter
# File: utils/loss/covisibility_depth.py (NEW)
# Mục đích:
#   - Per-pixel weight cho L_phot dựa trên (a) covisibility — số views
#     "nhìn thấy" được điểm 3D tương ứng pixel (forward-warp dùng aligned
#     depth) và (b) optional combine với CRS_pix.
#   - Pixel reliable hơn (visible từ nhiều views, CRS cao) → weight cao →
#     loss focus vào reliable region; pixel chỉ 1-view (textureless edge,
#     view-dependent reflection, occluded boundary) → weight floor (γ).
#
# Reuse aligned DAV2 depth từ utils/depth/depth_alignment.py — KHÔNG cần
# DUSt3R/MASt3R external model. One-shot precompute trước training loop;
# cov_maps dict cached qua mọi iter (camera poses fixed).
#
# Camera convention (verified utils/depth + scene/cameras):
#   - cam.R: cam→world rotation (qvec2rotmat(qvec).T)
#   - cam.T: world→cam translation
#   - cam.world_view_transform: W2C transposed (column-major), → .T = W2C row-major
#   - aligned_depth: dict {cam.uid: (H, W) float32 GPU} metric scale
# ============================================================

import torch
import torch.nn.functional as F

from utils.graphics_utils import fov2focal


@torch.no_grad()
def compute_covisibility_maps(cameras, aligned_depths, depth_consistency_thr=0.05):
    """One-shot precompute cov_map per training camera.

    Algorithm cho mỗi cam A:
      1. Unproject pixel grid của A bằng aligned_depth_A → 3D world points
      2. For each cam B ≠ A:
         - Project world points → pixel coords + predicted z trên cam B
         - Bilinear sample aligned_depth_B tại pixel coords → z_B_actual
         - Mark covisible: in-bounds AND z_B_pred > 0 AND
                            |z_B_pred − z_B_actual| / z_B_actual < threshold
      3. cov_A[p] = sum over B của covisible(B, p) ∈ [0, N-1]

    Args:
        cameras: list training Camera objects (có .uid, .original_image,
                 .world_view_transform GPU, .FoVx/.FoVy, .image_width/_height).
        aligned_depths: dict {cam.uid: Tensor (H, W) GPU float32}.
        depth_consistency_thr: relative diff threshold (default 5%).

    Returns:
        dict {cam.uid: Tensor (H, W) int32 GPU} — count covisible views per pixel.
        Empty dict nếu < 2 cams có aligned depth.
    """
    cams = [c for c in cameras if c.uid in aligned_depths
            and hasattr(c, 'original_image')]  # filter pseudo cams + missing depth
    n_cams = len(cams)
    if n_cams < 2:
        return {}

    cov_maps = {}

    for A_idx, cam_A in enumerate(cams):
        H_A = cam_A.image_height
        W_A = cam_A.image_width
        depth_A = aligned_depths[cam_A.uid]   # (H_A, W_A) GPU
        device = depth_A.device

        # ── 1. Pixel grid (u, v) ──
        # indexing='ij' → first axis = row (v), second axis = col (u). Stick to convention.
        v_grid, u_grid = torch.meshgrid(
            torch.arange(H_A, device=device, dtype=torch.float32),
            torch.arange(W_A, device=device, dtype=torch.float32),
            indexing='ij',
        )  # both (H_A, W_A)

        # Intrinsic A
        fx_A = fov2focal(cam_A.FoVx, W_A)
        fy_A = fov2focal(cam_A.FoVy, H_A)
        cx_A, cy_A = W_A / 2.0, H_A / 2.0

        # ── Unproject (u, v, depth) → cam-A coords ──
        # x_cam = (u - cx) * z / fx; y_cam = (v - cy) * z / fy; z_cam = depth.
        z_cam = depth_A
        x_cam = (u_grid - cx_A) * z_cam / fx_A
        y_cam = (v_grid - cy_A) * z_cam / fy_A
        ones = torch.ones_like(z_cam)
        pts_cam_A = torch.stack([x_cam, y_cam, z_cam, ones], dim=-1)  # (H_A, W_A, 4)

        # ── Cam A coords → World ──
        # W2C_A = cam_A.world_view_transform.T (verified pattern utils/depth + utils/crs)
        # C2W_A = inv(W2C_A). Inverse of 4×4 cheap.
        W2C_A = cam_A.world_view_transform.T          # (4, 4) GPU
        C2W_A = torch.inverse(W2C_A)
        # pts_world = pts_cam_A @ C2W_A.T (matmul last-dim). Shape (H, W, 4).
        pts_world = pts_cam_A @ C2W_A.T

        cov = torch.zeros(H_A, W_A, dtype=torch.int32, device=device)
        # Self-pixel valid mask: skip pixels có depth ≤ 0 (DAV2 fail) — không project được.
        depth_A_valid = depth_A > 1e-6

        # ── 2. For each other cam B ──
        for B_idx, cam_B in enumerate(cams):
            if B_idx == A_idx:
                continue

            H_B = cam_B.image_height
            W_B = cam_B.image_width
            depth_B = aligned_depths[cam_B.uid]       # (H_B, W_B) GPU
            W2C_B = cam_B.world_view_transform.T

            fx_B = fov2focal(cam_B.FoVx, W_B)
            fy_B = fov2focal(cam_B.FoVy, H_B)
            cx_B, cy_B = W_B / 2.0, H_B / 2.0

            # World → cam B
            pts_cam_B = pts_world @ W2C_B.T           # (H_A, W_A, 4)
            z_B_pred = pts_cam_B[..., 2]

            # Project to pixel (u_B, v_B). Tránh div-by-0: chỉ dùng khi z > eps.
            safe_z = z_B_pred.clamp(min=1e-6)
            u_B = pts_cam_B[..., 0] / safe_z * fx_B + cx_B
            v_B = pts_cam_B[..., 1] / safe_z * fy_B + cy_B

            # In-bounds + frontmost (z > 0)
            in_bounds = (
                (z_B_pred > 0)
                & (u_B >= 0) & (u_B < W_B)
                & (v_B >= 0) & (v_B < H_B)
            )

            # ── Bilinear sample depth_B tại (u_B, v_B) ──
            # F.grid_sample yêu cầu grid normalized [-1, 1] và shape (N, H_out, W_out, 2).
            # align_corners=True → biên 0 và H-1, W-1 map đúng [-1, 1].
            grid_x = 2.0 * u_B / max(W_B - 1, 1) - 1.0
            grid_y = 2.0 * v_B / max(H_B - 1, 1) - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)   # (1, H_A, W_A, 2)
            depth_B_4d = depth_B.unsqueeze(0).unsqueeze(0)               # (1, 1, H_B, W_B)
            sampled = F.grid_sample(
                depth_B_4d, grid,
                mode='bilinear', padding_mode='zeros', align_corners=True,
            ).squeeze(0).squeeze(0)                                     # (H_A, W_A)

            # ── Consistency check ──
            valid_sample = sampled > 1e-6
            rel_diff = torch.abs(z_B_pred - sampled) / (sampled + 1e-6)
            consistent = (
                in_bounds & valid_sample & depth_A_valid
                & (rel_diff < depth_consistency_thr)
            )
            cov += consistent.int()

        cov_maps[cam_A.uid] = cov

    return cov_maps


def compute_reliability_weight(cov_norm, crs_pix=None, gamma=0.3, combine_mode='min'):
    """Compose covisibility (+ optional CRS_pix) → per-pixel weight ∈ [γ, 1].

    w(p) = γ + (1 - γ) · R(p)
    với R(p) = cov_norm(p)                       nếu crs_pix is None
            = min(cov_norm(p), crs_pix(p))       nếu combine_mode='min'
            = 0.5 · (cov_norm + crs_pix)         nếu combine_mode='mean'

    Floor γ giữ gradient flow tại pixel "unreliable" (không triệt tiêu).

    Args:
        cov_norm: (H, W) float ∈ [0, 1] = cov_count / (N - 1).
        crs_pix:  (H, W) float ∈ [0, 1] hoặc None.
        gamma:    weight floor (default 0.3).
        combine_mode: 'min' | 'mean' khi crs_pix có.
    """
    if crs_pix is None:
        R = cov_norm
    elif combine_mode == 'min':
        R = torch.minimum(cov_norm, crs_pix)
    elif combine_mode == 'mean':
        R = 0.5 * (cov_norm + crs_pix)
    else:
        raise ValueError(f"[Phase 11 Step 1] combine_mode invalid: {combine_mode}")
    w = gamma + (1.0 - gamma) * R
    return w.clamp(gamma, 1.0)


def weighted_l1_loss(I_render, I_GT, weight_map):
    """Weighted L1 loss — mean over pixels theo weight.

    diff_per_pixel = |I_render - I_GT|.mean(dim=channel)        (H, W)
    loss = Σ (diff · w) / (Σ w + eps)

    Args:
        I_render, I_GT: (3, H, W) tensors.
        weight_map: (H, W) ∈ [0, 1].
    """
    diff = torch.abs(I_render - I_GT).mean(dim=0)   # (H, W)
    return (diff * weight_map).sum() / (weight_map.sum() + 1e-6)
