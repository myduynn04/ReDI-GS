# ============================================================
# [CRSGaussian Phase 11 Step 4] Cross-view Feature MPC.
# File: utils/loss/feature_mpc_crossview.py (NEW)
# Mục đích (true ICO-GS adapted với DINOv2):
#   1. Render image @ cam_A → F_A_render (DINO patch features).
#   2. Forward warp F_B_GT (cached) từ cam_B sang cam_A bằng aligned
#      depth_A + camera geometry.
#   3. Loss = 1 - mean_cos_sim(F_A_render, F_B_warped) trên valid pixel.
# Cross-view consistency → buộc render align với scene-level geometry,
# không chỉ pixel-level RGB.
#
# Cost: ~50ms/iter trên LLFF -r 8. Gated bởi feature_mpc_freq=10
# (compute mỗi 10 iter, average tổng overhead ~5ms/iter).
#
# Camera convention (verified):
#   - cam.world_view_transform.T = W2C row-major
#   - C2W = inverse(W2C). Patch coords trên grid (Hp, Wp).
# ============================================================

import torch
import torch.nn.functional as F

from utils.graphics_utils import fov2focal


def find_nearest_other_cam(cam_A, all_cameras):
    """Find training cam khác cam_A theo Euclidean distance giữa camera centers.

    Khác `find_nearest_training_cam` (ở utils/depth/depth_warping.py):
      - Đó là angular distance pseudo→training (tìm cam nhìn cùng hướng).
      - Đây là Euclidean train→train (tìm cam gần nhất khác cam hiện tại
        — baseline càng nhỏ, warp càng chính xác cho MPC).
    """
    pos_A = cam_A.camera_center
    best, best_d = None, float('inf')
    for cam_B in all_cameras:
        if cam_B is cam_A or cam_B.uid == cam_A.uid:
            continue
        if not hasattr(cam_B, 'camera_center') or not hasattr(cam_B, 'original_image'):
            continue
        d = (pos_A - cam_B.camera_center).norm().item()
        if d < best_d:
            best_d = d
            best = cam_B
    return best


def feature_mpc_loss(image_render_A, cam_A, cam_B, depth_A, dino_cache, dino):
    """Cross-view feature MPC loss giữa render @ cam_A và warped GT @ cam_B.

    Pipeline:
      1. F_A_render = DINO(image_render_A)   — gradient-enabled (Np_A, D)
      2. F_B_GT    = dino_cache[cam_B.uid]  — (Np_B, D), reshape (Hp, Wp, D)
      3. Generate patch grid trong cam_A native res, sample depth_A tại
         patch centers (bilinear).
      4. Unproject (patch_uv, depth_A) → 3D world points.
      5. Project world → cam_B native res → pixel (u_B, v_B).
      6. Convert (u_B, v_B) → patch coords [0, Wp-1]; bilinear sample
         F_B_GT_2d → F_B_warped (Hp, Wp, D).
      7. Loss = 1 - cos_sim(F_A_render, F_B_warped).mean() trên valid mask.

    Args:
        image_render_A: (3, H, W) tensor — rendered output @ cam_A.
        cam_A, cam_B: Camera objects (cam_A là current viewpoint).
        depth_A: (H, W) aligned DAV2 depth GPU @ cam_A.
        dino_cache: FeatureCache — chứa F_B_GT.
        dino: DINOWrapper.

    Returns:
        loss: scalar tensor.
        valid_mask: (Hp*Wp,) bool — patches visible từ cả cam_A và cam_B.
    """
    F_B_GT = dino_cache.get(cam_B.uid)
    Hp, Wp = dino_cache.grid_shape
    device = image_render_A.device
    Np = Hp * Wp

    if F_B_GT is None:
        return torch.zeros((), device=device), torch.zeros(Np, dtype=torch.bool, device=device)

    # 1. F_A from render (grad-enabled).
    F_A_render = dino.extract_patch_features(image_render_A)   # (Np, D)
    # F_B GT reshape về (Hp, Wp, D) cho grid_sample sau.
    D = F_B_GT.shape[-1]
    F_B_GT_2d = F_B_GT.view(Hp, Wp, D)

    # 2. Patch centers trong cam_A native res.
    H_A = cam_A.image_height
    W_A = cam_A.image_width
    # Patch i ↦ pixel ((i + 0.5) × W_A / Wp). Center của patch.
    py_a, px_a = torch.meshgrid(
        (torch.arange(Hp, device=device, dtype=torch.float32) + 0.5) * H_A / Hp,
        (torch.arange(Wp, device=device, dtype=torch.float32) + 0.5) * W_A / Wp,
        indexing='ij',
    )  # (Hp, Wp)

    # 3. Sample depth_A tại patch centers (bilinear).
    grid_x = 2.0 * px_a / max(W_A - 1, 1) - 1.0
    grid_y = 2.0 * py_a / max(H_A - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)   # (1, Hp, Wp, 2)
    depth_A_4d = depth_A.unsqueeze(0).unsqueeze(0)
    z_A = F.grid_sample(
        depth_A_4d, grid,
        mode='bilinear', padding_mode='zeros', align_corners=True,
    ).squeeze(0).squeeze(0)                                      # (Hp, Wp)

    # 4. Unproject (u, v, z) cam_A → cam-A 3D, rồi → world.
    fx_A = fov2focal(cam_A.FoVx, W_A)
    fy_A = fov2focal(cam_A.FoVy, H_A)
    cx_A, cy_A = W_A / 2.0, H_A / 2.0
    x_cam = (px_a - cx_A) * z_A / fx_A
    y_cam = (py_a - cy_A) * z_A / fy_A
    ones = torch.ones_like(z_A)
    pts_cam_A = torch.stack([x_cam, y_cam, z_A, ones], dim=-1)   # (Hp, Wp, 4)

    W2C_A = cam_A.world_view_transform.T
    C2W_A = torch.inverse(W2C_A)
    pts_world = pts_cam_A @ C2W_A.T                              # (Hp, Wp, 4)

    # 5. World → cam_B → pixel.
    H_B = cam_B.image_height
    W_B = cam_B.image_width
    W2C_B = cam_B.world_view_transform.T
    fx_B = fov2focal(cam_B.FoVx, W_B)
    fy_B = fov2focal(cam_B.FoVy, H_B)
    cx_B, cy_B = W_B / 2.0, H_B / 2.0

    pts_cam_B = pts_world @ W2C_B.T                              # (Hp, Wp, 4)
    z_B = pts_cam_B[..., 2]
    safe_z = z_B.clamp(min=1e-6)
    pixel_x_B = pts_cam_B[..., 0] / safe_z * fx_B + cx_B
    pixel_y_B = pts_cam_B[..., 1] / safe_z * fy_B + cy_B

    # 6. Pixel cam_B → patch coords (0..Wp-1, 0..Hp-1).
    # Tương tự r_feature.py: patch = pixel × Wp / W (vì Wp = input/14, scale tổng là Wp/W).
    patch_x_B = pixel_x_B * Wp / W_B
    patch_y_B = pixel_y_B * Hp / H_B

    in_bounds = (
        (z_B > 0) & (z_A > 1e-6)
        & (patch_x_B >= 0) & (patch_x_B < Wp)
        & (patch_y_B >= 0) & (patch_y_B < Hp)
    )

    # Bilinear sample F_B_GT_2d (Hp, Wp, D) tại (patch_x_B, patch_y_B).
    # grid_sample input: (N, C, H, W); ta cần permute (D, Hp, Wp).
    F_B_input = F_B_GT_2d.permute(2, 0, 1).unsqueeze(0)          # (1, D, Hp, Wp)
    sample_gx = 2.0 * patch_x_B / max(Wp - 1, 1) - 1.0
    sample_gy = 2.0 * patch_y_B / max(Hp - 1, 1) - 1.0
    sample_grid = torch.stack([sample_gx, sample_gy], dim=-1).unsqueeze(0)  # (1, Hp, Wp, 2)
    F_B_warped = F.grid_sample(
        F_B_input, sample_grid,
        mode='bilinear', padding_mode='zeros', align_corners=True,
    ).squeeze(0)                                                  # (D, Hp, Wp)
    F_B_warped = F_B_warped.permute(1, 2, 0).reshape(Np, D)       # (Np, D)

    # 7. Cosine sim per-patch, mask invalid, mean.
    sim = F.cosine_similarity(F_A_render, F_B_warped, dim=-1)     # (Np,) ∈ [-1, 1]
    valid_flat = in_bounds.flatten()                              # (Np,)
    if valid_flat.sum() == 0:
        return torch.zeros((), device=device), valid_flat
    loss = 1.0 - sim[valid_flat].mean()
    return loss, valid_flat
