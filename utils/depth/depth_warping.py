# ============================================================
# [CRSGaussian] Pseudo-view Depth Warping
# File: utils/depth/depth_warping.py  (TẠO MỚI)
# Mục đích:
#   Forward warp aligned DAV2 depth từ training cam A → pseudo-cam P
#   để tạo depth reference EXTERNAL (không phụ thuộc Gaussian model).
#
#   Loss = pearson(rendered_depth_P, warped_depth_P) ép Gaussian model
#   đúng geometry ở novel views → giảm overfit.
#
# Convention (verified ở tests/verify_camera_convention.py — round-trip
# error < 1e-14):
#   - cam.R = C2W rotation (numpy 3x3)
#   - cam.T = W2C translation (numpy 3,)
#   - W2C: xyz_cam   = R.T @ xyz_world + T
#   - C2W: xyz_world = R   @ (xyz_cam - T)
#   - fx = fov2focal(FoVx, W), fy = fov2focal(FoVy, H)
#   - cx = W/2, cy = H/2
#
# Được gọi từ: train.py, mỗi pseudo-view iteration
# ============================================================

import torch
import numpy as np
from utils.graphics_utils import fov2focal


# ════════════════════════════════════════════════════════════
# Helper — chuyển cam.R, cam.T về torch tensor trên đúng device
# ════════════════════════════════════════════════════════════
def _cam_RT(cam, device, dtype=torch.float32):
    """Trả về (R, T) torch tensor trên device.
    cam.R, cam.T thường là numpy → convert + cache đơn giản."""
    R = torch.as_tensor(np.asarray(cam.R), device=device, dtype=dtype)  # (3,3) C2W rot
    T = torch.as_tensor(np.asarray(cam.T), device=device, dtype=dtype)  # (3,) W2C trans
    return R, T


def _cam_intrinsics(cam):
    """Tính (fx, fy, cx, cy, H, W) — convention CRSGaussian."""
    H = int(cam.image_height)
    W = int(cam.image_width)
    fx = float(fov2focal(cam.FoVx, W))
    fy = float(fov2focal(cam.FoVy, H))
    cx = W / 2.0
    cy = H / 2.0
    return fx, fy, cx, cy, H, W


# ════════════════════════════════════════════════════════════
# FUNCTION 1 — find_nearest_training_cam
# ════════════════════════════════════════════════════════════
def find_nearest_training_cam(pseudo_cam, train_cameras):
    """[CRSGaussian] Tìm training cam có forward direction gần nhất với pseudo-cam.

    Tại sao angular distance (không phải Euclidean):
        - Camera rotation quan trọng hơn translation cho coverage warp.
        - 2 cams cùng nhìn 1 hướng nhưng cách xa → vẫn warp được nhiều pixels
          (chỉ scale/parallax thay đổi).
        - 2 cams gần nhau nhưng nhìn khác hướng → occlusion lớn → coverage thấp.

    Forward direction = cột 3 của C2W rotation = cam.R[:, 2]
    (Verified ở test 2: angular distances 2-7° giữa fern training cams.)
    """
    p_R = np.asarray(pseudo_cam.R, dtype=np.float64)
    p_fwd = p_R[:, 2]
    p_fwd = p_fwd / (np.linalg.norm(p_fwd) + 1e-12)

    best_cam = None
    best_cos = -2.0  # cos ∈ [-1,1], khởi tạo nhỏ hơn
    for cam in train_cameras:
        c_R = np.asarray(cam.R, dtype=np.float64)
        c_fwd = c_R[:, 2]
        c_fwd = c_fwd / (np.linalg.norm(c_fwd) + 1e-12)
        # Cos similarity càng lớn → góc càng nhỏ → càng gần
        cos = float(np.dot(p_fwd, c_fwd))
        if cos > best_cos:
            best_cos = cos
            best_cam = cam

    return best_cam


# ════════════════════════════════════════════════════════════
# FUNCTION 2 — forward_warp_depth
# ════════════════════════════════════════════════════════════
@torch.no_grad()
def forward_warp_depth(aligned_depth_A, cam_A, cam_P,
                       return_stats=False):
    """[CRSGaussian] Forward warp aligned depth từ cam A → cam P.

    Pipeline:
        1. Build grid pixels (u_A, v_A) của cam A
        2. Filter: chỉ giữ pixel có depth_A > 0
        3. Unproject (u, v, d) → 3D world dùng cam_A intrinsics + extrinsics
        4. Project 3D world → cam P pixel + depth
        5. Filter valid: depth_P > 0, in-frame [0,W)x[0,H)
        6. Sort theo depth_P descending → scatter → nearest depth wins
           (last write trong tensor scatter ghi đè giá trị cũ)
        7. Trả về depth_ref_P (H,W) với valid_mask_P

    Args:
        aligned_depth_A: (H,W) torch.Tensor metric depth tại cam A
        cam_A: training Camera object (có original_image, uid)
        cam_P: pseudo Camera object (chỉ có R, T, FoV, W, H)
        return_stats: nếu True trả về thêm dict thống kê collision

    Returns:
        depth_ref_P: (H_P, W_P) tensor — 0 tại holes
        valid_mask_P: (H_P, W_P) bool — True tại pixel có depth ref
        (optional) stats: dict với n_total, n_inframe, n_unique, collision_rate
    """
    device = aligned_depth_A.device
    dtype = aligned_depth_A.dtype

    # ── Intrinsics + extrinsics ──
    fx_A, fy_A, cx_A, cy_A, H_A, W_A = _cam_intrinsics(cam_A)
    fx_P, fy_P, cx_P, cy_P, H_P, W_P = _cam_intrinsics(cam_P)
    R_A, T_A = _cam_RT(cam_A, device, dtype)
    R_P, T_P = _cam_RT(cam_P, device, dtype)

    # ── Step 1: build pixel grid của cam A ──
    # meshgrid trả về (H,W) — flatten để vector hoá
    vv, uu = torch.meshgrid(
        torch.arange(H_A, device=device, dtype=dtype),
        torch.arange(W_A, device=device, dtype=dtype),
        indexing='ij',
    )
    u_flat = uu.reshape(-1)        # (H*W,)
    v_flat = vv.reshape(-1)
    d_flat = aligned_depth_A.reshape(-1)

    # ── Step 2: filter pixel có depth hợp lệ ──
    valid_A = d_flat > 1e-6
    if valid_A.sum() == 0:
        empty = torch.zeros(H_P, W_P, device=device, dtype=dtype)
        m = torch.zeros(H_P, W_P, device=device, dtype=torch.bool)
        if return_stats:
            return empty, m, dict(n_total=0, n_inframe=0, n_unique=0,
                                  collision_rate=0.0)
        return empty, m

    u_a = u_flat[valid_A]
    v_a = v_flat[valid_A]
    d_a = d_flat[valid_A]
    n_total = u_a.shape[0]

    # ── Step 3: unproject (u_A, v_A, d_A) → xyz_world ──
    # Pinhole inverse: x_cam = (u-cx)/fx*d, y_cam = (v-cy)/fy*d, z_cam = d
    x_cam = (u_a - cx_A) / fx_A * d_a
    y_cam = (v_a - cy_A) / fy_A * d_a
    z_cam = d_a
    xyz_cam_A = torch.stack([x_cam, y_cam, z_cam], dim=1)  # (n, 3)

    # C2W: xyz_world = R_A @ (xyz_cam - T_A)
    # cam.R là C2W rotation → tránh nhầm với W2C (R.T)
    xyz_world = (xyz_cam_A - T_A.unsqueeze(0)) @ R_A.T  # (n,3) — vì (R@v).T = v.T@R.T

    # ── Step 4: project xyz_world → cam P ──
    # W2C: xyz_cam_P = R_P.T @ xyz_world + T_P
    xyz_cam_P = xyz_world @ R_P + T_P.unsqueeze(0)  # equivalent to (R_P.T @ x.T).T + T
    # Note: (R.T @ x.T).T = x @ R, vì (AB)^T = B^T A^T, và (R.T @ x.T)^T = x @ R^T^T = x @ R

    z_P = xyz_cam_P[:, 2]
    # Clamp tránh chia 0 — clamp_min để giữ sign nhưng vẫn filter sau
    z_safe = torch.clamp(z_P, min=1e-6)
    u_P = fx_P * xyz_cam_P[:, 0] / z_safe + cx_P
    v_P = fy_P * xyz_cam_P[:, 1] / z_safe + cy_P

    # ── Step 5: filter in-frame + depth dương ──
    valid_P = (
        (z_P > 1e-6) &
        (u_P >= 0) & (u_P < W_P) &
        (v_P >= 0) & (v_P < H_P)
    )
    n_inframe = int(valid_P.sum().item())
    if n_inframe == 0:
        empty = torch.zeros(H_P, W_P, device=device, dtype=dtype)
        m = torch.zeros(H_P, W_P, device=device, dtype=torch.bool)
        if return_stats:
            return empty, m, dict(n_total=n_total, n_inframe=0, n_unique=0,
                                  collision_rate=0.0)
        return empty, m

    u_v = u_P[valid_P].long()
    v_v = v_P[valid_P].long()
    z_v = z_P[valid_P]

    # ── Step 6: collision resolution — nearest depth wins ──
    # Sort theo depth descending → khi scatter, depth nhỏ (gần) ghi đè cuối
    # → kết quả: pixel giữ depth nhỏ nhất = nearest, đúng vật lý occlusion.
    sort_idx = torch.argsort(z_v, descending=True)
    u_v = u_v[sort_idx]
    v_v = v_v[sort_idx]
    z_v = z_v[sort_idx]

    # Scatter
    depth_ref_P = torch.zeros(H_P, W_P, device=device, dtype=dtype)
    valid_mask_P = torch.zeros(H_P, W_P, device=device, dtype=torch.bool)
    # index_put_ với accumulate=False → last write wins (đúng với sort desc)
    depth_ref_P.index_put_((v_v, u_v), z_v, accumulate=False)
    valid_mask_P.index_put_(
        (v_v, u_v),
        torch.ones_like(z_v, dtype=torch.bool),
        accumulate=False,
    )

    if return_stats:
        n_unique = int(valid_mask_P.sum().item())
        collision_rate = (n_inframe - n_unique) / max(n_inframe, 1)
        stats = dict(
            n_total=n_total,
            n_inframe=n_inframe,
            n_unique=n_unique,
            collision_rate=float(collision_rate),
        )
        return depth_ref_P, valid_mask_P, stats

    return depth_ref_P, valid_mask_P


# ════════════════════════════════════════════════════════════
# FUNCTION 4 — warp_image_forward
# ════════════════════════════════════════════════════════════
@torch.no_grad()
def warp_image_forward(image_A, aligned_depth_A, cam_A, cam_P,
                       return_stats=False):
    """[CRSGaussian] Forward warp GT image từ training cam A → pseudo-cam P.

    Mục đích: tạo warped GT image làm reference photometric tại pseudo-cam.
    Khác với warp depth ở 1 điểm: scatter RGB thay vì depth.
    Collision resolution: nearest depth wins (occlusion-correct).

    Pipeline:
        1. Build pixel grid của cam A
        2. Filter pixels có depth > 0
        3. Unproject (u_A, v_A, d_A) → xyz_world (dùng cam_A.R, cam_A.T)
        4. Project xyz_world → cam P → (u_P, v_P, d_P)
        5. Filter valid: in-frame [0,W)x[0,H) + d_P > 0
        6. Sort theo d_P descending → scatter RGB → nearest wins
        7. Trả về (warped_image_P, valid_mask_P)

    Convention: cam.R = C2W rotation, cam.T = W2C translation
    (verified ở tests/verify_camera_convention.py)

    Args:
        image_A: (3,H,W) torch.Tensor RGB image tại cam A, range [0,1]
        aligned_depth_A: (H,W) torch.Tensor metric depth tại cam A
        cam_A: training Camera object
        cam_P: pseudo Camera object
        return_stats: nếu True trả thêm dict stats

    Returns:
        warped_image_P: (3, H_P, W_P) tensor — 0 tại holes
        valid_mask_P: (H_P, W_P) bool — True tại pixel có RGB ref
    """
    device = image_A.device
    dtype = image_A.dtype

    fx_A, fy_A, cx_A, cy_A, H_A, W_A = _cam_intrinsics(cam_A)
    fx_P, fy_P, cx_P, cy_P, H_P, W_P = _cam_intrinsics(cam_P)
    R_A, T_A = _cam_RT(cam_A, device, dtype)
    R_P, T_P = _cam_RT(cam_P, device, dtype)

    # ── Build pixel grid của cam A ──
    vv, uu = torch.meshgrid(
        torch.arange(H_A, device=device, dtype=dtype),
        torch.arange(W_A, device=device, dtype=dtype),
        indexing='ij',
    )
    u_flat = uu.reshape(-1)        # (H*W,)
    v_flat = vv.reshape(-1)
    d_flat = aligned_depth_A.reshape(-1)

    # Image: (3,H,W) → (H*W, 3)
    img_flat = image_A.permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3)

    # ── Filter pixel có depth hợp lệ ──
    valid_A = d_flat > 1e-6
    if valid_A.sum() == 0:
        empty_img = torch.zeros(3, H_P, W_P, device=device, dtype=dtype)
        empty_mask = torch.zeros(H_P, W_P, device=device, dtype=torch.bool)
        if return_stats:
            return empty_img, empty_mask, dict(n_total=0, n_inframe=0,
                                               n_unique=0, collision_rate=0.0)
        return empty_img, empty_mask

    u_a = u_flat[valid_A]
    v_a = v_flat[valid_A]
    d_a = d_flat[valid_A]
    rgb_a = img_flat[valid_A]      # (n, 3)
    n_total = u_a.shape[0]

    # ── Unproject → xyz_world ──
    x_cam = (u_a - cx_A) / fx_A * d_a
    y_cam = (v_a - cy_A) / fy_A * d_a
    z_cam = d_a
    xyz_cam_A = torch.stack([x_cam, y_cam, z_cam], dim=1)  # (n, 3)

    # C2W: xyz_world = R_A @ (xyz_cam - T_A); batch form: (xyz_cam - T) @ R.T
    xyz_world = (xyz_cam_A - T_A.unsqueeze(0)) @ R_A.T

    # ── Project xyz_world → cam P ──
    # W2C: xyz_cam_P = R_P.T @ xyz_world + T_P; batch: xyz_world @ R_P + T_P
    xyz_cam_P = xyz_world @ R_P + T_P.unsqueeze(0)

    z_P = xyz_cam_P[:, 2]
    z_safe = torch.clamp(z_P, min=1e-6)
    u_P = fx_P * xyz_cam_P[:, 0] / z_safe + cx_P
    v_P = fy_P * xyz_cam_P[:, 1] / z_safe + cy_P

    # ── Filter in-frame ──
    valid_P = (
        (z_P > 1e-6) &
        (u_P >= 0) & (u_P < W_P) &
        (v_P >= 0) & (v_P < H_P)
    )
    n_inframe = int(valid_P.sum().item())
    if n_inframe == 0:
        empty_img = torch.zeros(3, H_P, W_P, device=device, dtype=dtype)
        empty_mask = torch.zeros(H_P, W_P, device=device, dtype=torch.bool)
        if return_stats:
            return empty_img, empty_mask, dict(n_total=n_total, n_inframe=0,
                                               n_unique=0, collision_rate=0.0)
        return empty_img, empty_mask

    u_v = u_P[valid_P].long()
    v_v = v_P[valid_P].long()
    z_v = z_P[valid_P]
    rgb_v = rgb_a[valid_P]  # (n_inframe, 3)

    # ── Collision resolution: nearest depth wins ──
    # Sort descending → smallest depth ghi cuối → win.
    sort_idx = torch.argsort(z_v, descending=True)
    u_v = u_v[sort_idx]
    v_v = v_v[sort_idx]
    rgb_v = rgb_v[sort_idx]

    # Scatter RGB từng channel
    warped_image_P = torch.zeros(3, H_P, W_P, device=device, dtype=dtype)
    valid_mask_P = torch.zeros(H_P, W_P, device=device, dtype=torch.bool)
    for c in range(3):
        warped_image_P[c].index_put_((v_v, u_v), rgb_v[:, c], accumulate=False)
    valid_mask_P.index_put_(
        (v_v, u_v),
        torch.ones_like(z_v, dtype=torch.bool),
        accumulate=False,
    )

    if return_stats:
        n_unique = int(valid_mask_P.sum().item())
        collision_rate = (n_inframe - n_unique) / max(n_inframe, 1)
        stats = dict(
            n_total=n_total,
            n_inframe=n_inframe,
            n_unique=n_unique,
            collision_rate=float(collision_rate),
        )
        return warped_image_P, valid_mask_P, stats

    return warped_image_P, valid_mask_P


# ════════════════════════════════════════════════════════════
# FUNCTION 3 — compute_warp_coverage
# ════════════════════════════════════════════════════════════
def compute_warp_coverage(valid_mask):
    """[CRSGaussian] Tỉ lệ pixels có depth reference (sau warp).

    Threshold tham khảo:
        < 30%: warp coverage quá thấp → loss yếu, có thể bỏ iter này
        30-60%: marginal — vẫn có signal
        > 60%: coverage tốt → loss có signal mạnh
    """
    if valid_mask.numel() == 0:
        return 0.0
    return float(valid_mask.float().mean().item())
