# ============================================================
# [CRSGaussian Tier 2-min] D_cycle — cycle-depth consistency per Gaussian
# File: utils/crs/d_cycle.py  (TẠO MỚI)
# Mục đích: Replace D_DAV2 với multi-view geometric self-check.
#           Cho mỗi Gaussian, project xuống view A, sample rendered
#           depth, unproject world, project sang view B, sample
#           rendered depth, unproject lại, project về A. Cycle error
#           ||P_a - P_a'|| đo geometric consistency mà KHÔNG cần DAV2.
# Được gọi từ: crs_module.update_crs() khi opt.use_d_cycle=True
#              (sau opt.d_cycle_warmup) — cache mỗi opt.d_cycle_update_freq.
#
# Camera convention (verified từ source):
#   - scene/cameras.py:57: world_view_transform = getWorld2View2(R, T).T (đã transpose)
#   - utils/crs/crs_module.py pattern: W2C = cam.world_view_transform.T (T lại để có row-major W2C)
#     pts_cam = (W2C @ xyz_hom.T).T — column-vector convention sau transpose.
#   - utils/graphics_utils.py: image y-axis follows COLMAP (+y down trong image space).
#   - Depth z: positive forward (camera space, sau W2C); pixel filter dùng depth>0.
#   - fov2focal(fov, pixels) = pixels / (2 * tan(fov/2)); cx=W/2, cy=H/2 (no shift).
#   - Image bounds: 0 ≤ pixel_x < W, 0 ≤ pixel_y < H.
# ============================================================

import torch
from itertools import combinations
from utils.graphics_utils import fov2focal


@torch.no_grad()
def _project_world_to_camera(xyz_hom, cam):
    """World → camera-space coords (theo pattern crs_module.py)."""
    # cam.world_view_transform stored transposed → .T trả về W2C row-major (4,4).
    W2C = cam.world_view_transform.T
    # (4,4) @ (4,N) → (4,N) → transpose về (N,4)
    pts_cam = (W2C @ xyz_hom.T).T
    return pts_cam


@torch.no_grad()
def _camera_to_pixel(pts_cam, cam):
    """Camera-space → pixel coords + raw depth z.

    Returns:
        pixel_x, pixel_y, depth: tensors (N,) float
    """
    H = cam.image_height
    W = cam.image_width
    fx = fov2focal(cam.FoVx, W)
    fy = fov2focal(cam.FoVy, H)
    cx, cy = W / 2.0, H / 2.0

    depth = pts_cam[:, 2]
    # Avoid div-by-zero — clamp depth tối thiểu (sẽ bị filter bằng valid mask).
    safe_depth = torch.where(depth.abs() < 1e-6, torch.full_like(depth, 1e-6), depth)
    pixel_x = pts_cam[:, 0] / safe_depth * fx + cx
    pixel_y = pts_cam[:, 1] / safe_depth * fy + cy
    return pixel_x, pixel_y, depth


@torch.no_grad()
def _sample_depth_map(depth_map, pixel_x, pixel_y, valid_mask):
    """Bilinear sample depth_map [1,H,W] (hoặc [H,W]) tại (pixel_x, pixel_y).

    Args:
        depth_map: rendered depth từ rasterizer, shape [1,H,W] hoặc [H,W].
        pixel_x, pixel_y: (M,) float pixel coords (đã filter valid).
        valid_mask: (M,) bool — đảm bảo pixel trong bounds.

    Returns:
        d_sampled: (M,) tensor — depth lookup (giá trị 0 cho invalid sample).
    """
    if depth_map.dim() == 3:
        depth_map = depth_map.squeeze(0)  # (H, W)
    H, W = depth_map.shape

    # F.grid_sample dùng normalized coords [-1, 1]. Build grid (1, M, 1, 2).
    # x_norm = 2*x/(W-1) - 1, y_norm tương tự.
    # Clamp pixel in-bounds trước khi normalize (ngoài bounds → grid_sample
    # padding_mode='zeros' trả 0 → ổn vì invalid mask sẽ filter sau).
    x_norm = 2.0 * pixel_x / max(W - 1, 1) - 1.0
    y_norm = 2.0 * pixel_y / max(H - 1, 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)  # (1, M, 1, 2)

    # depth_map [H, W] → [1, 1, H, W] cho grid_sample
    dm = depth_map.unsqueeze(0).unsqueeze(0)
    sampled = torch.nn.functional.grid_sample(
        dm, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )  # (1, 1, M, 1)
    return sampled.view(-1)


@torch.no_grad()
def _unproject_pixel_depth_to_world(pixel_x, pixel_y, depth, cam):
    """Pixel + depth → world coords.

    Inverse của _project_world_to_camera ∘ _camera_to_pixel:
      pts_cam = [(px-cx)/fx * z, (py-cy)/fy * z, z, 1]
      pts_world = (C2W @ pts_cam.T).T[:, :3]
    """
    H = cam.image_height
    W = cam.image_width
    fx = fov2focal(cam.FoVx, W)
    fy = fov2focal(cam.FoVy, H)
    cx, cy = W / 2.0, H / 2.0

    x_cam = (pixel_x - cx) / fx * depth
    y_cam = (pixel_y - cy) / fy * depth
    z_cam = depth
    M = pixel_x.shape[0]
    ones = torch.ones(M, device=pixel_x.device, dtype=pixel_x.dtype)
    pts_cam_hom = torch.stack([x_cam, y_cam, z_cam, ones], dim=1)  # (M, 4)

    # C2W = inverse(W2C). cam.world_view_transform.T = W2C → inverse → C2W.
    W2C = cam.world_view_transform.T
    C2W = torch.inverse(W2C)
    pts_world_hom = (C2W @ pts_cam_hom.T).T  # (M, 4)
    return pts_world_hom[:, :3]


@torch.no_grad()
def _world_to_pixel(pts_world, cam):
    """World coords (M, 3) → pixel coords (M,) + depth (M,) + in-bounds mask."""
    M = pts_world.shape[0]
    ones = torch.ones(M, device=pts_world.device, dtype=pts_world.dtype)
    xyz_hom = torch.cat([pts_world, ones.unsqueeze(-1)], dim=1)  # (M, 4)
    pts_cam = _project_world_to_camera(xyz_hom, cam)
    px, py, depth = _camera_to_pixel(pts_cam, cam)
    H, W = cam.image_height, cam.image_width
    in_bounds = (
        (depth > 0)
        & (px >= 0) & (px < W)
        & (py >= 0) & (py < H)
    )
    return px, py, depth, in_bounds


@torch.no_grad()
def compute_D_cycle(
    gaussians,
    train_cameras: list,
    render_func,
    pipe,
    bg,
    sigma: float = 5.0,
    depth_maps: dict = None,
):
    """Compute D_cycle per Gaussian — cycle-depth consistency over view pairs.

    Args:
        depth_maps: optional pre-rendered {cam.uid: tensor} dict. Khi None,
            hàm tự render. Dùng để tránh double-render khi caller (update_crs)
            đã render cho R_visible hoặc khác. [Phase 8a optimization]

    Returns:
        D: (N, 1) tensor float32 GPU, range [0, 1].
           D ≈ 1.0: cycle close → geometry consistent (surface).
           D ≈ 0.0: cycle break → floater hoặc occlusion.
           D = 0.5: invisible từ < 2 cameras (neutral, không đủ pair để cycle).
    """
    xyz = gaussians.get_xyz.detach()
    N = xyz.shape[0]
    device = xyz.device

    # Lọc training cams (skip pseudo nếu có)
    valid_cams = [c for c in train_cameras if hasattr(c, 'original_image')]
    K = len(valid_cams)
    if K < 2:
        return torch.full((N, 1), 0.5, device=device)

    # ── Step 1: render depth maps (hoặc reuse pre-rendered) ──
    # disable_dropout=True để render full-size, depth không bị thiếu Gaussian.
    if depth_maps is None:
        depth_maps = {}
        for cam in valid_cams:
            try:
                pkg = render_func(cam, gaussians, pipe, bg, disable_dropout=True)
            except TypeError:
                pkg = render_func(cam, gaussians, pipe, bg)
            depth_maps[cam.uid] = pkg["depth"].detach()  # (1, H, W)

    # ── Step 2: cycle test trên từng cặp views ──
    # Cho mỗi cặp (a, b) trong unordered pairs:
    #   P_a = project(xyz → cam_a), valid_a = depth>0 & in-bounds
    #   d_a = sample_depth(depth_map_a, P_a)
    #   P_world_a = unproject(P_a, d_a, cam_a)
    #   P_b = project(P_world_a → cam_b), valid_b
    #   d_b = sample_depth(depth_map_b, P_b)
    #   P_world_b = unproject(P_b, d_b, cam_b)
    #   P_a' = project(P_world_b → cam_a), valid_back
    #   cycle_error = ||P_a − P_a'|| (Euclidean pixel distance)
    # Aggregate: mean cycle_error across valid pairs per Gaussian.
    err_sum = torch.zeros(N, device=device)
    pair_count = torch.zeros(N, device=device)

    ones_col = torch.ones(N, 1, device=device, dtype=xyz.dtype)
    xyz_hom = torch.cat([xyz, ones_col], dim=1)  # (N, 4)

    for cam_a, cam_b in combinations(valid_cams, 2):
        H_a, W_a = cam_a.image_height, cam_a.image_width
        H_b, W_b = cam_b.image_height, cam_b.image_width

        # — Project Gaussian → cam_a —
        pts_cam_a = _project_world_to_camera(xyz_hom, cam_a)
        px_a, py_a, dz_a = _camera_to_pixel(pts_cam_a, cam_a)
        valid_a = (
            (dz_a > 0)
            & (px_a >= 0) & (px_a < W_a)
            & (py_a >= 0) & (py_a < H_a)
        )
        if valid_a.sum() == 0:
            continue

        # — Sample rendered depth tại pixel_a —
        # Filter trước để giảm cost grid_sample.
        idx_a = torch.where(valid_a)[0]
        d_a_sampled = _sample_depth_map(
            depth_maps[cam_a.uid], px_a[idx_a], py_a[idx_a], valid_a[idx_a]
        )
        # Pixel có rendered_depth ≤ 0 (chưa cover) → loại khỏi cycle test.
        valid_d_a = d_a_sampled > 1e-6
        if valid_d_a.sum() == 0:
            continue
        idx_a = idx_a[valid_d_a]
        d_a_sampled = d_a_sampled[valid_d_a]

        # — Unproject (P_a, d_a) → world coord W_a (sample-anchored, không phải gaussian-anchored) —
        # Dùng rendered_depth (surface frontmost) để cycle test geometry CỦA SURFACE.
        # Floater Gaussian project ra pixel → surface_world_a có thể khác Gaussian's true position
        # → khi cycle về cam_a có thể lệch → cycle_error lớn → D_cycle thấp.
        P_world_a = _unproject_pixel_depth_to_world(
            px_a[idx_a], py_a[idx_a], d_a_sampled, cam_a
        )

        # — Project W_a → cam_b —
        px_b, py_b, dz_b, valid_b = _world_to_pixel(P_world_a, cam_b)
        if valid_b.sum() == 0:
            continue
        idx_b = torch.where(valid_b)[0]
        d_b_sampled = _sample_depth_map(
            depth_maps[cam_b.uid], px_b[idx_b], py_b[idx_b], valid_b[idx_b]
        )
        valid_d_b = d_b_sampled > 1e-6
        if valid_d_b.sum() == 0:
            continue
        idx_b = idx_b[valid_d_b]
        d_b_sampled = d_b_sampled[valid_d_b]

        # — Unproject (P_b, d_b) → world coord W_b —
        P_world_b = _unproject_pixel_depth_to_world(
            px_b[idx_b], py_b[idx_b], d_b_sampled, cam_b
        )

        # — Project W_b → cam_a (cycle back) —
        px_a_back, py_a_back, dz_a_back, valid_back = _world_to_pixel(P_world_b, cam_a)
        if valid_back.sum() == 0:
            continue

        # — Compute cycle error trên các Gaussian survived 3-step cycle —
        # idx_a là indexing relative đến ban đầu N. Phải reduce qua các valid.
        # Pattern: idx_a → idx_b (subset trên top idx_a). idx_b chỉ vào trong idx_a → cần map.
        # Solution: track final_idx = idx_a[valid_d_b] sau bước b.
        # Nhưng valid_back lại trên top final_idx — phải apply 1 lần nữa.
        # Đơn giản hơn: track running indices tại mỗi step.
        # Hiện tại idx_b đã point vào subset của idx_a (sau valid_b ∧ valid_d_b).
        # Vì valid_b/d_b/back áp lên (M, ) shape của idx_a[valid_d_a] → cần map lại.
        # Reset chain:
        #   step a: idx_global = idx_a (sau valid_a ∧ valid_d_a)  shape (Ma,)
        #   step b: idx_global = idx_a[idx_b] (sau valid_b ∧ valid_d_b)  shape (Mb,)
        # Tại step back: idx_global ở chain có shape Mb; valid_back shape (Mb,)
        idx_global_after_b = idx_a[idx_b]  # (Mb,) global indices
        idx_back = torch.where(valid_back)[0]  # local indices into Mb
        idx_global_final = idx_global_after_b[idx_back]  # (Mfinal,) global
        if idx_global_final.numel() == 0:
            continue

        # Original P_a tại các Gaussian này (lookup từ px_a, py_a originally)
        px_a_orig_final = px_a[idx_global_final]
        py_a_orig_final = py_a[idx_global_final]
        # Cycled-back P_a' (đã filter tới chain Mb → idx_back chọn ra Mfinal)
        px_a_back_final = px_a_back[idx_back]
        py_a_back_final = py_a_back[idx_back]

        cycle_err = torch.sqrt(
            (px_a_orig_final - px_a_back_final).pow(2)
            + (py_a_orig_final - py_a_back_final).pow(2)
            + 1e-12
        )

        # Accumulate
        err_sum[idx_global_final] += cycle_err
        pair_count[idx_global_final] += 1.0

    # ── Step 3: aggregate per Gaussian ──
    # Gaussian có ≥1 valid pair → D = exp(-mean_err / sigma).
    # Gaussian không có valid pair → neutral 0.5.
    has_pair = pair_count > 0
    mean_err = torch.zeros_like(err_sum)
    mean_err[has_pair] = err_sum[has_pair] / pair_count[has_pair]

    D = torch.full((N, 1), 0.5, device=device)
    D[has_pair, 0] = torch.exp(-mean_err[has_pair] / max(sigma, 1e-6)).clamp(0.0, 1.0)
    return D
