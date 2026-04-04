# ============================================================
# [CRSGaussian] Task: T1.2 — depth_alignment.py
# File: CRSGaussian/utils/depth/depth_alignment.py  (TẠO MỚI)
# Mục đích: Align relative depth (từ DAV2) với COLMAP sparse
#           depth dùng closed-form weighted least squares.
#           Weight = 1/reprojection_error → SfM points tin cậy
#           được tin nhiều hơn.
# Được gọi từ: train.py, 1 lần trước training loop,
#              sau precompute_depth_priors()
# ============================================================

import os
import torch
import numpy as np
from utils.graphics_utils import fov2focal
from scene.colmap_loader import read_points3D_binary, read_points3D_text


def align_depth_to_colmap(depth_prior_dict, cameras, source_path, n_views,
                          max_reproj_err=2.5, depth_range_percentile=0.5):
    """Align mỗi relative depth map với COLMAP sparse points.

    Tìm scale, shift sao cho:
        aligned_depth = scale * relative_depth + shift ≈ COLMAP depth
    Dùng closed-form weighted least squares, weight = 1/reproj_error.

    Args:
        depth_prior_dict: {cam.uid: Tensor (H,W)} từ depth_model.py
        cameras: list of Camera objects
        source_path: đường dẫn dataset (args.source_path)
        n_views: số views (args.n_views), dùng để tìm đúng points3D.bin
        max_reproj_err: bỏ COLMAP points có error > ngưỡng này (pixel)
        depth_range_percentile: bỏ % extreme depths ở 2 đầu

    Returns:
        aligned_depth_dict: {cam.uid: Tensor (H,W) float32 CPU}
            Aligned depth — scale khớp với COLMAP metric space.
        depth_range: float — median(far) - median(near) across cameras.
            Dùng cho epsilon_depth và normalize D_i sau này.
    """
    # -- Đọc COLMAP 3D points + reprojection errors --
    sparse_xyz, sparse_errors = _load_colmap_points(source_path, n_views)

    # -- Filter outlier points theo reprojection error --
    # Points có error cao là unreliable → bỏ trước khi align
    valid_mask = sparse_errors.squeeze() <= max_reproj_err
    sparse_xyz = sparse_xyz[valid_mask]
    sparse_errors = sparse_errors[valid_mask]
    print(f"[CRSGaussian] COLMAP points: {valid_mask.sum()}/{len(valid_mask)} "
          f"passed reproj_err filter (<= {max_reproj_err}px)")

    # -- Align từng camera --
    aligned_depth_dict = {}
    all_near, all_far = [], []

    for cam in cameras:
        if cam.uid not in depth_prior_dict:
            continue

        dense_depth = depth_prior_dict[cam.uid]  # (H, W) relative depth
        H, W = dense_depth.shape

        # Project COLMAP 3D points vào camera → pixel coords + depth
        pixels, depths_sparse = _project_points_to_camera(
            sparse_xyz, cam, H, W)

        if len(depths_sparse) < 3:
            # Quá ít points → không align được, giữ nguyên relative depth
            print(f"[CRSGaussian] WARNING: cam {cam.uid} has <3 visible "
                  f"COLMAP points, skipping alignment")
            aligned_depth_dict[cam.uid] = dense_depth.cuda()
            continue

        # Lấy dense depth tại pixel locations tương ứng
        px = pixels[:, 0].long().clamp(0, W - 1)
        py = pixels[:, 1].long().clamp(0, H - 1)
        depths_dense = dense_depth[py, px]

        # Lấy weights cho visible points
        # _project_points_to_camera đã filter → cần map lại errors
        weights = _get_weights_for_visible(
            sparse_xyz, sparse_errors, cam, H, W)

        # Filter extreme depth values (outlier pruning 2 đầu)
        # Tương tự DepthRegularizedGS: bỏ 0.5% nhỏ nhất và lớn nhất
        if len(depths_sparse) > 20:
            sorted_d = torch.sort(depths_sparse).values
            n = len(sorted_d)
            lo = sorted_d[int(n * depth_range_percentile / 100)]
            hi = sorted_d[int(n * (1 - depth_range_percentile / 100))]
            keep = (depths_sparse >= lo) & (depths_sparse <= hi)
            depths_sparse = depths_sparse[keep]
            depths_dense = depths_dense[keep]
            weights = weights[keep]

        # Closed-form weighted least squares
        # Giải: scale * dense + shift ≈ sparse (weighted)
        scale, shift = _closed_form_wls(depths_dense, depths_sparse, weights)

        # Align toàn bộ dense depth map
        aligned = scale * dense_depth + shift
        aligned = aligned.clamp(min=1e-6)  # tránh depth ≤ 0

        # Lưu trên GPU — tránh CPU→GPU transfer mỗi iter khi dùng trong
        # depth loss (T3.3) và CRS update (T2.6).
        aligned_depth_dict[cam.uid] = aligned.cuda()

        # Thu thập near/far cho depth_range
        all_near.append(aligned.min().item())
        all_far.append(aligned.max().item())

        print(f"[CRSGaussian] cam {cam.uid}: scale={scale:.4f}, "
              f"shift={shift:.4f}, #pts={len(depths_sparse)}")

    # -- depth_range = median(far) - median(near) --
    # Dùng median robust với outlier cameras
    if all_near and all_far:
        depth_range = float(np.median(all_far) - np.median(all_near))
    else:
        depth_range = 1.0  # fallback
    print(f"[CRSGaussian] depth_range = {depth_range:.4f}")

    return aligned_depth_dict, depth_range


# ============================================================
# Internal helpers
# ============================================================

def _load_colmap_points(source_path, n_views):
    """Đọc COLMAP 3D points + reprojection errors từ points3D.bin.

    Thử nhiều path theo convention của CRSGaussian dataset_readers.py:
    1. {source_path}/{n_views}_views/triangulated/points3D.bin  (LLFF n_views>0)
    2. {source_path}/sparse/0/points3D.bin                      (fallback)
    """
    candidates = []
    if n_views > 0:
        candidates.append(os.path.join(
            source_path, f"{n_views}_views", "triangulated", "points3D.bin"))
    candidates.append(os.path.join(source_path, "sparse", "0", "points3D.bin"))

    for bin_path in candidates:
        if os.path.exists(bin_path):
            xyz, _, errors = read_points3D_binary(bin_path)
            # xyz: (N, 3) float64, errors: (N, 1) float64
            return (torch.from_numpy(xyz).float(),
                    torch.from_numpy(errors).float())

    # Thử .txt fallback
    txt_candidates = [p.replace('.bin', '.txt') for p in candidates]
    for txt_path in txt_candidates:
        if os.path.exists(txt_path):
            xyz, _, errors = read_points3D_text(txt_path)
            return (torch.from_numpy(xyz).float(),
                    torch.from_numpy(errors).float())

    raise FileNotFoundError(
        f"Cannot find points3D.bin or .txt in any of: {candidates}")


def _project_points_to_camera(points_world, cam, H, W):
    """Project 3D world points vào camera image plane.

    Dùng cam.world_view_transform (đã bao gồm translate+scale từ
    getWorld2View2) thay vì tự tính R.T @ P + T.

    Args:
        points_world: (N, 3) tensor float
        cam: Camera object
        H, W: image dimensions

    Returns:
        pixels: (M, 2) tensor float — [x, y] pixel coords (valid only)
        depths: (M,) tensor float — depth values in camera space
    """
    # world_view_transform lưu column-major (transposed) trên cuda
    # .T chuyển về row-major: W2C (4x4) chuẩn
    W2C = cam.world_view_transform.cpu().float().T  # (4, 4) row-major

    # Homogeneous coords: (N, 4)
    N = points_world.shape[0]
    ones = torch.ones(N, 1, dtype=torch.float32)
    pts_hom = torch.cat([points_world, ones], dim=1)  # (N, 4)

    # World → Camera: (N, 4)
    pts_cam = (W2C @ pts_hom.T).T  # (N, 4)

    # Camera intrinsics từ FoV
    fx = fov2focal(cam.FoVx, W)
    fy = fov2focal(cam.FoVy, H)
    cx, cy = W / 2.0, H / 2.0

    depth = pts_cam[:, 2]

    # Project to pixel
    pixel_x = pts_cam[:, 0] / depth * fx + cx
    pixel_y = pts_cam[:, 1] / depth * fy + cy

    # Filter: depth > 0 và trong frame
    valid = ((depth > 0) &
             (pixel_x >= 0) & (pixel_x < W) &
             (pixel_y >= 0) & (pixel_y < H))

    pixels = torch.stack([pixel_x[valid], pixel_y[valid]], dim=1)  # (M, 2)
    depths = depth[valid]  # (M,)

    return pixels, depths


def _get_weights_for_visible(points_world, errors, cam, H, W):
    """Lấy weight = 1/error cho points visible trong camera.

    Cùng logic filter như _project_points_to_camera để đảm bảo
    index mapping khớp.
    """
    W2C = cam.world_view_transform.cpu().float().T
    N = points_world.shape[0]
    ones = torch.ones(N, 1, dtype=torch.float32)
    pts_hom = torch.cat([points_world, ones], dim=1)
    pts_cam = (W2C @ pts_hom.T).T

    fx = fov2focal(cam.FoVx, W)
    fy = fov2focal(cam.FoVy, H)
    cx, cy = W / 2.0, H / 2.0

    depth = pts_cam[:, 2]
    pixel_x = pts_cam[:, 0] / depth * fx + cx
    pixel_y = pts_cam[:, 1] / depth * fy + cy

    valid = ((depth > 0) &
             (pixel_x >= 0) & (pixel_x < W) &
             (pixel_y >= 0) & (pixel_y < H))

    # Weight = 1 / error, errors shape (N, 1)
    err = errors[valid].squeeze()  # (M,)
    # Clamp error tối thiểu để tránh chia cho ~0
    err = err.clamp(min=0.1)
    weights = 1.0 / err

    # Normalize về [0, 1] — weight lớn nhất = 1.0
    if weights.max() > 0:
        weights = weights / weights.max()

    return weights


def _closed_form_wls(prediction, target, weights):
    """Closed-form weighted least squares cho scale + shift.

    Giải hệ 2x2:
        [Σw*x²  Σw*x] [s]   [Σw*x*y]
        [Σw*x   Σw  ] [t] = [Σw*y  ]

    Trong đó x = prediction (dense), y = target (sparse), w = weights.
    Adapted from MonoSDF (qua dn-splatter/align_depth.py).

    Args:
        prediction: (M,) tensor — dense depth tại sparse pixel locations
        target: (M,) tensor — COLMAP sparse depth
        weights: (M,) tensor — alignment weights

    Returns:
        scale, shift: float
    """
    w = weights
    x = prediction
    y = target

    # Hệ phương trình Ax = b
    a_00 = torch.sum(w * x * x)
    a_01 = torch.sum(w * x)
    a_11 = torch.sum(w)
    b_0 = torch.sum(w * x * y)
    b_1 = torch.sum(w * y)

    # Determinant
    det = a_00 * a_11 - a_01 * a_01

    if det.abs() < 1e-8:
        # Ma trận suy biến → fallback: scale=1, shift=0
        print("[CRSGaussian] WARNING: WLS determinant near zero, "
              "using identity alignment")
        return 1.0, 0.0

    scale = (a_11 * b_0 - a_01 * b_1) / det
    shift = (-a_01 * b_0 + a_00 * b_1) / det

    return scale.item(), shift.item()
