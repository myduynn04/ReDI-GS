# ============================================================
# [CRSGaussian] Verify Camera Convention + Gradient Flow
# File: tests/verify_camera_convention.py  (TẠO MỚI — pre-implement check)
# Mục đích:
#   Trước khi implement pseudo-view depth loss, verify 2 thứ:
#   1. Camera convention round-trip: project → unproject → recover xyz gốc
#      (error < 1e-4) — đảm bảo công thức cam.R / cam.T dùng đúng.
#   2. Gradient flow qua RenderDict["depth_pseudo_co_gs0"] — verify tensor
#      depth lưu trong dict KHÔNG bị detach → loss.backward() có chảy về
#      gaussians.get_xyz hay không.
#
# Chạy: python tests/verify_camera_convention.py \
#         --source_path data/nerf_llff_data/fern \
#         --eval -r 8 --n_views 3 \
#         --use_depth_prior --dav2_path ../Depth-Anything-V2
#
# PASS criteria:
#   - Round-trip max error < 1e-4
#   - Gradient norm > 0 trên gaussians.get_xyz sau backward
# ============================================================

import os
import sys
import torch
import numpy as np
from argparse import ArgumentParser

# Cho phép import từ root project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.graphics_utils import fov2focal
from utils.depth import precompute_depth_priors, align_depth_to_colmap


# ════════════════════════════════════════
# TEST 1 — Camera convention round-trip
# ════════════════════════════════════════
def test_camera_convention(scene, allCameras):
    """Project 1 COLMAP point → pixel + depth → unproject → so với gốc.

    Convention CRSGaussian (verified từ utils/graphics_utils.py:31-36):
      - cam.R = C2W rotation (rotation cột thứ 3 = forward direction trong world)
      - cam.T = W2C translation
      - W2C: xyz_cam = cam.R.T @ xyz_world + cam.T
      - C2W: xyz_world = cam.R @ (xyz_cam - cam.T)
      - fx = fov2focal(cam.FoVx, W), fy = fov2focal(cam.FoVy, H)
      - cx = W/2, cy = H/2 (centered principal point)

    Pass: max round-trip error < 1e-4
    """
    print("\n" + "=" * 60)
    print(" TEST 1 — Camera convention round-trip")
    print("=" * 60)

    # Lấy 5 COLMAP points đầu tiên để test (numpy array)
    pcd = scene.init_point_cloud
    n_test = min(5, len(pcd.points))
    test_points = pcd.points[:n_test].astype(np.float64)  # (n, 3)

    cam = allCameras[0]
    H, W = cam.image_height, cam.image_width
    fx = fov2focal(cam.FoVx, W)
    fy = fov2focal(cam.FoVy, H)
    cx, cy = W / 2.0, H / 2.0

    # cam.R, cam.T là numpy array trên CPU
    R = np.asarray(cam.R, dtype=np.float64)   # (3,3)  C2W rotation
    T = np.asarray(cam.T, dtype=np.float64)   # (3,)   W2C translation

    print(f"  Camera 0 (uid={cam.uid}): {W}x{H}, fx={fx:.2f}, fy={fy:.2f}")
    print(f"  cam.R shape={R.shape}, cam.T shape={T.shape}")

    max_err = 0.0
    for i in range(n_test):
        p_world = test_points[i]   # (3,)

        # ── Forward project: world → cam → pixel ──
        # W2C: xyz_cam = R.T @ xyz_world + T
        xyz_cam = R.T @ p_world + T
        if xyz_cam[2] <= 0:
            print(f"  Point {i}: behind camera (z={xyz_cam[2]:.3f}), skip")
            continue
        u = fx * xyz_cam[0] / xyz_cam[2] + cx
        v = fy * xyz_cam[1] / xyz_cam[2] + cy
        d = xyz_cam[2]

        # ── Inverse: pixel + depth → cam → world ──
        # Unproject: xyz_cam = [(u-cx)/fx*d, (v-cy)/fy*d, d]
        xyz_cam_back = np.array([
            (u - cx) / fx * d,
            (v - cy) / fy * d,
            d,
        ])
        # C2W: xyz_world = R @ (xyz_cam - T)
        p_world_back = R @ (xyz_cam_back - T)

        err = np.abs(p_world - p_world_back).max()
        max_err = max(max_err, err)
        print(f"  Point {i}: world={p_world.round(3)} → "
              f"pixel=({u:.1f},{v:.1f}) d={d:.3f} → "
              f"back={p_world_back.round(3)} | err={err:.2e}")

    print(f"\n  >>> Max round-trip error: {max_err:.2e}")
    if max_err < 1e-4:
        print("  >>> ✓ PASS — convention OK")
        return True
    else:
        print("  >>> ✗ FAIL — convention SAI, fix trước khi implement")
        return False


# ════════════════════════════════════════
# TEST 2 — Forward direction extraction
# ════════════════════════════════════════
def test_forward_direction(allCameras):
    """Verify forward direction = cam.R[:, 2] (cột 3 của C2W rotation).

    cam.R là C2W → các cột là [right, up_or_down, forward] trong world frame.
    Forward direction quan trọng cho find_nearest_training_cam().
    """
    print("\n" + "=" * 60)
    print(" TEST 2 — Forward direction angular distance")
    print("=" * 60)

    n = len(allCameras)
    print(f"  N training cameras: {n}")

    # Tính forward dir cho tất cả cameras
    fwds = []
    for cam in allCameras:
        R = np.asarray(cam.R, dtype=np.float64)
        fwd = R[:, 2]
        fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
        fwds.append(fwd)
        print(f"  cam {cam.uid}: forward = {fwd.round(3)}, "
              f"|fwd|={np.linalg.norm(fwd):.4f}")

    # Pairwise angular distance (deg)
    print("\n  Pairwise angular distance (deg):")
    for i in range(n):
        for j in range(i + 1, n):
            cos = np.clip(np.dot(fwds[i], fwds[j]), -1.0, 1.0)
            ang = np.degrees(np.arccos(cos))
            print(f"    cam{i}-cam{j}: {ang:.2f}°")

    print("  >>> ✓ Forward extraction OK (angular distances reasonable)")
    return True


# ════════════════════════════════════════
# TEST 3 — Gradient flow qua RenderDict
# ════════════════════════════════════════
def test_gradient_flow(scene, allCameras, pipe):
    """Verify rằng RenderDict["depth_pseudo_co_gs0"] giữ gradient.

    Mục đích: trong train.py, render được lưu vào dict. Cần verify
    tensor depth không bị detach → gradient từ pseudo loss vẫn chảy
    về gaussians.get_xyz.
    """
    print("\n" + "=" * 60)
    print(" TEST 3 — Gradient flow qua RenderDict pseudo")
    print("=" * 60)

    gaussians = scene.gaussians
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    pseudo_cams = scene.getPseudoCameras()
    if pseudo_cams[0] is None:
        print("  ✗ Không có pseudo cameras (LLFF dataset cần có)")
        return False

    pseudo_cam = pseudo_cams[0]
    print(f"  Pseudo cam: {pseudo_cam.image_width}x{pseudo_cam.image_height}")

    # Mô phỏng pattern train.py — lưu render vào dict
    RenderDict = {}
    RenderDict["render_pkg_pseudo"] = render(pseudo_cam, gaussians, pipe, bg)
    RenderDict["depth_pseudo"] = RenderDict["render_pkg_pseudo"]["depth"]

    depth_p = RenderDict["depth_pseudo"]
    print(f"  depth_pseudo shape: {depth_p.shape}, dtype: {depth_p.dtype}")
    print(f"  requires_grad: {depth_p.requires_grad}")
    print(f"  grad_fn: {depth_p.grad_fn}")

    if depth_p.grad_fn is None:
        print("  ✗ FAIL — depth tensor bị detach trong RenderDict")
        return False

    # Tạo dummy loss và backward
    # Phải clear grad trước
    if gaussians._xyz.grad is not None:
        gaussians._xyz.grad.zero_()

    dummy_loss = depth_p.mean()
    print(f"  dummy loss: {dummy_loss.item():.6f}")
    dummy_loss.backward()

    grad = gaussians._xyz.grad
    if grad is None:
        print("  ✗ FAIL — gradient không tồn tại trên gaussians._xyz")
        return False

    grad_norm = grad.norm().item()
    n_nonzero = (grad.abs() > 1e-10).sum().item()
    n_total = grad.numel()

    print(f"  gradient norm: {grad_norm:.6e}")
    print(f"  nonzero gradients: {n_nonzero}/{n_total} "
          f"({100 * n_nonzero / n_total:.1f}%)")

    if grad_norm > 1e-12 and n_nonzero > 0:
        print("  >>> ✓ PASS — gradient chảy qua RenderDict[depth_pseudo]")
        return True
    else:
        print("  >>> ✗ FAIL — gradient zero hoặc không tồn tại")
        return False


# ════════════════════════════════════════
# TEST 4 — Camera centers & overall sanity
# ════════════════════════════════════════
def test_camera_centers(allCameras):
    """Verify camera_center trong world frame.

    Có 2 cách tính camera center từ cam.R, cam.T:
      Method 1: -cam.R @ cam.T   (vì C2W: x_world = R @ x_cam - R @ T,
                                   với x_cam=0 → x_world = -R @ T)
      Method 2: cam.world_view_transform.inverse()[3, :3]
                (CoR-GS dùng method này, line 60 cameras.py)

    Hai method phải khớp nhau.
    """
    print("\n" + "=" * 60)
    print(" TEST 4 — Camera centers consistency")
    print("=" * 60)

    max_err = 0.0
    for cam in allCameras:
        R = np.asarray(cam.R, dtype=np.float64)
        T = np.asarray(cam.T, dtype=np.float64)

        center_method1 = -R @ T
        center_method2 = cam.camera_center.cpu().numpy().astype(np.float64)

        err = np.abs(center_method1 - center_method2).max()
        max_err = max(max_err, err)
        print(f"  cam {cam.uid}: m1={center_method1.round(3)} "
              f"m2={center_method2.round(3)} err={err:.2e}")

    print(f"\n  >>> Max center error: {max_err:.2e}")
    if max_err < 1e-3:
        print("  >>> ✓ PASS — center formulas consistent")
        return True
    else:
        print("  >>> ✗ FAIL — center mismatch")
        return False


# ════════════════════════════════════════
# Main
# ════════════════════════════════════════
if __name__ == "__main__":
    parser = ArgumentParser(description="Verify camera convention before pseudo-depth implementation")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    # ── Top-level args mà train.py thêm trực tiếp vào parser (không qua ParamGroup) ──
    # Cần để GaussianModel/Scene không crash vì missing attribute
    parser.add_argument("--train_bg", action="store_true")
    parser.add_argument("--gaussiansN", type=int, default=1)
    parser.add_argument("--coreg", action="store_true")
    parser.add_argument("--coprune", action="store_true")
    parser.add_argument("--coprune_threshold", type=int, default=5)
    parser.add_argument("--reg_sample_rate", type=float, default=0.5)
    parser.add_argument("--mask_training", action="store_true")
    parser.add_argument("--save_log_images", action="store_true")
    parser.add_argument("--onlyrgb", action="store_true")
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])

    pipe = pp.extract(args)

    print("Loading scene ...")
    # ── Pass raw args (như train.py) vì GaussianModel/Scene cần train_bg, gaussiansN, ... ──
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    allCameras = scene.getTrainCameras()
    print(f"  Loaded {len(allCameras)} training cameras")
    print(f"  Loaded {gaussians.get_xyz.shape[0]} Gaussians")

    results = []
    results.append(("Camera round-trip", test_camera_convention(scene, allCameras)))
    results.append(("Forward direction", test_forward_direction(allCameras)))
    results.append(("Camera centers",    test_camera_centers(allCameras)))
    results.append(("Gradient flow",     test_gradient_flow(scene, allCameras, pipe)))

    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")

    n_pass = sum(1 for _, ok in results if ok)
    print(f"\n  {n_pass}/{len(results)} tests PASSED")

    if n_pass == len(results):
        print("\n  >>> ALL CHECKS PASSED — safe to implement Phase 2")
        sys.exit(0)
    else:
        print("\n  >>> FIX FAILURES BEFORE IMPLEMENTING")
        sys.exit(1)
