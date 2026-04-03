# ============================================================
# [CRSGaussian] Task: T2.3 — Unit test cho compute_depth_consistency
# File: CRSGaussian/tests/test_depth_consistency.py  (TẠO MỚI)
# Mục đích: Verify D_i logic — projection, neutral default,
#           range [0,1], average across cameras.
# Chạy: cd CRSGaussian && python tests/test_depth_consistency.py
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from utils.graphics_utils import getWorld2View2, fov2focal
from utils.crs.crs_module import compute_depth_consistency


class MockCamera:
    """Camera giả lập — cùng interface với scene/cameras.py Camera."""
    def __init__(self, R, T, FoVx, FoVy, W, H, uid=0):
        self.uid = uid
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = W
        self.image_height = H
        W2C = getWorld2View2(R, T)
        self.world_view_transform = (
            torch.tensor(W2C, dtype=torch.float32)
            .transpose(0, 1)
            .cuda()
        )


def _make_identity_camera(uid=0, W=100, H=100):
    """Camera nhìn theo -Z, center tại origin."""
    R = np.eye(3, dtype=np.float32)
    T = np.zeros(3, dtype=np.float32)
    FoVx = 2.0 * np.arctan(W / (2.0 * W))   # fx = W → FoVx ≈ 53°
    FoVy = 2.0 * np.arctan(H / (2.0 * H))   # fy = H → FoVy ≈ 53°
    return MockCamera(R, T, FoVx, FoVy, W, H, uid=uid)


def test_on_surface_gaussian():
    """Gaussian nằm đúng vị trí depth prior → D_i ≈ 1.0."""
    cam = _make_identity_camera(uid=0)
    H, W = cam.image_height, cam.image_width

    # Gaussian ở (0, 0, 5) — trước camera, center frame
    xyz = torch.tensor([[0.0, 0.0, 5.0]], device="cuda")

    # Project thủ công để tìm pixel
    fx = fov2focal(cam.FoVx, W)
    fy = fov2focal(cam.FoVy, H)
    px_x = int(0.0 / 5.0 * fx + W / 2.0)
    px_y = int(0.0 / 5.0 * fy + H / 2.0)

    # Depth prior = 5.0 tại pixel đó (khớp hoàn toàn)
    depth_map = torch.zeros(H, W, dtype=torch.float32)
    depth_map[px_y, px_x] = 5.0

    aligned_depth_dict = {0: depth_map}
    depth_range = 30.0

    D = compute_depth_consistency(xyz, [cam], aligned_depth_dict, depth_range)

    assert D.shape == (1, 1), f"Expected (1,1), got {D.shape}"
    assert D.item() > 0.95, f"On-surface Gaussian should have D≈1.0, got {D.item():.4f}"
    print(f"  PASS: on-surface D_i = {D.item():.4f}")


def test_floater_gaussian():
    """Gaussian cách xa depth prior → D_i thấp."""
    cam = _make_identity_camera(uid=0)
    H, W = cam.image_height, cam.image_width

    # Gaussian ở (0, 0, 5) nhưng depth prior = 20 (xa 15 units)
    xyz = torch.tensor([[0.0, 0.0, 5.0]], device="cuda")

    fx = fov2focal(cam.FoVx, W)
    fy = fov2focal(cam.FoVy, H)
    px_x = int(0.0 / 5.0 * fx + W / 2.0)
    px_y = int(0.0 / 5.0 * fy + H / 2.0)

    depth_map = torch.zeros(H, W, dtype=torch.float32)
    depth_map[px_y, px_x] = 20.0  # surface ở 20, Gaussian ở 5

    aligned_depth_dict = {0: depth_map}
    depth_range = 30.0

    D = compute_depth_consistency(xyz, [cam], aligned_depth_dict, depth_range)

    # |5 - 20| / 30 = 0.5 → D = 0.5
    expected = 1.0 - abs(5.0 - 20.0) / 30.0
    assert abs(D.item() - expected) < 0.01, \
        f"Floater D_i expected {expected:.4f}, got {D.item():.4f}"
    print(f"  PASS: floater D_i = {D.item():.4f} (expected {expected:.4f})")


def test_behind_camera_neutral():
    """Gaussian phía sau camera → D_i = 0.5 (neutral)."""
    cam = _make_identity_camera(uid=0)
    H, W = cam.image_height, cam.image_width

    # Z = -5 → phía sau camera (depth < 0)
    xyz = torch.tensor([[0.0, 0.0, -5.0]], device="cuda")

    depth_map = torch.ones(H, W, dtype=torch.float32) * 10.0
    aligned_depth_dict = {0: depth_map}
    depth_range = 30.0

    D = compute_depth_consistency(xyz, [cam], aligned_depth_dict, depth_range)

    assert abs(D.item() - 0.5) < 1e-6, \
        f"Behind-camera Gaussian should be 0.5, got {D.item():.4f}"
    print(f"  PASS: behind-camera D_i = {D.item():.4f}")


def test_outside_frame_neutral():
    """Gaussian ngoài frame → D_i = 0.5 (neutral)."""
    cam = _make_identity_camera(uid=0)
    H, W = cam.image_height, cam.image_width

    # X = 1000 → project ra ngoài image bounds
    xyz = torch.tensor([[1000.0, 0.0, 1.0]], device="cuda")

    depth_map = torch.ones(H, W, dtype=torch.float32) * 10.0
    aligned_depth_dict = {0: depth_map}
    depth_range = 30.0

    D = compute_depth_consistency(xyz, [cam], aligned_depth_dict, depth_range)

    assert abs(D.item() - 0.5) < 1e-6, \
        f"Outside-frame Gaussian should be 0.5, got {D.item():.4f}"
    print(f"  PASS: outside-frame D_i = {D.item():.4f}")


def test_multi_gaussian_batch():
    """N=100 Gaussians — mix surface, floater, invisible."""
    cam = _make_identity_camera(uid=0)
    H, W = cam.image_height, cam.image_width

    N = 100
    xyz = torch.zeros(N, 3, device="cuda")

    # 50 Gaussians trước camera (visible, near center)
    xyz[:50, 2] = 5.0
    xyz[:50, 0] = torch.linspace(-0.5, 0.5, 50, device="cuda")

    # 30 Gaussians phía sau camera
    xyz[50:80, 2] = -5.0

    # 20 Gaussians ngoài frame
    xyz[80:, 0] = 1000.0
    xyz[80:, 2] = 1.0

    depth_map = torch.ones(H, W, dtype=torch.float32) * 5.0  # surface = 5
    aligned_depth_dict = {0: depth_map}
    depth_range = 30.0

    D = compute_depth_consistency(xyz, [cam], aligned_depth_dict, depth_range)

    assert D.shape == (N, 1), f"Expected ({N},1), got {D.shape}"

    # Visible Gaussians (0-49) at depth=5, prior=5 → D ≈ 1.0
    visible_D = D[:50]
    assert (visible_D > 0.9).all(), \
        f"Visible on-surface should be >0.9, min={visible_D.min():.4f}"

    # Behind camera (50-79) → neutral 0.5
    behind_D = D[50:80]
    assert (torch.abs(behind_D - 0.5) < 1e-6).all(), \
        f"Behind-camera should be 0.5, got range [{behind_D.min():.4f}, {behind_D.max():.4f}]"

    # Outside frame (80-99) → neutral 0.5
    outside_D = D[80:]
    assert (torch.abs(outside_D - 0.5) < 1e-6).all(), \
        f"Outside-frame should be 0.5, got range [{outside_D.min():.4f}, {outside_D.max():.4f}]"

    print(f"  PASS: N=100 batch — visible mean={visible_D.mean():.4f}, "
          f"behind={behind_D[0].item():.4f}, outside={outside_D[0].item():.4f}")


def test_multi_camera_average():
    """D_i trung bình trên 2 cameras."""
    cam0 = _make_identity_camera(uid=0)
    cam1 = _make_identity_camera(uid=1)
    H, W = cam0.image_height, cam0.image_width

    # Gaussian ở (0, 0, 5)
    xyz = torch.tensor([[0.0, 0.0, 5.0]], device="cuda")

    # Camera 0: depth prior = 5.0 (khớp) → D = 1.0
    depth_map0 = torch.zeros(H, W, dtype=torch.float32)
    depth_map0[H // 2, W // 2] = 5.0

    # Camera 1: depth prior = 20.0 (lệch) → D = 1 - 15/30 = 0.5
    depth_map1 = torch.zeros(H, W, dtype=torch.float32)
    depth_map1[H // 2, W // 2] = 20.0

    aligned_depth_dict = {0: depth_map0, 1: depth_map1}
    depth_range = 30.0

    D = compute_depth_consistency(
        xyz, [cam0, cam1], aligned_depth_dict, depth_range)

    # Average = (1.0 + 0.5) / 2 = 0.75
    expected = 0.75
    assert abs(D.item() - expected) < 0.05, \
        f"Multi-camera average expected ~{expected}, got {D.item():.4f}"
    print(f"  PASS: multi-camera average D_i = {D.item():.4f} (expected ~{expected})")


def test_output_range_clamped():
    """Error > depth_range → D_i clamp tại 0.0, không âm."""
    cam = _make_identity_camera(uid=0)
    H, W = cam.image_height, cam.image_width

    # Gaussian ở depth=5, prior=100, depth_range=10
    # |5-100|/10 = 9.5 → 1-9.5 = -8.5 → clamp → 0.0
    xyz = torch.tensor([[0.0, 0.0, 5.0]], device="cuda")

    fx = fov2focal(cam.FoVx, W)
    px_x = int(W / 2.0)
    px_y = int(H / 2.0)

    depth_map = torch.zeros(H, W, dtype=torch.float32)
    depth_map[px_y, px_x] = 100.0

    aligned_depth_dict = {0: depth_map}
    depth_range = 10.0

    D = compute_depth_consistency(xyz, [cam], aligned_depth_dict, depth_range)

    assert D.item() >= 0.0, f"D_i should be >= 0, got {D.item():.4f}"
    assert D.item() < 0.01, f"Extreme floater D_i should be ~0.0, got {D.item():.4f}"
    print(f"  PASS: extreme floater D_i = {D.item():.4f} (clamped to 0)")


def test_no_cameras_all_neutral():
    """Không có camera nào trong aligned_depth_dict → tất cả neutral."""
    cam = _make_identity_camera(uid=99)  # uid=99 không có trong dict

    xyz = torch.tensor([[0.0, 0.0, 5.0]], device="cuda")
    aligned_depth_dict = {0: torch.ones(100, 100)}  # uid=0, không phải 99
    depth_range = 30.0

    D = compute_depth_consistency(xyz, [cam], aligned_depth_dict, depth_range)

    assert abs(D.item() - 0.5) < 1e-6, \
        f"No matching camera → neutral 0.5, got {D.item():.4f}"
    print(f"  PASS: no matching camera → D_i = {D.item():.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("[CRSGaussian T2.3] Testing compute_depth_consistency")
    print("=" * 60)

    tests = [
        ("1. On-surface Gaussian → D≈1.0", test_on_surface_gaussian),
        ("2. Floater Gaussian → D low", test_floater_gaussian),
        ("3. Behind camera → neutral 0.5", test_behind_camera_neutral),
        ("4. Outside frame → neutral 0.5", test_outside_frame_neutral),
        ("5. Batch N=100 mixed", test_multi_gaussian_batch),
        ("6. Multi-camera average", test_multi_camera_average),
        ("7. Extreme error → clamp 0.0", test_output_range_clamped),
        ("8. No matching cameras → neutral", test_no_cameras_all_neutral),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            print(f"\n{name}")
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
