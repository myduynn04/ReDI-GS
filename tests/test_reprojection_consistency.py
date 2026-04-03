# ============================================================
# [CRSGaussian] Task: T2.4 — Unit test cho compute_reprojection_consistency
# File: CRSGaussian/tests/test_reprojection_consistency.py  (TẠO MỚI)
# Mục đích: Verify R_i logic — GT color pairwise, neutral default,
#           range [0,1], floater detection.
# Chạy: cd CRSGaussian && python tests/test_reprojection_consistency.py
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from utils.graphics_utils import getWorld2View2, fov2focal
from utils.crs.crs_module import compute_reprojection_consistency


class MockCamera:
    """Camera giả lập — cùng interface với scene/cameras.py Camera.
    Thêm original_image để test R_i (GT color lookup).
    """
    def __init__(self, R, T, FoVx, FoVy, W, H, uid=0, image=None):
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
        # GT image: (3, H, W) float [0,1] on cuda
        if image is not None:
            self.original_image = image.cuda()
        else:
            self.original_image = torch.zeros(3, H, W, device="cuda")


def _make_camera(uid=0, W=100, H=100, image=None):
    """Camera nhìn theo -Z, center tại origin."""
    R = np.eye(3, dtype=np.float32)
    T = np.zeros(3, dtype=np.float32)
    FoVx = 2.0 * np.arctan(W / (2.0 * W))
    FoVy = 2.0 * np.arctan(H / (2.0 * H))
    return MockCamera(R, T, FoVx, FoVy, W, H, uid=uid, image=image)


def test_surface_gaussian_consistent_color():
    """Gaussian trên surface → cùng GT color từ mọi camera → R ≈ 1.0."""
    H, W = 100, 100

    # 3 cameras cùng pose (đơn giản hóa) — GT images cùng màu đỏ tại center
    images = []
    for _ in range(3):
        img = torch.zeros(3, H, W)
        # Vùng center pixel: màu đỏ (1, 0, 0)
        img[0, 45:55, 45:55] = 1.0
        images.append(img)

    cams = [_make_camera(uid=i, image=images[i]) for i in range(3)]

    # Gaussian ở (0, 0, 5) — project ra center (50, 50) trên cả 3 cameras
    xyz = torch.tensor([[0.0, 0.0, 5.0]], device="cuda")

    R = compute_reprojection_consistency(xyz, cams)

    assert R.shape == (1, 1), f"Expected (1,1), got {R.shape}"
    assert R.item() > 0.95, \
        f"Consistent color should give R≈1.0, got {R.item():.4f}"
    print(f"  PASS: surface R_i = {R.item():.4f}")


def test_floater_gaussian_inconsistent_color():
    """Floater → GT colors khác nhau giữa cameras → R thấp."""
    H, W = 100, 100

    # Camera 0: pixel center = đỏ
    img0 = torch.zeros(3, H, W)
    img0[0, 45:55, 45:55] = 1.0

    # Camera 1: pixel center = xanh lá
    img1 = torch.zeros(3, H, W)
    img1[1, 45:55, 45:55] = 1.0

    # Camera 2: pixel center = xanh dương
    img2 = torch.zeros(3, H, W)
    img2[2, 45:55, 45:55] = 1.0

    cams = [
        _make_camera(uid=0, image=img0),
        _make_camera(uid=1, image=img1),
        _make_camera(uid=2, image=img2),
    ]

    # Gaussian project ra center trên cả 3 cameras → nhưng colors khác nhau
    xyz = torch.tensor([[0.0, 0.0, 5.0]], device="cuda")

    R = compute_reprojection_consistency(xyz, cams)

    # Pair (0,1): |red - green| = mean(|1-0|, |0-1|, |0-0|) = 2/3
    # Pair (0,2): |red - blue|  = mean(|1-0|, |0-0|, |0-1|) = 2/3
    # Pair (1,2): |green - blue| = mean(|0-0|, |1-0|, |0-1|) = 2/3
    # mean = 2/3, R = 1 - 2/3 ≈ 0.333
    assert R.item() < 0.5, \
        f"Inconsistent color should give R<0.5, got {R.item():.4f}"
    print(f"  PASS: floater R_i = {R.item():.4f}")


def test_visible_one_camera_neutral():
    """Gaussian visible từ chỉ 1 camera → R = 0.5 (neutral)."""
    H, W = 100, 100
    img = torch.ones(3, H, W) * 0.5

    cam0 = _make_camera(uid=0, image=img)
    cam1 = _make_camera(uid=1, image=img)

    # Gaussian ở (0, 0, 5) — visible từ cả 2
    # Gaussian ở (0, 0, -5) — behind cả 2 cameras → invisible
    # Gaussian ở (1000, 0, 1) — outside frame
    xyz = torch.tensor([
        [0.0, 0.0, 5.0],    # visible from both
        [0.0, 0.0, -5.0],   # behind both
        [1000.0, 0.0, 1.0], # outside frame
    ], device="cuda")

    R = compute_reprojection_consistency(xyz, [cam0, cam1])

    # Behind camera → neutral
    assert abs(R[1].item() - 0.5) < 1e-6, \
        f"Behind-camera should be 0.5, got {R[1].item():.4f}"

    # Outside frame → neutral
    assert abs(R[2].item() - 0.5) < 1e-6, \
        f"Outside-frame should be 0.5, got {R[2].item():.4f}"

    print(f"  PASS: behind={R[1].item():.4f}, outside={R[2].item():.4f}")


def test_single_camera_all_neutral():
    """Chỉ 1 camera → không có pair → tất cả neutral 0.5."""
    H, W = 100, 100
    img = torch.ones(3, H, W) * 0.5
    cam = _make_camera(uid=0, image=img)

    xyz = torch.tensor([[0.0, 0.0, 5.0]], device="cuda")

    R = compute_reprojection_consistency(xyz, [cam])

    assert abs(R.item() - 0.5) < 1e-6, \
        f"Single camera should be 0.5, got {R.item():.4f}"
    print(f"  PASS: single camera → R = {R.item():.4f}")


def test_output_shape_and_range():
    """Output shape (N, 1), range [0, 1]."""
    H, W = 100, 100
    N = 50
    img0 = torch.rand(3, H, W)
    img1 = torch.rand(3, H, W)

    cam0 = _make_camera(uid=0, image=img0)
    cam1 = _make_camera(uid=1, image=img1)

    xyz = torch.randn(N, 3, device="cuda")
    xyz[:, 2] = xyz[:, 2].abs() + 1.0  # ensure depth > 0

    R = compute_reprojection_consistency(xyz, [cam0, cam1])

    assert R.shape == (N, 1), f"Expected ({N},1), got {R.shape}"
    assert (R >= 0.0).all() and (R <= 1.0).all(), \
        f"R out of [0,1]: min={R.min():.4f}, max={R.max():.4f}"
    print(f"  PASS: shape={R.shape}, range=[{R.min():.4f}, {R.max():.4f}]")


def test_no_original_image_skipped():
    """Camera không có original_image (PseudoCamera) → bị skip."""
    H, W = 100, 100
    img = torch.ones(3, H, W) * 0.5

    cam_real = _make_camera(uid=0, image=img)

    # Mock PseudoCamera — không có original_image
    class FakePseudo:
        pass
    cam_pseudo = FakePseudo()

    xyz = torch.tensor([[0.0, 0.0, 5.0]], device="cuda")

    # 1 real + 1 pseudo = chỉ 1 valid camera → neutral
    R = compute_reprojection_consistency(xyz, [cam_real, cam_pseudo])

    assert abs(R.item() - 0.5) < 1e-6, \
        f"1 real + 1 pseudo should be neutral, got {R.item():.4f}"
    print(f"  PASS: PseudoCamera skipped → R = {R.item():.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("[CRSGaussian T2.4] Testing compute_reprojection_consistency")
    print("=" * 60)

    tests = [
        ("1. Surface Gaussian → R≈1.0 (consistent color)", test_surface_gaussian_consistent_color),
        ("2. Floater Gaussian → R low (inconsistent color)", test_floater_gaussian_inconsistent_color),
        ("3. Visible <2 cameras → neutral 0.5", test_visible_one_camera_neutral),
        ("4. Single camera → all neutral", test_single_camera_all_neutral),
        ("5. Output shape (N,1) and range [0,1]", test_output_shape_and_range),
        ("6. PseudoCamera skipped", test_no_original_image_skipped),
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
