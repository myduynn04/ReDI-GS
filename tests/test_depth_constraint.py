# ============================================================
# [CRSGaussian] Task: T4.1 — Unit test cho _depth_constraint_mask
# File: CRSGaussian/tests/test_depth_constraint.py  (TẠO MỚI)
# Chạy: cd CRSGaussian && python tests/test_depth_constraint.py
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from utils.graphics_utils import getWorld2View2
from scene.gaussian_model import _depth_constraint_mask


class MockCamera:
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


def _make_camera(uid=0, W=100, H=100):
    R = np.eye(3, dtype=np.float32)
    T = np.zeros(3, dtype=np.float32)
    FoVx = 2.0 * np.arctan(W / (2.0 * W))
    FoVy = 2.0 * np.arctan(H / (2.0 * H))
    return MockCamera(R, T, FoVx, FoVy, W, H, uid=uid)


def test_near_surface_accepted():
    """Gaussian gần depth prior → ACCEPT."""
    cam = _make_camera(uid=0)
    H, W = cam.image_height, cam.image_width

    # Gaussian ở depth=5.0, prior=5.0 → diff=0 < epsilon
    new_xyz = torch.tensor([[0.0, 0.0, 5.0]], device="cuda")

    depth_map = torch.zeros(H, W, device="cuda")
    depth_map[45:55, 45:55] = 5.0
    aligned_depth_dict = {0: depth_map}

    epsilon = 1.5  # 0.05 * 30 = 1.5
    keep = _depth_constraint_mask(new_xyz, [cam], aligned_depth_dict, epsilon)

    assert keep[0].item() == True, "Near-surface should be accepted"
    print(f"  PASS: near surface → accepted")


def test_far_from_surface_rejected():
    """Gaussian xa depth prior → REJECT."""
    cam = _make_camera(uid=0)
    H, W = cam.image_height, cam.image_width

    # Gaussian ở depth=5.0, prior=20.0 → diff=15 > epsilon=1.5
    new_xyz = torch.tensor([[0.0, 0.0, 5.0]], device="cuda")

    depth_map = torch.zeros(H, W, device="cuda")
    depth_map[45:55, 45:55] = 20.0
    aligned_depth_dict = {0: depth_map}

    epsilon = 1.5
    keep = _depth_constraint_mask(new_xyz, [cam], aligned_depth_dict, epsilon)

    assert keep[0].item() == False, "Far from surface should be rejected"
    print(f"  PASS: far from surface → rejected")


def test_no_depth_dict_all_rejected():
    """Không có depth dict → tất cả rejected (invisible từ mọi camera)."""
    cam = _make_camera(uid=99)  # uid=99 không có trong dict
    new_xyz = torch.tensor([[0.0, 0.0, 5.0]], device="cuda")
    aligned_depth_dict = {0: torch.ones(100, 100, device="cuda")}

    keep = _depth_constraint_mask(new_xyz, [cam], aligned_depth_dict, 1.5)

    assert keep[0].item() == False, "No matching camera → rejected"
    print(f"  PASS: no matching camera → rejected")


def test_batch_mixed():
    """Batch: 1 near + 1 far → [True, False]."""
    cam = _make_camera(uid=0)
    H, W = cam.image_height, cam.image_width

    new_xyz = torch.tensor([
        [0.0, 0.0, 5.0],   # near prior=5
        [0.0, 0.0, 25.0],  # far from prior=5
    ], device="cuda")

    depth_map = torch.zeros(H, W, device="cuda")
    depth_map[45:55, 45:55] = 5.0
    aligned_depth_dict = {0: depth_map}

    keep = _depth_constraint_mask(new_xyz, [cam], aligned_depth_dict, 1.5)

    assert keep[0].item() == True, "Near should be accepted"
    assert keep[1].item() == False, "Far should be rejected"
    print(f"  PASS: batch [near=True, far=False]")


def test_all_tensors_same_shape():
    """Sau filter, tất cả tensors phải có cùng shape[0]."""
    cam = _make_camera(uid=0)
    H, W = cam.image_height, cam.image_width

    N = 20
    new_xyz = torch.randn(N, 3, device="cuda")
    new_xyz[:, 2] = 5.0  # all at depth 5

    depth_map = torch.zeros(H, W, device="cuda")
    depth_map[45:55, 45:55] = 5.0
    aligned_depth_dict = {0: depth_map}

    keep = _depth_constraint_mask(new_xyz, [cam], aligned_depth_dict, 1.5)

    # Simulate filtering 6 tensors
    new_scaling = torch.randn(N, 3, device="cuda")
    new_rotation = torch.randn(N, 4, device="cuda")
    new_features_dc = torch.randn(N, 1, 3, device="cuda")
    new_features_rest = torch.randn(N, 15, 3, device="cuda")
    new_opacity = torch.randn(N, 1, device="cuda")

    filtered = [
        new_xyz[keep], new_scaling[keep], new_rotation[keep],
        new_features_dc[keep], new_features_rest[keep], new_opacity[keep]
    ]

    shapes = [t.shape[0] for t in filtered]
    assert len(set(shapes)) == 1, f"Shape mismatch: {shapes}"
    print(f"  PASS: all tensors shape[0]={shapes[0]} (from N={N}, kept={keep.sum().item()})")


if __name__ == "__main__":
    print("=" * 60)
    print("[CRSGaussian T4.1] Testing _depth_constraint_mask")
    print("=" * 60)

    tests = [
        ("1. Near surface → ACCEPT", test_near_surface_accepted),
        ("2. Far from surface → REJECT", test_far_from_surface_rejected),
        ("3. No matching camera → REJECT", test_no_depth_dict_all_rejected),
        ("4. Batch mixed [True, False]", test_batch_mixed),
        ("5. All tensors same shape after filter", test_all_tensors_same_shape),
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
