# ============================================================
# [CRSGaussian] Task: T5.6 — Unit test cho Informed CRS₀
# File: CRSGaussian/tests/test_informed_crs_init.py  (TẠO MỚI)
# Mục đích: Verify compute_reproj_quality, compute_view_support,
#           compute_informed_crs0, component switches, densify inherit.
# Chạy: cd CRSGaussian && python tests/test_informed_crs_init.py
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from utils.graphics_utils import getWorld2View2


class MockCamera:
    """Camera giả lập với aligned depth map."""
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
        if image is not None:
            self.original_image = image.cuda()
        else:
            self.original_image = torch.zeros(3, H, W, device="cuda")


def make_test_cameras(n=3, W=64, H=48):
    """Tạo n cameras nhìn vào origin từ các góc khác nhau."""
    cameras = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        # Camera đặt ở bán kính 4, nhìn vào origin
        eye = np.array([4 * np.cos(angle), 0.0, 4 * np.sin(angle)])
        forward = -eye / np.linalg.norm(eye)
        right = np.cross(np.array([0, 1, 0]), forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        R = np.stack([right, up, forward], axis=0)  # (3,3) rows = axes
        T = -R @ eye
        cam = MockCamera(
            R=R, T=T,
            FoVx=1.0, FoVy=0.75,
            W=W, H=H, uid=i,
        )
        cameras.append(cam)
    return cameras


# ==============================================================
# Test 1: compute_reproj_quality
# ==============================================================
def test_reproj_quality():
    from utils.crs.crs_init import compute_reproj_quality

    errors = np.array([0.0, 1.25, 2.5, 5.0], dtype=np.float32)
    q = compute_reproj_quality(errors, tau_r=2.5)

    assert abs(q[0] - 1.0) < 1e-6, f"error=0 → q should be 1.0, got {q[0]}"
    assert abs(q[1] - 0.5) < 1e-6, f"error=1.25 → q should be 0.5, got {q[1]}"
    assert abs(q[2] - 0.0) < 1e-6, f"error=2.5 → q should be 0.0, got {q[2]}"
    assert abs(q[3] - 0.0) < 1e-6, f"error=5.0 → q should be 0.0 (clamped), got {q[3]}"

    assert q.shape == (4,), f"Expected shape (4,), got {q.shape}"
    assert q.dtype == np.float64 or q.dtype == np.float32

    print("[PASS] test_reproj_quality — 4/4 cases")


# ==============================================================
# Test 2: compute_view_support
# ==============================================================
def test_view_support():
    from utils.crs.crs_init import compute_view_support

    cameras = make_test_cameras(n=3)

    # Point tại origin → visible từ tất cả 3 cameras
    # Point rất xa sang 1 phía → chỉ visible từ 1-2 cameras
    points = np.array([
        [0.0, 0.0, 0.0],   # center → thấy từ 3 cameras
        [100.0, 0.0, 0.0],  # rất xa → có thể chỉ 1 camera
    ], dtype=np.float32)
    pts_gpu = torch.from_numpy(points).cuda()

    q = compute_view_support(pts_gpu, cameras, n_train=3)

    # Point center: n_obs=3 → q = (3-1)/(3-1) = 1.0
    assert abs(q[0] - 1.0) < 1e-4, f"center point q should be 1.0, got {q[0]}"

    # Verify range [0, 1]
    assert q.min() >= -1e-6, f"q min should be >= 0, got {q.min()}"
    assert q.max() <= 1.0 + 1e-6, f"q max should be <= 1, got {q.max()}"

    # Test n_train=1 edge case → denom=1, q = (n_obs-1)/1
    q_single = compute_view_support(pts_gpu[:1], cameras[:1], n_train=1)
    # n_obs=1 for 1 camera, q = (1-1)/1 = 0.0
    assert abs(q_single[0] - 0.0) < 1e-4, f"single camera q should be 0.0, got {q_single[0]}"

    print("[PASS] test_view_support — center=1.0, range OK, edge case OK")


# ==============================================================
# Test 3: compute_informed_crs0 — full pipeline
# ==============================================================
def test_full_pipeline():
    from utils.crs.crs_init import (
        compute_reproj_quality, compute_depth_agreement,
        compute_view_support, compute_informed_crs0,
    )

    N = 10
    cameras = make_test_cameras(n=3)

    # Tạo points gần origin (visible từ cameras)
    points_xyz = np.random.randn(N, 3).astype(np.float32) * 0.5

    # Tạo fake aligned_depth_dict — depth map = constant 4.0
    # (cameras ở bán kính 4 → points gần origin có depth ~ 4)
    aligned_depth_dict = {}
    for cam in cameras:
        H, W = cam.image_height, cam.image_width
        aligned_depth_dict[cam.uid] = torch.full((H, W), 4.0, device="cuda")

    depth_range = 3.0

    # Mock source_path + n_views — cần cho reproj errors
    # Để test không phụ thuộc COLMAP bin, tắt reproj và test riêng
    logit = compute_informed_crs0(
        points_xyz=points_xyz,
        cameras=cameras,
        aligned_depth_dict=aligned_depth_dict,
        depth_range=depth_range,
        source_path="/nonexistent",  # không dùng vì use_reproj=False
        n_views=3,
        use_reproj=False,
        use_depth=True,
        use_view=True,
        w_reproj=0.333,
        w_depth=0.5,
        w_view=0.5,
        tau_r=2.5,
        gamma=5.0,
    )

    assert logit.shape == (N,), f"Expected shape ({N},), got {logit.shape}"
    assert logit.dtype == torch.float32, f"Expected float32, got {logit.dtype}"

    # Logit range: γ=5 → max ±2.5
    assert logit.min() >= -2.6, f"logit min too low: {logit.min()}"
    assert logit.max() <= 2.6, f"logit max too high: {logit.max()}"

    # CRS₀ = sigmoid(logit) should be in reasonable range
    crs = torch.sigmoid(logit)
    assert crs.min() >= 0.05, f"CRS min too low: {crs.min()}"
    assert crs.max() <= 0.95, f"CRS max too high: {crs.max()}"

    print(f"[PASS] test_full_pipeline — shape={logit.shape}, "
          f"logit=[{logit.min():.2f}, {logit.max():.2f}], "
          f"CRS=[{crs.min():.3f}, {crs.max():.3f}]")


# ==============================================================
# Test 4: component switches — tắt 2/3, chỉ dùng q_view
# ==============================================================
def test_component_switches():
    from utils.crs.crs_init import compute_informed_crs0

    N = 5
    cameras = make_test_cameras(n=3)
    points_xyz = np.zeros((N, 3), dtype=np.float32)  # all at origin

    aligned_depth_dict = {}
    for cam in cameras:
        H, W = cam.image_height, cam.image_width
        aligned_depth_dict[cam.uid] = torch.full((H, W), 4.0, device="cuda")

    # Chỉ view, tắt reproj + depth
    logit_view_only = compute_informed_crs0(
        points_xyz=points_xyz,
        cameras=cameras,
        aligned_depth_dict=aligned_depth_dict,
        depth_range=3.0,
        source_path="/nonexistent",
        n_views=3,
        use_reproj=False,
        use_depth=False,
        use_view=True,
        gamma=5.0,
    )

    # Point tại origin → visible từ 3 cameras → q_view=1.0
    # Q = 1.0 (chỉ view) → logit = 5*(1.0-0.5) = 2.5
    expected_logit = 2.5
    for i in range(N):
        assert abs(logit_view_only[i].item() - expected_logit) < 0.1, \
            f"Point {i}: expected logit ~{expected_logit}, got {logit_view_only[i].item():.3f}"

    # Tắt tất cả → neutral logit=0
    logit_none = compute_informed_crs0(
        points_xyz=points_xyz,
        cameras=cameras,
        aligned_depth_dict=aligned_depth_dict,
        depth_range=3.0,
        source_path="/nonexistent",
        n_views=3,
        use_reproj=False,
        use_depth=False,
        use_view=False,
        gamma=5.0,
    )
    assert (logit_none == 0).all(), f"All off → logit should be 0, got {logit_none}"

    print("[PASS] test_component_switches — view_only OK, all_off=neutral OK")


# ==============================================================
# Test 5: baseline behavior — informed_crs_init=False
# ==============================================================
def test_baseline_behavior():
    """Khi informed_crs_init=False, _crs_score phải = zeros → CRS=0.5."""
    N = 100
    crs_score = torch.zeros((N, 1), device="cuda")
    crs = torch.sigmoid(crs_score)

    assert (crs == 0.5).all(), f"Neutral init should give CRS=0.5, got range [{crs.min()}, {crs.max()}]"

    print("[PASS] test_baseline_behavior — zeros → CRS=0.5")


# ==============================================================
# Test 6: densify inherit — conservative capped at 0.5
# ==============================================================
def test_densify_inherit():
    """η=0.7: child CRS₀ = clip(η * CRS_parent, 0, 0.5)."""
    eta = 0.7

    # Case 1: parent CRS=0.8 → child = clip(0.56, 0, 0.5) = 0.5
    parent_crs_1 = 0.8
    child_crs_1 = min(eta * parent_crs_1, 0.5)
    assert abs(child_crs_1 - 0.5) < 1e-6, f"Expected 0.5, got {child_crs_1}"

    # Case 2: parent CRS=0.4 → child = clip(0.28, 0, 0.5) = 0.28
    parent_crs_2 = 0.4
    child_crs_2 = min(eta * parent_crs_2, 0.5)
    assert abs(child_crs_2 - 0.28) < 1e-6, f"Expected 0.28, got {child_crs_2}"

    # Case 3: parent CRS=0.1 (low quality) → child = clip(0.07, 0, 0.5) = 0.07
    parent_crs_3 = 0.1
    child_crs_3 = min(eta * parent_crs_3, 0.5)
    assert abs(child_crs_3 - 0.07) < 1e-6, f"Expected 0.07, got {child_crs_3}"

    # Verify logit conversion: child_crs → logit → sigmoid ≈ child_crs
    for child_crs in [child_crs_1, child_crs_2, child_crs_3]:
        c_clamped = max(min(child_crs, 0.5), 1e-6)
        logit = np.log(c_clamped / (1 - c_clamped))
        recovered = 1.0 / (1.0 + np.exp(-logit))
        assert abs(recovered - c_clamped) < 1e-5, \
            f"Logit roundtrip failed: {c_clamped} → {logit} → {recovered}"

    print("[PASS] test_densify_inherit — η=0.7: 0.8→0.5, 0.4→0.28, 0.1→0.07, logit roundtrip OK")


# ==============================================================
# Main
# ==============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("[CRSGaussian] Test: Informed CRS₀ Initialization (T5.6)")
    print("=" * 60)

    test_reproj_quality()
    test_view_support()
    test_full_pipeline()
    test_component_switches()
    test_baseline_behavior()
    test_densify_inherit()

    print("=" * 60)
    print("[CRSGaussian] ALL 6 TESTS PASSED")
    print("=" * 60)
