# ============================================================
# [CRSGaussian] Task: T2.5 — Unit test cho update_crs
# File: CRSGaussian/tests/test_update_crs.py  (TẠO MỚI)
# Mục đích: Verify update_crs() — EMA behavior, CRS range,
#           surface vs floater separation, neutral stability.
# Chạy: cd CRSGaussian && python tests/test_update_crs.py
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from utils.graphics_utils import getWorld2View2
from utils.crs.crs_module import update_crs


class MockCamera:
    """Camera giả lập với GT image."""
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


class MockGaussians:
    """GaussianModel giả lập — chỉ cần get_xyz và _crs_score."""
    def __init__(self, xyz):
        self._xyz = xyz
        # Init logit=0 → sigmoid=0.5 (neutral), giống T2.2
        self._crs_score = torch.zeros(xyz.shape[0], 1, device=xyz.device)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_crs(self):
        return torch.sigmoid(self._crs_score)


def _make_camera(uid=0, W=100, H=100, image=None):
    R = np.eye(3, dtype=np.float32)
    T = np.zeros(3, dtype=np.float32)
    FoVx = 2.0 * np.arctan(W / (2.0 * W))
    FoVy = 2.0 * np.arctan(H / (2.0 * H))
    return MockCamera(R, T, FoVx, FoVy, W, H, uid=uid, image=image)


def test_surface_high_crs():
    """Surface Gaussian (D≈1, R≈1) → CRS cao sau update."""
    H, W = 100, 100

    # 3 cameras, GT images cùng màu tại center
    imgs = []
    for _ in range(3):
        img = torch.zeros(3, H, W)
        img[0, 45:55, 45:55] = 0.8  # red
        imgs.append(img)
    cams = [_make_camera(uid=i, image=imgs[i]) for i in range(3)]

    # Gaussian ở (0, 0, 5), depth prior = 5 (khớp)
    xyz = torch.tensor([[0.0, 0.0, 5.0]], device="cuda")
    gaussians = MockGaussians(xyz)

    # Depth map: surface = 5.0 tại center
    depth_map = torch.zeros(H, W, dtype=torch.float32)
    depth_map[45:55, 45:55] = 5.0
    aligned_depth_dict = {i: depth_map.clone() for i in range(3)}

    # Chạy 5 updates để EMA kịp kéo khỏi init (logit=0).
    # 1 update với ema=0.9 chỉ dịch 10% → sigmoid(0.25)=0.56, chưa đủ.
    for _ in range(5):
        update_crs(gaussians, cams, aligned_depth_dict, depth_range=30.0)

    crs = gaussians.get_crs.item()
    assert crs > 0.7, f"Surface should have high CRS after 5 updates, got {crs:.4f}"
    print(f"  PASS: surface CRS = {crs:.4f} (after 5 updates)")


def test_floater_low_crs():
    """Floater (D≈0, R low) → CRS thấp sau update."""
    H, W = 100, 100

    # 3 cameras, GT images có màu khác nhau tại center
    img0 = torch.zeros(3, H, W); img0[0, 45:55, 45:55] = 1.0  # red
    img1 = torch.zeros(3, H, W); img1[1, 45:55, 45:55] = 1.0  # green
    img2 = torch.zeros(3, H, W); img2[2, 45:55, 45:55] = 1.0  # blue
    cams = [
        _make_camera(uid=0, image=img0),
        _make_camera(uid=1, image=img1),
        _make_camera(uid=2, image=img2),
    ]

    # Gaussian ở depth=5, nhưng depth prior=20 (xa) → D thấp
    xyz = torch.tensor([[0.0, 0.0, 5.0]], device="cuda")
    gaussians = MockGaussians(xyz)

    depth_map = torch.zeros(H, W, dtype=torch.float32)
    depth_map[45:55, 45:55] = 20.0  # surface ở 20, Gaussian ở 5
    aligned_depth_dict = {i: depth_map.clone() for i in range(3)}

    # 5 updates để EMA kịp dịch
    for _ in range(5):
        update_crs(gaussians, cams, aligned_depth_dict, depth_range=30.0)

    crs = gaussians.get_crs.item()
    assert crs < 0.5, f"Floater should have CRS < 0.5 after 5 updates, got {crs:.4f}"
    print(f"  PASS: floater CRS = {crs:.4f} (after 5 updates)")


def test_ema_converges():
    """EMA hội tụ sau nhiều lần update — _crs_score không bị stuck ở init."""
    H, W = 100, 100
    img = torch.zeros(3, H, W)
    img[:, 45:55, 45:55] = 0.5
    cams = [_make_camera(uid=i, image=img.clone()) for i in range(3)]

    xyz = torch.tensor([[0.0, 0.0, 5.0]], device="cuda")
    gaussians = MockGaussians(xyz)

    depth_map = torch.zeros(H, W, dtype=torch.float32)
    depth_map[45:55, 45:55] = 5.0
    aligned_depth_dict = {i: depth_map.clone() for i in range(3)}

    # Init: logit=0 → CRS=0.5
    crs_values = [gaussians.get_crs.item()]

    # 20 updates — ema=0.9 cần ~15 updates để ổn định
    for _ in range(20):
        update_crs(gaussians, cams, aligned_depth_dict, depth_range=30.0)
        crs_values.append(gaussians.get_crs.item())

    # CRS nên hội tụ (không còn thay đổi nhiều)
    diff_last = abs(crs_values[-1] - crs_values[-2])
    assert diff_last < 0.02, \
        f"EMA should converge, last diff={diff_last:.6f}"

    # CRS nên khác init (0.5) vì D=1, R≈1 → CRS cao
    assert abs(crs_values[-1] - 0.5) > 0.1, \
        f"CRS should move from 0.5 init, final={crs_values[-1]:.4f}"

    print(f"  PASS: EMA converges. init=0.5000, "
          f"final={crs_values[-1]:.4f}, last_diff={diff_last:.6f}")


def test_invisible_stays_neutral():
    """Gaussian behind camera → D=0.5, R=0.5 → logit=0 → CRS stays ~0.5."""
    H, W = 100, 100
    img = torch.ones(3, H, W) * 0.5
    cams = [_make_camera(uid=i, image=img.clone()) for i in range(2)]

    # Gaussian behind camera
    xyz = torch.tensor([[0.0, 0.0, -5.0]], device="cuda")
    gaussians = MockGaussians(xyz)

    depth_map = torch.ones(H, W, dtype=torch.float32) * 10.0
    aligned_depth_dict = {i: depth_map.clone() for i in range(2)}

    update_crs(gaussians, cams, aligned_depth_dict, depth_range=30.0)

    crs = gaussians.get_crs.item()
    assert abs(crs - 0.5) < 0.05, \
        f"Invisible Gaussian should stay ~0.5, got {crs:.4f}"
    print(f"  PASS: invisible CRS = {crs:.4f}")


def test_custom_weights():
    """w1/w2 khác default → CRS khác nhau khi D ≠ R."""
    H, W = 100, 100

    # Colors khác nhau giữa cameras → R thấp (~0.33)
    img0 = torch.zeros(3, H, W); img0[0, 45:55, 45:55] = 1.0
    img1 = torch.zeros(3, H, W); img1[1, 45:55, 45:55] = 1.0
    img2 = torch.zeros(3, H, W); img2[2, 45:55, 45:55] = 1.0
    cams = [
        _make_camera(uid=0, image=img0),
        _make_camera(uid=1, image=img1),
        _make_camera(uid=2, image=img2),
    ]

    # Gaussian ở depth=5, prior=5 → D≈1.0 nhưng R≈0.33
    # Khi D ≠ R, w1/w2 phải tạo ra CRS khác nhau
    xyz = torch.tensor([[0.0, 0.0, 5.0]], device="cuda")
    depth_map = torch.zeros(H, W, dtype=torch.float32)
    depth_map[45:55, 45:55] = 5.0
    aligned_depth_dict = {i: depth_map.clone() for i in range(3)}

    # w1=0.8 (D-heavy, D≈1) vs w1=0.2 (R-heavy, R≈0.33)
    g1 = MockGaussians(xyz.clone())
    g2 = MockGaussians(xyz.clone())

    for _ in range(5):
        update_crs(g1, cams, aligned_depth_dict, 30.0, w1=0.8, w2=0.2)
        update_crs(g2, cams, aligned_depth_dict, 30.0, w1=0.2, w2=0.8)

    crs1 = g1.get_crs.item()
    crs2 = g2.get_crs.item()

    # D-heavy nên cho CRS cao hơn R-heavy (vì D>R ở đây)
    assert crs1 > crs2, \
        f"D-heavy (w1=0.8) should > R-heavy (w1=0.2), got {crs1:.4f} vs {crs2:.4f}"
    print(f"  PASS: w1=0.8 (D-heavy) → CRS={crs1:.4f}, w1=0.2 (R-heavy) → CRS={crs2:.4f}")


def test_batch_mixed():
    """N=3: surface, floater, invisible — CRS tách biệt."""
    H, W = 100, 100

    img = torch.zeros(3, H, W)
    img[:, 45:55, 45:55] = 0.5  # uniform color at center
    cams = [_make_camera(uid=i, image=img.clone()) for i in range(3)]

    xyz = torch.tensor([
        [0.0, 0.0, 5.0],     # surface: depth=5, prior=5
        [0.0, 0.0, 5.0],     # floater: depth=5, prior=25 (xa)
        [0.0, 0.0, -5.0],    # invisible: behind camera
    ], device="cuda")
    gaussians = MockGaussians(xyz)

    # Depth maps: surface pixel=5, floater pixel=25
    depth_map = torch.zeros(H, W, dtype=torch.float32)
    depth_map[45:55, 45:55] = 5.0
    # Gaussian 0 và 1 project ra cùng pixel, nhưng depth prior khác
    # → cần depth map khác cho từng Gaussian? Không — cả 2 project
    # ra cùng pixel → cùng depth prior. Phải dùng depth prior khác.
    # Giải pháp: cho depth prior = 5 (surface Gaussian đúng),
    # floater sẽ bị penalize nếu xyz[1] depth ≠ 5.
    # → Sửa: floater ở depth khác
    xyz[1] = torch.tensor([0.0, 0.0, 25.0], device="cuda")  # depth=25, prior=5

    depth_map_surface = torch.zeros(H, W, dtype=torch.float32)
    depth_map_surface[45:55, 45:55] = 5.0
    aligned_depth_dict = {i: depth_map_surface.clone() for i in range(3)}

    # Multiple updates cho EMA ổn định
    for _ in range(5):
        update_crs(gaussians, cams, aligned_depth_dict, depth_range=30.0)

    crs = gaussians.get_crs  # (3, 1)

    surface_crs = crs[0].item()
    floater_crs = crs[1].item()
    invisible_crs = crs[2].item()

    # Surface > floater
    assert surface_crs > floater_crs, \
        f"Surface ({surface_crs:.4f}) should > floater ({floater_crs:.4f})"

    # Invisible ≈ 0.5
    assert abs(invisible_crs - 0.5) < 0.05, \
        f"Invisible should be ~0.5, got {invisible_crs:.4f}"

    print(f"  PASS: surface={surface_crs:.4f}, floater={floater_crs:.4f}, "
          f"invisible={invisible_crs:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("[CRSGaussian T2.5] Testing update_crs")
    print("=" * 60)

    tests = [
        ("1. Surface → high CRS", test_surface_high_crs),
        ("2. Floater → low CRS", test_floater_low_crs),
        ("3. EMA converges", test_ema_converges),
        ("4. Invisible → neutral ~0.5", test_invisible_stays_neutral),
        ("5. Custom weights w1/w2", test_custom_weights),
        ("6. Batch mixed: surface > floater > invisible", test_batch_mixed),
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
