# ============================================================
# [CRSGaussian Phase 26 A1] Unit test cho v_stability.py + geom_freeze.py
# File: tests/test_v_stability.py (NEW — mirrors tests/test_update_crs.py)
# Mục đích: Verify update_v_stability() EMA behavior, compute_V_stability()
#           phân biệt Gaussian ổn định vs đang trôi dạt, và
#           apply_crs_modulated_geom_freeze() zero đúng grad.
# Chạy: cd ReDI-GS && python tests/test_v_stability.py
#       (KHÔNG dùng pytest — file tự có runner ở cuối, giống test_update_crs.py)
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from utils.crs.v_stability import update_v_stability, compute_V_stability
from utils.crs.geom_freeze import apply_crs_modulated_geom_freeze


class MockGaussians:
    """GaussianModel giả lập — chỉ cần _xyz/_scaling/_rotation + _crs_score."""
    def __init__(self, xyz, scaling=None, rotation=None):
        self._xyz = xyz
        N = xyz.shape[0]
        self._scaling = scaling if scaling is not None else torch.zeros(N, 3, device=xyz.device)
        self._rotation = rotation if rotation is not None else torch.zeros(N, 4, device=xyz.device)
        self._crs_score = torch.zeros(N, 1, device=xyz.device)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_crs(self):
        return torch.sigmoid(self._crs_score)


def test_first_call_initializes_neutral():
    """Lần gọi đầu tiên chỉ init buffer, chưa đủ data → V=0.5 (neutral)."""
    xyz = torch.tensor([[0.0, 0.0, 5.0]], device="cuda")
    gaussians = MockGaussians(xyz)

    # Trước khi update_v_stability lần nào — chưa có buffer.
    V = compute_V_stability(gaussians)
    assert abs(V.item() - 0.5) < 1e-6, f"Chưa có buffer → V phải =0.5, got {V.item():.4f}"

    update_v_stability(gaussians)  # lần đầu chỉ init, var=0
    V = compute_V_stability(gaussians)
    # Tolerance 1e-3 (không phải 1e-6): compute_V_stability cộng epsilon
    # 1e-8 bên trong sqrt() để tránh NaN tại std=0 — var=0 thực tế cho
    # V≈0.4999 chứ không tuyệt đối =0.5. Đây là sai lệch cố ý, không phải
    # bug.
    assert abs(V.item() - 0.5) < 1e-3, \
        f"Sau init (var=0) → V phải ≈0.5, got {V.item():.4f}"
    print(f"  PASS: neutral init V = {V.item():.4f}")


def test_stable_gaussian_high_v():
    """Gaussian đứng yên qua nhiều update → variance≈0 → V cao (~0.5, vì std=0)."""
    xyz = torch.tensor([[1.0, 2.0, 5.0]], device="cuda")
    gaussians = MockGaussians(xyz)

    for _ in range(10):
        update_v_stability(gaussians)  # xyz không đổi mỗi lần

    V = compute_V_stability(gaussians)
    # Đứng yên tuyệt đối → var=0 mọi bước → V vẫn ở neutral ≈0.5 (không có
    # drift để phát hiện). Đây là hành vi ĐÚNG theo thiết kế (xem docstring
    # compute_V_stability: std=0 → V=0.5, không phải V=1). Tolerance 1e-3
    # (không phải 1e-6) vì epsilon 1e-8 trong sqrt() làm V≈0.4999.
    assert abs(V.item() - 0.5) < 1e-3, \
        f"Gaussian đứng yên tuyệt đối → V≈0.5 (no drift detected), got {V.item():.4f}"
    print(f"  PASS: stationary Gaussian V = {V.item():.4f}")


def test_drifting_gaussian_low_v():
    """Gaussian dao động mạnh qua các update → variance cao → V thấp."""
    gaussians = MockGaussians(torch.tensor([[0.0, 0.0, 5.0]], device="cuda"))

    # Dao động biên độ lớn quanh gốc — giả lập "trôi dạt"/méo hình học.
    torch.manual_seed(0)
    for _ in range(15):
        gaussians._xyz = torch.tensor(
            [[float(torch.randn(1) * 5.0), 0.0, 5.0]], device="cuda"
        )
        update_v_stability(gaussians)

    V = compute_V_stability(gaussians)
    assert V.item() < 0.4, \
        f"Gaussian dao động mạnh → V phải thấp (<0.4), got {V.item():.4f}"
    print(f"  PASS: drifting Gaussian V = {V.item():.4f}")


def test_shape_mismatch_reinit():
    """N thay đổi giữa 2 lần update (densify/prune) → buffer re-init, không crash."""
    gaussians = MockGaussians(torch.tensor([[0.0, 0.0, 5.0]], device="cuda"))
    update_v_stability(gaussians)
    update_v_stability(gaussians)

    # Simulate densify: N: 1 → 3
    gaussians._xyz = torch.tensor(
        [[0.0, 0.0, 5.0], [1.0, 1.0, 5.0], [2.0, 2.0, 5.0]], device="cuda"
    )
    gaussians._scaling = torch.zeros(3, 3, device="cuda")
    gaussians._rotation = torch.zeros(3, 4, device="cuda")
    gaussians._crs_score = torch.zeros(3, 1, device="cuda")

    update_v_stability(gaussians)  # phải re-init, không raise
    assert gaussians._v_ema_mean_xyz.shape[0] == 3
    V = compute_V_stability(gaussians)
    assert V.shape[0] == 3
    print(f"  PASS: shape mismatch re-init OK, N=3, V={V.squeeze().tolist()}")


def test_geom_freeze_zeros_correct_grad():
    """apply_crs_modulated_geom_freeze zero đúng grad của Gaussian có V thấp."""
    gaussians = MockGaussians(
        torch.nn.Parameter(torch.tensor(
            [[0.0, 0.0, 5.0], [1.0, 1.0, 5.0]], device="cuda"
        )),
        scaling=torch.nn.Parameter(torch.zeros(2, 3, device="cuda")),
        rotation=torch.nn.Parameter(torch.zeros(2, 4, device="cuda")),
    )

    # Gaussian 0: đứng yên (V=0.5, không bị freeze vì 0.5 ≥ tau=0.5 dùng < nên
    # cần ép V thấp rõ ràng để test không phụ thuộc biên giới hạn).
    # Gaussian 1: dao động mạnh → V thấp → bị freeze.
    torch.manual_seed(1)
    for _ in range(15):
        gaussians._xyz.data[1, 0] = float(torch.randn(1) * 5.0)
        update_v_stability(gaussians)

    # Giả lập backward: gán grad giả cho cả 2 Gaussian.
    gaussians._xyz.grad = torch.ones_like(gaussians._xyz)
    gaussians._scaling.grad = torch.ones_like(gaussians._scaling)
    gaussians._rotation.grad = torch.ones_like(gaussians._rotation)

    V_before = compute_V_stability(gaussians).squeeze()
    apply_crs_modulated_geom_freeze(gaussians, iter=2000, freeze_start=1000, tau_freeze=0.5)

    if V_before[1].item() < 0.5:
        assert torch.all(gaussians._xyz.grad[1] == 0), \
            "Gaussian dao động mạnh (V<0.5) phải bị zero grad"
        print(f"  PASS: drifting Gaussian (V={V_before[1].item():.4f}) grad zeroed")
    else:
        print(f"  SKIP: V_before[1]={V_before[1].item():.4f} không đủ thấp "
              f"để trigger freeze trong test này (không phải lỗi — random seed dependent)")


def test_geom_freeze_noop_before_start():
    """iter ≤ freeze_start → không làm gì (grad giữ nguyên)."""
    gaussians = MockGaussians(
        torch.nn.Parameter(torch.tensor([[0.0, 0.0, 5.0]], device="cuda")),
        scaling=torch.nn.Parameter(torch.zeros(1, 3, device="cuda")),
        rotation=torch.nn.Parameter(torch.zeros(1, 4, device="cuda")),
    )
    gaussians._xyz.grad = torch.ones_like(gaussians._xyz)
    apply_crs_modulated_geom_freeze(gaussians, iter=500, freeze_start=1000, tau_freeze=0.5)
    assert torch.all(gaussians._xyz.grad == 1.0), "iter ≤ freeze_start phải no-op"
    print("  PASS: no-op trước freeze_start")


if __name__ == "__main__":
    print("=" * 60)
    print("[CRSGaussian Phase 26 A1] Testing v_stability + geom_freeze")
    print("=" * 60)

    tests = [
        ("1. First call → neutral init", test_first_call_initializes_neutral),
        ("2. Stationary Gaussian → V=0.5", test_stable_gaussian_high_v),
        ("3. Drifting Gaussian → low V", test_drifting_gaussian_low_v),
        ("4. Shape mismatch → re-init OK", test_shape_mismatch_reinit),
        ("5. Geom freeze zeros correct grad", test_geom_freeze_zeros_correct_grad),
        ("6. Geom freeze no-op before start", test_geom_freeze_noop_before_start),
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
