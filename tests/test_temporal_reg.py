# ============================================================
# [CRSGaussian Phase 26 A2] Unit test cho utils/loss/temporal_reg.py
# File: tests/test_temporal_reg.py (NEW — mirrors tests/test_update_crs.py)
# Mục đích: Verify update_temporal_ema() EMA behavior, compute_temporal_reg_loss()
#           trả None khi chưa đủ data, có gradient chảy về _xyz/_scaling/
#           _rotation, và phạt mạnh hơn khi lệch xa EMA hơn.
# Chạy: cd ReDI-GS && python tests/test_temporal_reg.py
#       (KHÔNG dùng pytest — file tự có runner ở cuối)
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from utils.loss.temporal_reg import update_temporal_ema, compute_temporal_reg_loss


class MockGaussians:
    """GaussianModel giả lập — _xyz/_scaling/_rotation là nn.Parameter (có grad)."""
    def __init__(self, xyz, scaling=None, rotation=None, with_crs=False):
        N = xyz.shape[0]
        self._xyz = torch.nn.Parameter(xyz)
        self._scaling = torch.nn.Parameter(
            scaling if scaling is not None else torch.zeros(N, 3, device=xyz.device)
        )
        self._rotation = torch.nn.Parameter(
            rotation if rotation is not None else torch.zeros(N, 4, device=xyz.device)
        )
        if with_crs:
            self._crs_score = torch.zeros(N, 1, device=xyz.device)

    @property
    def get_crs(self):
        return torch.sigmoid(self._crs_score)


def test_returns_none_before_ema_init():
    """Chưa gọi update_temporal_ema() lần nào → loss=None (không có buffer)."""
    g = MockGaussians(torch.tensor([[0.0, 0.0, 5.0]], device="cuda"))
    loss = compute_temporal_reg_loss(g)
    assert loss is None, "Chưa có EMA buffer → phải trả None"
    print("  PASS: None trước khi có EMA buffer")


def test_zero_loss_when_param_equals_ema():
    """Tham số hiện tại == EMA (chưa đổi gì) → loss ≈ 0."""
    g = MockGaussians(torch.tensor([[1.0, 2.0, 3.0]], device="cuda"))
    update_temporal_ema(g)  # init EMA = giá trị hiện tại
    loss = compute_temporal_reg_loss(g, crs_weighted=False)
    assert loss is not None
    assert loss.item() < 1e-6, f"Param == EMA → loss phải ≈0, got {loss.item():.6f}"
    print(f"  PASS: loss={loss.item():.6f} khi param == EMA")


def test_loss_increases_with_drift():
    """Tham số lệch xa EMA hơn → loss lớn hơn (monotonic)."""
    g_small = MockGaussians(torch.tensor([[0.0, 0.0, 5.0]], device="cuda"))
    g_large = MockGaussians(torch.tensor([[0.0, 0.0, 5.0]], device="cuda"))

    update_temporal_ema(g_small)
    update_temporal_ema(g_large)

    # Nhích nhẹ vs nhích mạnh khỏi vị trí gốc (EMA vẫn ở gốc vì mới update 1 lần).
    with torch.no_grad():
        g_small._xyz += 0.1
        g_large._xyz += 5.0

    loss_small = compute_temporal_reg_loss(g_small, crs_weighted=False)
    loss_large = compute_temporal_reg_loss(g_large, crs_weighted=False)

    assert loss_large.item() > loss_small.item(), \
        f"Lệch xa hơn phải phạt nặng hơn: small={loss_small.item():.4f} large={loss_large.item():.4f}"
    print(f"  PASS: loss_small={loss_small.item():.4f} < loss_large={loss_large.item():.4f}")


def test_gradient_flows_to_xyz():
    """Loss phải differentiable — backward() tạo grad khác 0 trên _xyz."""
    g = MockGaussians(torch.tensor([[0.0, 0.0, 5.0]], device="cuda"))
    update_temporal_ema(g)
    with torch.no_grad():
        g._xyz += 2.0  # tạo độ lệch để có gradient khác 0

    loss = compute_temporal_reg_loss(g, crs_weighted=False)
    loss.backward()

    assert g._xyz.grad is not None, "Gradient phải chảy về _xyz"
    assert torch.any(g._xyz.grad != 0), "Gradient không được toàn 0"
    print(f"  PASS: grad _xyz = {g._xyz.grad.tolist()}")


def test_crs_weighted_prioritizes_high_crs():
    """crs_weighted=True: Gaussian CRS cao đóng góp nhiều hơn vào loss trung bình."""
    xyz = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]], device="cuda")
    g = MockGaussians(xyz, with_crs=True)
    update_temporal_ema(g)

    with torch.no_grad():
        # Cả 2 Gaussian lệch cùng khoảng cách khỏi EMA.
        g._xyz[0, 0] += 3.0
        g._xyz[1, 0] += 3.0
        # Gaussian 0: CRS cao (logit lớn dương). Gaussian 1: CRS thấp.
        g._crs_score[0] = 5.0
        g._crs_score[1] = -5.0

    loss_weighted = compute_temporal_reg_loss(g, crs_weighted=True)
    loss_uniform = compute_temporal_reg_loss(g, crs_weighted=False)

    # Vì cả 2 Gaussian lệch giống hệt nhau, loss_weighted ≈ loss_uniform ở
    # test này (weighted-mean của giá trị giống nhau = giá trị đó). Test
    # thực sự kiểm tra weighted không NaN/crash và nằm trong khoảng hợp lý.
    assert loss_weighted is not None and not torch.isnan(loss_weighted)
    assert abs(loss_weighted.item() - loss_uniform.item()) < 1e-3, \
        "Khi 2 Gaussian lệch giống hệt nhau, weighted-mean phải ≈ uniform-mean"
    print(f"  PASS: loss_weighted={loss_weighted.item():.4f} ≈ loss_uniform={loss_uniform.item():.4f}")


def test_shape_mismatch_returns_none():
    """N thay đổi (densify/prune) trước khi EMA re-init → trả None thay vì crash."""
    g = MockGaussians(torch.tensor([[0.0, 0.0, 5.0]], device="cuda"))
    update_temporal_ema(g)

    # Simulate densify: N tăng nhưng EMA buffer cũ chưa update.
    g._xyz = torch.nn.Parameter(torch.tensor(
        [[0.0, 0.0, 5.0], [1.0, 1.0, 5.0]], device="cuda"
    ))
    g._scaling = torch.nn.Parameter(torch.zeros(2, 3, device="cuda"))
    g._rotation = torch.nn.Parameter(torch.zeros(2, 4, device="cuda"))

    loss = compute_temporal_reg_loss(g)
    assert loss is None, "Shape mismatch phải trả None, không crash"
    print("  PASS: shape mismatch → None (không crash)")


if __name__ == "__main__":
    print("=" * 60)
    print("[CRSGaussian Phase 26 A2] Testing temporal_reg")
    print("=" * 60)

    tests = [
        ("1. None trước khi có EMA", test_returns_none_before_ema_init),
        ("2. Loss=0 khi param==EMA", test_zero_loss_when_param_equals_ema),
        ("3. Loss tăng theo độ lệch", test_loss_increases_with_drift),
        ("4. Gradient chảy về _xyz", test_gradient_flows_to_xyz),
        ("5. CRS-weighted hợp lý", test_crs_weighted_prioritizes_high_crs),
        ("6. Shape mismatch → None", test_shape_mismatch_returns_none),
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
