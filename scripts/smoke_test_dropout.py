"""
[CRSGaussian Track B] Smoke test cho compute_dropout_mask().

Chạy trên server sau khi copy utils/regularizer/sh_dropout.py:
    cd ~/CRSGaussian && python scripts/smoke_test_dropout.py

Kiểm tra:
  1. Uniform mode: drop rate ≈ base (noise ±2%)
  2. sh_norm mode: drop prob varies per-Gaussian (std > 0)
  3. Hybrid mode: cả CRS và sh_norm contribute
  4. max_drop cap hoạt động (p95 ≤ max_drop)
  5. Output shape đúng, dtype đúng

KHÔNG cần load dataset — dùng dummy GaussianModel.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.regularizer.sh_dropout import compute_dropout_mask, log_dropout_stats


class DummyGaussians:
    """Mock minimal interface của GaussianModel cho smoke test."""
    def __init__(self, N=1000, K=3, device="cuda"):
        self.N = N
        # get_xyz property → (N, 3)
        self._xyz = torch.randn(N, 3, device=device)
        # get_crs → (N, 1), bimodal: ~30% floater (low CRS), ~70% surface (high)
        crs_bimodal = torch.cat([
            torch.rand(int(0.3 * N), device=device) * 0.4,         # [0, 0.4] floater
            torch.rand(N - int(0.3 * N), device=device) * 0.3 + 0.7,  # [0.7, 1] surface
        ])
        # _crs_score stored as logit
        self._crs_score = torch.logit(crs_bimodal.clamp(1e-4, 1 - 1e-4)).unsqueeze(-1)
        # _features_rest → (N, K, 3), Gaussians có norm khác nhau
        # 20% Gaussians có norm cao (suspect memorize), còn lại norm thấp
        rest = torch.randn(N, K, 3, device=device) * 0.1
        rest[:int(0.2 * N)] *= 10.0  # 20% có norm ~10x
        self._features_rest = rest

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_crs(self):
        return torch.sigmoid(self._crs_score)


def test_uniform():
    print("\n=== Test 1: mode=uniform, base=0.2 ===")
    g = DummyGaussians(N=10000)
    keep_mask, drop_prob = compute_dropout_mask(g, mode="uniform", base=0.2)

    assert keep_mask.dtype == torch.bool, f"keep_mask dtype: {keep_mask.dtype}"
    assert keep_mask.shape == (10000,), f"keep_mask shape: {keep_mask.shape}"
    assert drop_prob.shape == (10000,), f"drop_prob shape: {drop_prob.shape}"

    actual_drop = 1 - keep_mask.float().mean().item()
    print(f"  expected drop ~0.20, actual = {actual_drop:.4f}")
    assert abs(actual_drop - 0.2) < 0.02, "drop rate deviates from expected"
    print(f"  drop_prob unique values: {drop_prob.unique().tolist()}")
    assert drop_prob.std().item() < 1e-5, "uniform should have zero variance"
    print("  [PASS] uniform constant drop rate")


def test_sh_norm():
    print("\n=== Test 2: mode=sh_norm, base=0.1, w_sh=0.5 ===")
    g = DummyGaussians(N=10000)
    keep_mask, drop_prob = compute_dropout_mask(
        g, mode="sh_norm", base=0.1, w_sh=0.5, max_drop=0.6
    )

    assert drop_prob.std().item() > 0.01, "sh_norm should have variance across Gaussians"
    print(f"  drop_prob mean = {drop_prob.mean().item():.4f}, "
          f"std = {drop_prob.std().item():.4f}, "
          f"max = {drop_prob.max().item():.4f}")
    print(f"  drop_prob p50 = {drop_prob.quantile(0.5).item():.4f}, "
          f"p95 = {drop_prob.quantile(0.95).item():.4f}")
    assert drop_prob.max().item() <= 0.6 + 1e-5, "max_drop cap violated"

    actual_drop = 1 - keep_mask.float().mean().item()
    print(f"  actual drop = {actual_drop:.4f}")
    assert 0.05 < actual_drop < 0.5, "drop rate out of reasonable range"

    # Verify: Gaussians with high sh_norm should have higher drop_prob
    sh_norm = g._features_rest.flatten(1).norm(dim=1)
    top_idx = sh_norm.argsort(descending=True)[:100]
    bot_idx = sh_norm.argsort(descending=False)[:100]
    top_drop = drop_prob[top_idx].mean().item()
    bot_drop = drop_prob[bot_idx].mean().item()
    print(f"  top-100 sh_norm → drop_prob mean = {top_drop:.4f}")
    print(f"  bot-100 sh_norm → drop_prob mean = {bot_drop:.4f}")
    assert top_drop > bot_drop + 0.1, "sh_norm signal không correlate với drop_prob"
    print("  [PASS] sh_norm mode correlates drop_prob với ||features_rest||")


def test_hybrid():
    print("\n=== Test 3: mode=hybrid, base=0.1, w_crs=0.3, w_sh=0.3 ===")
    g = DummyGaussians(N=10000)
    keep_mask, drop_prob = compute_dropout_mask(
        g, mode="hybrid", base=0.1, w_crs=0.3, w_sh=0.3, max_drop=0.6
    )

    assert drop_prob.std().item() > 0.01, "hybrid should have variance"
    print(f"  drop_prob mean = {drop_prob.mean().item():.4f}, "
          f"std = {drop_prob.std().item():.4f}, "
          f"max = {drop_prob.max().item():.4f}")

    # Verify: low-CRS Gaussians should drop more
    crs = g.get_crs.squeeze(-1)
    low_crs_idx = crs.argsort()[:100]
    high_crs_idx = crs.argsort(descending=True)[:100]
    low_crs_drop = drop_prob[low_crs_idx].mean().item()
    high_crs_drop = drop_prob[high_crs_idx].mean().item()
    print(f"  low-CRS (top-100) drop_prob mean = {low_crs_drop:.4f}")
    print(f"  high-CRS (top-100) drop_prob mean = {high_crs_drop:.4f}")
    assert low_crs_drop > high_crs_drop, "CRS signal không correlate với drop_prob"
    print("  [PASS] hybrid mode: low-CRS → higher drop_prob")


def test_cap():
    print("\n=== Test 4: max_drop cap ===")
    g = DummyGaussians(N=1000)
    # Base cao + w_sh cao → dễ vượt cap nếu không clamp
    keep_mask, drop_prob = compute_dropout_mask(
        g, mode="sh_norm", base=0.3, w_sh=1.0, max_drop=0.5
    )
    assert drop_prob.max().item() <= 0.5 + 1e-5, \
        f"cap violated: max = {drop_prob.max().item()}"
    print(f"  max drop_prob = {drop_prob.max().item():.4f} ≤ 0.5 ✓")
    print("  [PASS] max_drop cap works")


def test_shape_and_dtype():
    print("\n=== Test 5: Shape + dtype invariance ===")
    for N in [1, 100, 50000]:
        g = DummyGaussians(N=N)
        keep_mask, drop_prob = compute_dropout_mask(g, mode="hybrid", base=0.1, w_crs=0.3, w_sh=0.3)
        assert keep_mask.shape == (N,), f"N={N}: keep_mask shape {keep_mask.shape}"
        assert keep_mask.dtype == torch.bool, f"N={N}: keep_mask dtype {keep_mask.dtype}"
        assert drop_prob.shape == (N,)
        assert drop_prob.dtype == torch.float32
    print("  [PASS] shapes + dtypes OK for N ∈ {1, 100, 50000}")


def test_error_mode():
    print("\n=== Test 6: Invalid mode raises ValueError ===")
    g = DummyGaussians(N=100)
    try:
        compute_dropout_mask(g, mode="invalid_mode")
        print("  [FAIL] should have raised ValueError")
        return
    except ValueError as e:
        print(f"  raised as expected: {e}")
        print("  [PASS] unknown mode raises")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("[WARN] No CUDA — running on CPU (slower but OK for smoke test)")
        # Patch DummyGaussians default device
        import functools
        _orig_init = DummyGaussians.__init__
        DummyGaussians.__init__ = functools.partialmethod(_orig_init, device="cpu")

    test_uniform()
    test_sh_norm()
    test_hybrid()
    test_cap()
    test_shape_and_dtype()
    test_error_mode()
    print("\n========================================")
    print("[Track B] ALL SMOKE TESTS PASSED")
    print("========================================")
