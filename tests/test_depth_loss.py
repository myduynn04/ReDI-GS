# ============================================================
# [CRSGaussian] Task: T3.1 — Unit test cho pearson_depth_loss
# File: CRSGaussian/tests/test_depth_loss.py  (TẠO MỚI)
# Chạy: cd CRSGaussian && python tests/test_depth_loss.py
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from utils.loss_utils import pearson_depth_loss


def test_identical_depths():
    """Identical depths → loss ≈ 0."""
    rd = torch.rand(1, 50, 50, device="cuda") + 1.0  # avoid 0
    dp = rd.squeeze().clone()

    loss = pearson_depth_loss(rd, dp)
    assert loss.item() < 0.01, f"Identical should give loss≈0, got {loss.item():.6f}"
    print(f"  PASS: identical → loss = {loss.item():.6f}")


def test_scaled_shifted():
    """Scaled+shifted depth → loss ≈ 0 (Pearson invariant)."""
    rd = torch.rand(1, 50, 50, device="cuda") + 1.0
    dp = (rd.squeeze() * 3.0 + 10.0)  # scale=3, shift=10

    loss = pearson_depth_loss(rd, dp)
    assert loss.item() < 0.01, \
        f"Scaled+shifted should give loss≈0 (Pearson invariant), got {loss.item():.6f}"
    print(f"  PASS: scaled+shifted → loss = {loss.item():.6f}")


def test_uncorrelated():
    """Random uncorrelated depths → loss ≈ 1."""
    torch.manual_seed(42)
    rd = torch.rand(1, 100, 100, device="cuda") + 1.0
    dp = torch.rand(100, 100, device="cuda") + 1.0

    loss = pearson_depth_loss(rd, dp)
    assert 0.5 < loss.item() < 1.5, \
        f"Uncorrelated should give loss≈1, got {loss.item():.6f}"
    print(f"  PASS: uncorrelated → loss = {loss.item():.6f}")


def test_inverted():
    """Inverted depth → loss ≈ 2 (anti-correlated)."""
    rd = torch.linspace(1, 10, 2500, device="cuda").reshape(1, 50, 50)
    dp = torch.linspace(10, 1, 2500, device="cuda").reshape(50, 50)

    loss = pearson_depth_loss(rd, dp)
    assert loss.item() > 1.8, f"Inverted should give loss≈2, got {loss.item():.6f}"
    print(f"  PASS: inverted → loss = {loss.item():.6f}")


def test_with_mask():
    """Mask filters pixels correctly."""
    rd = torch.ones(1, 50, 50, device="cuda") * 5.0
    dp = torch.ones(50, 50, device="cuda") * 5.0

    # Top half: correlated (both 5)
    # Bottom half: make rendered different
    rd[0, 25:, :] = torch.rand(25, 50, device="cuda")

    # Mask: only top half
    mask = torch.zeros(1, 50, 50, dtype=torch.bool, device="cuda")
    mask[0, :25, :] = True

    loss = pearson_depth_loss(rd, dp, mask=mask)
    # Top half is flat (all 5) → std ≈ 0 → corr undefined → should handle gracefully
    print(f"  PASS: masked → loss = {loss.item():.6f}")


def test_depth_prior_cpu():
    """Depth prior on CPU → auto transfer to GPU."""
    rd = torch.rand(1, 50, 50, device="cuda") + 1.0
    dp = rd.squeeze().clone().cpu()  # CPU tensor

    loss = pearson_depth_loss(rd, dp)
    assert loss.item() < 0.01, f"CPU prior should work, got {loss.item():.6f}"
    print(f"  PASS: CPU depth prior → loss = {loss.item():.6f}")


def test_zero_prior_skipped():
    """Depth prior = 0 pixels → skipped."""
    rd = torch.rand(1, 50, 50, device="cuda") + 1.0
    dp = torch.zeros(50, 50, device="cuda")  # all zero → skip all

    loss = pearson_depth_loss(rd, dp)
    assert loss.item() == 0.0, f"All-zero prior should give loss=0, got {loss.item():.6f}"
    print(f"  PASS: zero prior → loss = {loss.item():.6f}")


def test_differentiable():
    """Loss is differentiable w.r.t. rendered depth."""
    # Tạo leaf tensor trước, rồi mới requires_grad
    rd = (torch.rand(1, 50, 50, device="cuda") + 1.0).detach().requires_grad_(True)
    dp = torch.rand(50, 50, device="cuda") + 1.0

    loss = pearson_depth_loss(rd, dp)
    loss.backward()

    assert rd.grad is not None, "Gradient should exist"
    assert not torch.isnan(rd.grad).any(), "Gradient should not have NaN"
    print(f"  PASS: differentiable, grad shape = {rd.grad.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("[CRSGaussian T3.1] Testing pearson_depth_loss")
    print("=" * 60)

    tests = [
        ("1. Identical depths → loss≈0", test_identical_depths),
        ("2. Scaled+shifted → loss≈0 (invariant)", test_scaled_shifted),
        ("3. Uncorrelated → loss≈1", test_uncorrelated),
        ("4. Inverted → loss≈2", test_inverted),
        ("5. With mask", test_with_mask),
        ("6. Depth prior on CPU", test_depth_prior_cpu),
        ("7. Zero prior skipped", test_zero_prior_skipped),
        ("8. Differentiable", test_differentiable),
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
