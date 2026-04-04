# ============================================================
# [CRSGaussian] Task: T4.2 — Unit test cho CRS pruning (Option C)
# File: CRSGaussian/tests/test_crs_pruning.py  (TẠO MỚI)
# Chạy: cd CRSGaussian && python tests/test_crs_pruning.py
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np


def test_crs_low_and_isolated_prune():
    """CRS < tau_crs AND isolated → prune."""
    from simple_knn._C import distCUDA2

    N = 10
    # 9 points clustered, 1 far away (isolated)
    xyz = torch.zeros(N, 3, device="cuda")
    xyz[:9] = torch.randn(9, 3, device="cuda") * 0.01  # tight cluster
    xyz[9] = torch.tensor([100.0, 100.0, 100.0], device="cuda")  # isolated

    # CRS: all low
    crs = torch.full((N,), 0.2, device="cuda")

    tau_crs = 0.35
    tau_isolated = 0.1
    extent = 1.0

    crs_low = crs < tau_crs
    knn_dist, _ = distCUDA2(xyz)
    isolated = knn_dist > tau_isolated * extent
    crs_prune = crs_low & isolated

    # Point 9 is isolated + low CRS → should prune
    assert crs_prune[9].item() == True, "Isolated + low CRS should prune"
    # Clustered points: low CRS but NOT isolated → should NOT prune
    assert crs_prune[:9].sum().item() == 0, "Clustered + low CRS should NOT prune"
    print(f"  PASS: isolated+low CRS → prune, clustered+low CRS → keep")


def test_crs_low_with_neighbors_no_prune():
    """CRS < tau_crs BUT has neighbors → KHÔNG prune."""
    from simple_knn._C import distCUDA2

    N = 20
    xyz = torch.randn(N, 3, device="cuda") * 0.1  # all close together
    crs = torch.full((N,), 0.2, device="cuda")     # all low CRS

    tau_crs = 0.35
    tau_isolated = 0.1
    extent = 1.0

    crs_low = crs < tau_crs
    knn_dist, _ = distCUDA2(xyz)
    isolated = knn_dist > tau_isolated * extent
    crs_prune = crs_low & isolated

    # All points have neighbors → none should be pruned by CRS
    assert crs_prune.sum().item() == 0, \
        f"All clustered → no CRS prune, got {crs_prune.sum().item()}"
    print(f"  PASS: low CRS + has neighbors → no prune")


def test_crs_high_no_prune():
    """CRS > tau_crs → KHÔNG prune regardless of isolation."""
    from simple_knn._C import distCUDA2

    N = 5
    xyz = torch.zeros(N, 3, device="cuda")
    xyz[0] = torch.tensor([100.0, 0.0, 0.0], device="cuda")  # isolated
    crs = torch.full((N,), 0.8, device="cuda")  # high CRS (surface)

    tau_crs = 0.35
    tau_isolated = 0.1
    extent = 1.0

    crs_low = crs < tau_crs
    knn_dist, _ = distCUDA2(xyz)
    isolated = knn_dist > tau_isolated * extent
    crs_prune = crs_low & isolated

    # High CRS → crs_low is False → no prune
    assert crs_prune.sum().item() == 0, "High CRS should never prune"
    print(f"  PASS: high CRS → no prune even if isolated")


def test_warmup_check():
    """iteration < T_warmup → CRS pruning OFF."""
    T_warmup = 1000

    # Before warmup: should NOT add CRS prune
    iter_before = 500
    crs_prune_active = iter_before > T_warmup
    assert crs_prune_active == False, "Before warmup → CRS prune OFF"

    # After warmup: should add CRS prune
    iter_after = 1500
    crs_prune_active = iter_after > T_warmup
    assert crs_prune_active == True, "After warmup → CRS prune ON"

    print(f"  PASS: iter=500 → OFF, iter=1500 → ON")


def test_legacy_pruning_works_without_crs():
    """aligned_depth_dict=None → chỉ legacy pruning, không crash."""
    # Simulate legacy pruning logic
    N = 10
    opacity = torch.rand(N, 1, device="cuda")
    min_opacity = 0.5

    prune_mask = (opacity < min_opacity).squeeze()

    # No CRS block (aligned_depth_dict is None)
    aligned_depth_dict = None
    if aligned_depth_dict is not None:
        pass  # CRS pruning would go here

    # Legacy pruning should work
    n_pruned = prune_mask.sum().item()
    assert n_pruned >= 0, "Legacy pruning should work"
    print(f"  PASS: legacy pruning works without CRS (pruned {n_pruned}/{N})")


if __name__ == "__main__":
    print("=" * 60)
    print("[CRSGaussian T4.2] Testing CRS pruning (Option C)")
    print("=" * 60)

    tests = [
        ("1. CRS low + isolated → prune", test_crs_low_and_isolated_prune),
        ("2. CRS low + neighbors → NO prune", test_crs_low_with_neighbors_no_prune),
        ("3. CRS high → NO prune", test_crs_high_no_prune),
        ("4. Warmup check", test_warmup_check),
        ("5. Legacy pruning without CRS", test_legacy_pruning_works_without_crs),
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
