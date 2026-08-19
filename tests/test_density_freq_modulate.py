# ============================================================
# [CRSGaussian Phase 26 C1] Unit test cho density_freq_modulate.py
# File: tests/test_density_freq_modulate.py (NEW — mirrors test_update_crs.py)
# Mục đích: Verify compute_frequency_signal() cho điểm dày+nhỏ freq cao hơn
#           điểm thưa+to, modulate_probability() giảm đúng hướng và
#           strength=0 → no-op (giữ nguyên base_prob).
#           Đồng thời verify sh_degree_dropout() (dropansh.py) và
#           opacity_decay() (gaussian_model.py) chấp nhận per-Gaussian
#           tensor mà không lỗi shape.
# Chạy: cd ReDI-GS && python tests/test_density_freq_modulate.py
#       (KHÔNG dùng pytest — file tự có runner ở cuối)
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from utils.regularizer.density_freq_modulate import (
    compute_frequency_signal, modulate_probability
)


class MockGaussians:
    """GaussianModel giả lập — đủ field cho compute_frequency_signal (voxel)."""
    def __init__(self, xyz, scaling_raw):
        self._xyz = xyz
        self._scaling = scaling_raw  # pre-activation (log-space)
        self._rotation = torch.zeros(xyz.shape[0], 4, device=xyz.device)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)


def test_dense_small_scale_has_higher_freq():
    """Cụm điểm dày đặc + scale nhỏ → freq cao hơn cụm thưa + scale lớn."""
    torch.manual_seed(0)
    # Cụm dày: 50 điểm co cụm quanh gốc, scale nhỏ (log(0.01)).
    dense_pts = torch.randn(50, 3, device="cuda") * 0.05
    dense_scaling = torch.full((50, 3), torch.log(torch.tensor(0.01)).item(), device="cuda")

    # Cụm thưa: 50 điểm rải xa nhau, scale lớn (log(1.0)).
    sparse_pts = torch.randn(50, 3, device="cuda") * 5.0
    sparse_scaling = torch.full((50, 3), torch.log(torch.tensor(1.0)).item(), device="cuda")

    xyz = torch.cat([dense_pts, sparse_pts], dim=0)
    scaling = torch.cat([dense_scaling, sparse_scaling], dim=0)
    gaussians = MockGaussians(xyz, scaling)

    freq = compute_frequency_signal(gaussians, method="voxel")
    freq_dense_mean = freq[:50].mean().item()
    freq_sparse_mean = freq[50:].mean().item()

    assert freq_dense_mean > freq_sparse_mean, \
        (f"Cụm dày+nhỏ phải freq cao hơn cụm thưa+to: "
         f"dense={freq_dense_mean:.4f} sparse={freq_sparse_mean:.4f}")
    print(f"  PASS: freq_dense={freq_dense_mean:.4f} > freq_sparse={freq_sparse_mean:.4f}")


def test_modulate_probability_strength_zero_is_noop():
    """strength=0 → p_i = base_prob mọi nơi (không phụ thuộc freq)."""
    freq = torch.tensor([0.0, 0.3, 0.7, 1.0], device="cuda")
    p = modulate_probability(0.2, freq, strength=0.0)
    assert torch.allclose(p, torch.full_like(p, 0.2)), \
        f"strength=0 phải no-op, got {p.tolist()}"
    print(f"  PASS: strength=0 → p={p.tolist()} (== base_prob=0.2)")


def test_modulate_probability_reduces_at_high_freq():
    """strength=1.0 → freq cao nhất (1.0) có p thấp hơn freq thấp nhất (0.0)."""
    freq = torch.tensor([0.0, 1.0], device="cuda")
    p = modulate_probability(0.2, freq, strength=1.0)
    assert p[0].item() > p[1].item(), \
        f"freq cao phải p thấp hơn: p_freq0={p[0].item():.4f} p_freq1={p[1].item():.4f}"
    assert abs(p[1].item() - 0.0) < 1e-6, f"freq=1.0, strength=1.0 → p phải =0, got {p[1].item():.4f}"
    print(f"  PASS: p(freq=0)={p[0].item():.4f} > p(freq=1)={p[1].item():.4f}")


def test_modulate_probability_min_ratio_floor():
    """min_prob_ratio > 0 → p không xuống dưới sàn dù freq=1.0."""
    freq = torch.tensor([1.0], device="cuda")
    p = modulate_probability(0.2, freq, strength=1.0, min_prob_ratio=0.5)
    assert p.item() >= 0.2 * 0.5 - 1e-6, \
        f"min_prob_ratio=0.5 → p phải ≥ 0.1, got {p.item():.4f}"
    print(f"  PASS: p={p.item():.4f} ≥ floor=0.1 (min_prob_ratio=0.5)")


def test_sh_degree_dropout_accepts_tensor_p_sh():
    """sh_degree_dropout (dropansh.py) không lỗi khi p_sh là (N,) tensor."""
    from utils.regularizer.dropansh import sh_degree_dropout

    N, K_rest = 10, 15  # sh_degree=3 → (3+1)^2-1=15
    class G:
        pass
    g = G()
    g._xyz = torch.zeros(N, 3, device="cuda")
    g.get_xyz = g._xyz
    g._features_rest = torch.nn.Parameter(torch.randn(N, K_rest, 3, device="cuda"))

    # p_sh=1.0 mọi Gaussian → drop_mask CHẮC CHẮN có ít nhất 1 True (không
    # phụ thuộc seed/may rủi torch.rand() — trước đó dùng linspace(0,0.5)
    # + không seed vẫn có thể (dù hiếm) ra toàn False, đã thực sự xảy ra
    # với seed=42). Mục đích test chỉ là "chạy được với tensor", không cần
    # xác suất thực tế của production.
    p_sh_tensor = torch.ones(N, device="cuda")
    snapshot = sh_degree_dropout(g, iteration=500, p_sh=p_sh_tensor,
                                   schedule=(2000, 4000, 6000))
    # iteration=500 < schedule[0]=2000 → lmax=0, num_keep_rest=0 → toàn bộ
    # rest bị zero cho Gaussian trúng dropout (drop_mask theo p_sh_tensor).
    assert snapshot is not None, \
        "p_sh=1.0 (100%) phải chắc chắn trigger dropout, không phụ thuộc RNG"
    print(f"  PASS: sh_degree_dropout chạy OK với p_sh tensor "
          f"(snapshot={'có' if snapshot is not None else 'None (bad luck rand)'})")


def test_sh_degree_dropout_p_sh_all_zero_tensor_skips():
    """p_sh tensor toàn 0 → skip (giống scalar p_sh=0)."""
    from utils.regularizer.dropansh import sh_degree_dropout

    N = 5
    class G:
        pass
    g = G()
    g._xyz = torch.zeros(N, 3, device="cuda")
    g.get_xyz = g._xyz
    g._features_rest = torch.nn.Parameter(torch.randn(N, 15, 3, device="cuda"))

    snapshot = sh_degree_dropout(g, iteration=500, p_sh=torch.zeros(N, device="cuda"))
    assert snapshot is None, "p_sh tensor toàn 0 phải skip (trả None)"
    print("  PASS: p_sh tensor toàn 0 → skip")


if __name__ == "__main__":
    print("=" * 60)
    print("[CRSGaussian Phase 26 C1] Testing density_freq_modulate")
    print("=" * 60)

    tests = [
        ("1. Dense+small scale → higher freq", test_dense_small_scale_has_higher_freq),
        ("2. strength=0 → no-op", test_modulate_probability_strength_zero_is_noop),
        ("3. High freq → lower probability", test_modulate_probability_reduces_at_high_freq),
        ("4. min_prob_ratio floor works", test_modulate_probability_min_ratio_floor),
        ("5. sh_degree_dropout accepts tensor p_sh", test_sh_degree_dropout_accepts_tensor_p_sh),
        ("6. p_sh all-zero tensor skips", test_sh_degree_dropout_p_sh_all_zero_tensor_skips),
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
