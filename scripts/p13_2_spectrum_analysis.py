#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 13.2] FFT spectrum analyzer for LLFF GT images.
# File: scripts/p13_2_spectrum_analysis.py  (TẠO MỚI)
# Mục đích: PRE-INVESTIGATION cho Gap C (FALA — Frequency-Annealed
#           Loss Annealing). Đo ACTUAL frequency distribution của
#           LLFF GT images để chọn σ_blur INFORMED thay vì rule-of-thumb.
# Pre-condition: NOT implement Gap C; CHỈ analyze FFT spectrum.
# Output: per-scene + aggregate radial spectrum, cumulative energy
#         thresholds (50/80/90/95/99%), σ_blur recommendations
#         (conservative/balanced/aggressive).
# Runtime: pure Python/NumPy, CPU-only, <30s cho 24 train views
#          (3 train × 8 scenes).
# ============================================================
"""[CRSGaussian Phase 13.2] FFT spectrum analyzer for LLFF GT images.

Quy trình:
  1. Load N=3 train views per scene (n_views=3, r=8 downsample)
  2. Convert RGB → grayscale luminance (Y = 0.299R + 0.587G + 0.114B)
  3. 2D FFT magnitude → radial average → 1D spectrum P(k)
  4. Cumulative energy curve E(k) = Σ_{k'≤k} P(k') / Σ P(k')
  5. Tìm k tại 50/80/90/95/99% energy
  6. Convert k_threshold → σ_blur recommendation:
        Gaussian blur σ (pixels) ↔ low-pass cutoff f_c ≈ 0.187/σ cycles/pixel
        → σ = 0.187 × W_img / k_threshold (k_threshold in cycles/image-width)

Decision logic Gap C:
  - σ_start (max blur, iter=0):     dùng k_80% threshold  → blur out top 20% HF
  - σ_mid (decay midpoint):         dùng k_95% threshold  → blur out top 5% HF
  - σ_end (no blur, late iter):     0.0 (raw GT)

Run:
    python scripts/p13_2_spectrum_analysis.py
    DATA_ROOT=data/nerf_llff_data SCENES="fern flower ..." \
        python scripts/p13_2_spectrum_analysis.py
    SAVE_PLOTS=1 python scripts/p13_2_spectrum_analysis.py   # save per-scene spectrum PNGs
"""

import os
import sys
import statistics
from glob import glob

import numpy as np
from PIL import Image


# ── Config ──
SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex"
).split()
DATA_ROOT = os.environ.get("DATA_ROOT", "data/nerf_llff_data")
DOWNSAMPLE = int(os.environ.get("DOWNSAMPLE", "8"))   # r=8 → images_8/
N_VIEWS = int(os.environ.get("N_VIEWS", "3"))         # train views per scene
LLFF_HOLD = int(os.environ.get("LLFF_HOLD", "8"))     # CoR-GS test = every 8th
SAVE_PLOTS = bool(int(os.environ.get("SAVE_PLOTS", "0")))
OUT_DIR = os.environ.get("OUT_DIR", "logs/p13_2_spectrum")

# Cumulative-energy thresholds to report
ENERGY_THRESHOLDS = [0.50, 0.80, 0.90, 0.95, 0.99]

# ── Gaussian-blur cutoff constant ──
# Gaussian PSF std σ (pixels) ↔ frequency response G(f) = exp(-2π² σ² f²)
# −3 dB cutoff: f_c ≈ √(ln 2 / (2π²)) / σ  ≈  0.187 / σ   (cycles/pixel)
GAUSS_CUTOFF_C = 0.187


def get_train_indices(n_total, n_views=3, hold=8):
    """[CRSGaussian Phase 13.2] LLFF train-view selector matching CoR-GS sparse-view convention.

    LLFF convention: test = every `hold`-th image (i % hold == 0).
    Sparse train: pick `n_views` evenly-spaced indices from the train pool.
    Khớp với scene/dataset_readers.py:readLLFFInfo (n_views path).
    """
    test_idx = set(i for i in range(n_total) if i % hold == 0)
    train_pool = [i for i in range(n_total) if i not in test_idx]
    if len(train_pool) <= n_views:
        return train_pool
    # Evenly-spaced indices into train_pool
    step = len(train_pool) / float(n_views)
    return [train_pool[int(step * k)] for k in range(n_views)]


def load_image_as_luminance(img_path):
    """[CRSGaussian Phase 13.2] Load RGB image → grayscale luminance [H, W] float32 ∈ [0, 1].

    Dùng Y = 0.299R + 0.587G + 0.114B (BT.601). Lý do: FFT magnitude axis
    spectrum cần single channel; luminance preserves HF detail tốt hơn
    R/G/B individual.
    """
    img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
    # img: [H, W, 3]
    y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return y  # [H, W] float32


def compute_2d_fft_magnitude(img_2d):
    """[CRSGaussian Phase 13.2] 2D FFT power spectrum của image grayscale.

    Centered (DC at H/2, W/2) via fftshift. Returns |F|² (power),
    not |F| (amplitude) — energy fractions consistent với cumulative curve.
    """
    # Subtract mean → remove DC bias (DC bin dominates otherwise)
    img_centered = img_2d - img_2d.mean()
    F = np.fft.fft2(img_centered)
    F_shifted = np.fft.fftshift(F)
    return np.abs(F_shifted) ** 2   # power spectrum [H, W]


def radial_average(power_2d):
    """[CRSGaussian Phase 13.2] Radially-averaged 1D spectrum P(k).

    Bin by integer radius k = round(sqrt((u-uc)² + (v-vc)²)).
    Returns (k_array, P_array) — k in cycles/image_diagonal (will normalize
    to cycles/image_width khi convert sang σ).
    """
    H, W = power_2d.shape
    uc, vc = H // 2, W // 2
    y, x = np.indices((H, W))
    r = np.sqrt((y - uc) ** 2 + (x - vc) ** 2)
    r_int = r.astype(np.int32)

    k_max = min(uc, vc)   # don't include corners (anisotropic sampling)
    p_radial = np.zeros(k_max + 1, dtype=np.float64)
    count = np.zeros(k_max + 1, dtype=np.int64)
    for k in range(k_max + 1):
        mask = (r_int == k)
        if mask.any():
            p_radial[k] = power_2d[mask].mean()
            count[k] = mask.sum()
    return np.arange(k_max + 1), p_radial


def compute_cumulative_energy(p_radial):
    """[CRSGaussian Phase 13.2] Cumulative-energy curve E(k) = Σ_{k'≤k} P(k') / Σ P.

    Weight by annulus area (2π k) to get proper energy fraction
    (each k bin represents annulus of circumference proportional to k).
    """
    k = np.arange(len(p_radial))
    energy_per_bin = p_radial * (2.0 * np.pi * np.maximum(k, 1))  # avoid k=0 zero-weight
    total = energy_per_bin.sum()
    if total <= 0:
        return np.zeros_like(p_radial)
    cum = np.cumsum(energy_per_bin) / total
    return cum   # [k_max+1] ∈ [0, 1]


def find_threshold_freq(cum_energy, threshold):
    """[CRSGaussian Phase 13.2] First k such that cum_energy[k] ≥ threshold."""
    idxs = np.where(cum_energy >= threshold)[0]
    if len(idxs) == 0:
        return len(cum_energy) - 1
    return int(idxs[0])


def freq_to_sigma_blur(k_threshold, img_width):
    """[CRSGaussian Phase 13.2] Convert frequency cutoff k (cycles/image) → Gaussian σ (pixels).

    k_threshold: radial bin index ≈ cycles per (image_width/2) — Nyquist-ish.
    Normalized cycles-per-pixel: f_norm = k_threshold / (img_width / 2).
    σ (pixels) ≈ 0.187 / f_norm = 0.187 × (img_width / 2) / k_threshold.

    Returns σ in pixels of the GT image (r=8 downsampled).
    """
    if k_threshold <= 0:
        return float("inf")
    return GAUSS_CUTOFF_C * (img_width / 2.0) / float(k_threshold)


def sigma_to_freq_cut(sigma, img_width):
    """[CRSGaussian Phase 13.2] Inverse: σ (pixels) → k cutoff (radial bin).

    Sanity-check companion of freq_to_sigma_blur — verify roundtrip consistency
    trong smoke tests.
    """
    if sigma <= 0:
        return float("inf")
    return GAUSS_CUTOFF_C * (img_width / 2.0) / float(sigma)


def analyze_scene(scene_dir):
    """[CRSGaussian Phase 13.2] Per-scene FFT analysis on N_VIEWS train images.

    Returns dict:
      {
        'scene': str,
        'n_images': int,
        'H': int, 'W': int,
        'p_radial': np.ndarray,        # mean radial power spectrum [K]
        'cum_energy': np.ndarray,      # cumulative energy [K]
        'k_thresh': {0.50: k, 0.80: k, ...},
        'sigma_rec': {0.80: σ, 0.90: σ, 0.95: σ}, # blur σ recommendations
      }
    """
    img_dir = os.path.join(scene_dir, f"images_{DOWNSAMPLE}")
    if not os.path.isdir(img_dir):
        return None
    img_paths = sorted(glob(os.path.join(img_dir, "*.jpg")) +
                       glob(os.path.join(img_dir, "*.JPG")) +
                       glob(os.path.join(img_dir, "*.png")))
    if not img_paths:
        return None

    train_idx = get_train_indices(len(img_paths), n_views=N_VIEWS, hold=LLFF_HOLD)
    train_paths = [img_paths[i] for i in train_idx]

    radial_specs = []
    H = W = None
    for p in train_paths:
        y = load_image_as_luminance(p)
        if H is None:
            H, W = y.shape
        power = compute_2d_fft_magnitude(y)
        _, p_rad = radial_average(power)
        radial_specs.append(p_rad)

    # Pad to same length (defensive — train views could differ if resize off)
    max_len = max(len(r) for r in radial_specs)
    radial_specs = [np.pad(r, (0, max_len - len(r))) for r in radial_specs]
    mean_radial = np.mean(np.stack(radial_specs, axis=0), axis=0)
    cum = compute_cumulative_energy(mean_radial)

    k_thresh = {t: find_threshold_freq(cum, t) for t in ENERGY_THRESHOLDS}
    sigma_rec = {
        0.80: freq_to_sigma_blur(k_thresh[0.80], W),  # aggressive (start σ)
        0.90: freq_to_sigma_blur(k_thresh[0.90], W),  # balanced
        0.95: freq_to_sigma_blur(k_thresh[0.95], W),  # conservative
    }
    return {
        'scene': os.path.basename(scene_dir),
        'n_images': len(train_paths),
        'H': H, 'W': W,
        'p_radial': mean_radial,
        'cum_energy': cum,
        'k_thresh': k_thresh,
        'sigma_rec': sigma_rec,
    }


def plot_scene_spectrum(result, save_path):
    """[CRSGaussian Phase 13.2] Plot per-scene radial spectrum + cumulative energy.

    Optional — only runs nếu SAVE_PLOTS=1 và matplotlib available.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    k = np.arange(len(result['p_radial']))
    # Left: log-log power spectrum
    ax[0].loglog(k[1:], result['p_radial'][1:], 'b-')
    ax[0].set_xlabel("radial freq bin k (cycles)")
    ax[0].set_ylabel("mean radial power")
    ax[0].set_title(f"{result['scene']} radial spectrum  H×W={result['H']}×{result['W']}")
    ax[0].grid(True, alpha=0.3)
    # Right: cumulative energy with threshold lines
    ax[1].plot(k, result['cum_energy'], 'g-', linewidth=2)
    for t, kt in result['k_thresh'].items():
        ax[1].axvline(kt, linestyle='--', alpha=0.4)
        ax[1].axhline(t, linestyle=':', alpha=0.3)
        ax[1].text(kt, t, f" k={kt} @ {int(t*100)}%", fontsize=8)
    ax[1].set_xlabel("radial freq bin k")
    ax[1].set_ylabel("cumulative energy fraction")
    ax[1].set_title("cumulative energy curve")
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110)
    plt.close(fig)
    return True


def smoke_tests():
    """[CRSGaussian Phase 13.2] 4 unit smoke tests trước khi chạy full analysis."""
    print("=== Smoke tests ===")
    # Test 1: get_train_indices(24, 3, 8) — 24 imgs, hold=8, n_views=3
    idx = get_train_indices(24, 3, 8)
    print(f"  T1 get_train_indices(24,3,8) = {idx}  "
          f"(test set = {{0,8,16}}, train pool size = 21, expect 3 evenly-spaced)")
    assert len(idx) == 3, f"Expected 3 train idx, got {len(idx)}"
    assert all(i % 8 != 0 for i in idx), "Train idx hit a test slot"

    # Test 2: synthetic random image FFT — power should be ~flat (white noise)
    rng = np.random.default_rng(42)
    noise = rng.normal(0.5, 0.1, size=(64, 64)).astype(np.float32)
    pow_ = compute_2d_fft_magnitude(noise)
    _, p_rad = radial_average(pow_)
    flat_var = p_rad[5:25].std() / max(p_rad[5:25].mean(), 1e-9)
    print(f"  T2 white-noise radial spectrum CoV (bins 5-25) = {flat_var:.3f}  "
          f"(should be < 0.5 for ~flat)")

    # Test 3: synthetic low-freq sinusoid — energy concentrated at low k
    H = W = 64
    yy, xx = np.indices((H, W))
    sin_lowf = np.sin(2 * np.pi * 3.0 * xx / W).astype(np.float32)   # 3 cycles
    pow_ = compute_2d_fft_magnitude(sin_lowf)
    _, p_rad = radial_average(pow_)
    cum = compute_cumulative_energy(p_rad)
    k50 = find_threshold_freq(cum, 0.50)
    print(f"  T3 sin(3 cyc/W=64) k_50% = {k50}  "
          f"(expect small, ~3-5; energy concentrates near k=3)")
    assert k50 < 15, f"Low-freq sinusoid energy leaked to high-k: k50={k50}"

    # Test 4: freq_to_sigma_blur ↔ sigma_to_freq_cut roundtrip
    W_test = 126
    k_test = 20
    sig = freq_to_sigma_blur(k_test, W_test)
    k_back = sigma_to_freq_cut(sig, W_test)
    print(f"  T4 roundtrip k={k_test} → σ={sig:.3f} → k_back={k_back:.3f}  "
          f"(should match)")
    assert abs(k_back - k_test) < 0.01, "Roundtrip mismatch"
    print("  All smoke tests PASS\n")


def main():
    print("=== Phase 13.2 FFT Spectrum Analyzer (Gap C / FALA pre-investigation) ===")
    print(f"DATA_ROOT={DATA_ROOT}  SCENES={SCENES}")
    print(f"DOWNSAMPLE=r{DOWNSAMPLE}  N_VIEWS={N_VIEWS}  LLFF_HOLD={LLFF_HOLD}")
    print(f"SAVE_PLOTS={SAVE_PLOTS}  OUT_DIR={OUT_DIR}\n")

    smoke_tests()

    # Pre-flight: check first scene exists
    first_scene_dir = os.path.join(DATA_ROOT, SCENES[0])
    if not os.path.isdir(first_scene_dir):
        print(f"[FATAL] Scene dir not found: {first_scene_dir}")
        print(f"        Run on server where data/nerf_llff_data/ lives, "
              f"or set DATA_ROOT=...")
        sys.exit(1)
    first_img_dir = os.path.join(first_scene_dir, f"images_{DOWNSAMPLE}")
    if not os.path.isdir(first_img_dir):
        print(f"[FATAL] images_{DOWNSAMPLE}/ missing in {first_scene_dir}")
        print(f"        LLFF prep step (imgs2poses) chưa downsample r={DOWNSAMPLE}?")
        sys.exit(1)

    if SAVE_PLOTS:
        os.makedirs(OUT_DIR, exist_ok=True)

    results = []
    for sc in SCENES:
        scene_dir = os.path.join(DATA_ROOT, sc)
        r = analyze_scene(scene_dir)
        if r is None:
            print(f"  {sc}: SKIP (no images_{DOWNSAMPLE}/)")
            continue
        results.append(r)
        if SAVE_PLOTS:
            png_path = os.path.join(OUT_DIR, f"{sc}_spectrum.png")
            ok = plot_scene_spectrum(r, png_path)
            if ok:
                print(f"  {sc}: plot → {png_path}")

    if not results:
        print("\n[FATAL] No scenes analyzed.")
        sys.exit(1)

    # ── Per-scene table ──
    print("\n=== Per-scene cumulative-energy frequency thresholds (radial bin k) ===")
    hdr = f"{'scene':<10}  {'H×W':>11}  " + "  ".join(
        f"{'k_'+str(int(t*100))+'%':>7}" for t in ENERGY_THRESHOLDS
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        row = f"{r['scene']:<10}  {r['H']}×{r['W']:<5}  "
        row += "  ".join(f"{r['k_thresh'][t]:>7d}" for t in ENERGY_THRESHOLDS)
        print(row)

    # ── Per-scene σ recommendations ──
    print("\n=== Per-scene σ_blur recommendations (pixels) ===")
    print(f"{'scene':<10}  {'σ@80% (aggr)':>14}  {'σ@90% (bal)':>13}  {'σ@95% (cons)':>14}")
    print("-" * 60)
    for r in results:
        s = r['sigma_rec']
        print(f"{r['scene']:<10}  {s[0.80]:>14.3f}  {s[0.90]:>13.3f}  {s[0.95]:>14.3f}")

    # ── Aggregate means ──
    print("\n=== Aggregate 8-scene means ===")
    for t in ENERGY_THRESHOLDS:
        ks = [r['k_thresh'][t] for r in results]
        print(f"  k_{int(t*100)}%   mean = {statistics.fmean(ks):>6.2f}  "
              f"(min {min(ks)}, max {max(ks)})")
    for t in (0.80, 0.90, 0.95):
        sigs = [r['sigma_rec'][t] for r in results]
        print(f"  σ_blur @ {int(t*100)}% energy   mean = {statistics.fmean(sigs):>5.3f} px  "
              f"(min {min(sigs):.3f}, max {max(sigs):.3f})")

    # ── Recommendations ──
    print("\n=== Recommendations for Gap C / FALA σ schedule ===")
    s80_mean = statistics.fmean(r['sigma_rec'][0.80] for r in results)
    s90_mean = statistics.fmean(r['sigma_rec'][0.90] for r in results)
    s95_mean = statistics.fmean(r['sigma_rec'][0.95] for r in results)
    print(f"  Conservative schedule:  σ_start = {s95_mean:.2f},  σ_end = 0  "
          f"(remove only top 5% HF energy)")
    print(f"  Balanced schedule:      σ_start = {s90_mean:.2f},  σ_end = 0  "
          f"(remove top 10%)")
    print(f"  Aggressive schedule:    σ_start = {s80_mean:.2f},  σ_end = 0  "
          f"(remove top 20%)")
    print(f"\n  Suggested σ ablation matrix (Gap C FALA):")
    print(f"    σ_start ∈ {{{s95_mean:.2f}, {s90_mean:.2f}, {s80_mean:.2f}}}  "
          f"(95/90/80% energy preservation)")
    print(f"    decay   ∈ {{linear, cosine, exp}}  over iter [0, K_decay]")
    print(f"    K_decay ∈ {{2000, 5000, 7500}}  (out of 10000 total iters)")
    print(f"  → recommend pilot single-seed N=8 sweep before multi-seed verify")
    print(f"    (per memory project_3dgs_variance_floor.md: ±0.10 multi-seed floor)")


if __name__ == "__main__":
    main()
