#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 13.2] Render-vs-GT spectrum DIAGNOSTIC.
# File: scripts/p13_2_spectrum_diagnostic.py  (TẠO MỚI)
# Mục đích: PHÂN BIỆT model fail ở band nào — tiền-đề để chọn
#           freq-axis intervention (DWTGS HF-sparsity vs FALA
#           sharpen vs non-freq axis).
# Khác với p13_2_spectrum_analysis.py (chỉ GT spectrum):
#   Script này tính Δ(k) = P_render(k) − P_gt(k) per band
#   → identify failure signature, không phải scene content.
# Inputs: output/p13_lfcf/{A0,A3}_seed42_<scene>/test/ours_10000/
#           ├── renders/00000.png ... 00002.png   ← model output
#           └── gt/00000.png      ... 00002.png   ← ground truth copy
# Output: per-scene LF/MF/HF Δ table + improvement A3-vs-A0 +
#         failure-pattern classifier + mechanism recommendation.
# Pre-condition: render.py đã chạy cho A0+A3 × 8 scenes (16 runs).
# ============================================================
"""[CRSGaussian Phase 13.2] Render vs GT spectrum diagnostic.

Pipeline per (config, scene):
  1. Load 3 test-view GT  → luminance → DC-removed 2D FFT → radial spectrum
  2. Load 3 test-view render → same → radial spectrum
  3. Mean over 3 views per set → P_gt(k), P_render(k)
  4. rel_Δ(k) = log10(P_render(k) / P_gt(k))
        > 0 = render OVER-produces HF (spurious / floater shimmer / aliasing)
        < 0 = render UNDER-produces HF (over-smooth / missing detail)
        ≈ 0 = matched (no freq-axis problem)
  5. Band-mean rel_Δ at LF/MF/HF → failure signature

Decision tree per scene:
  HF signed rel_Δ_A3 > +0.20  → "SPURIOUS HF"   → DWTGS HF-sparsity loss
  HF signed rel_Δ_A3 < −0.20  → "MISSING HF"    → FALA-sharpen / FFT loss
  LF signed rel_Δ_A3 large    → "LF mismatch"   → non-freq axis (geometry/SH)
  All bands |rel_Δ| < 0.10    → "NEAR CEILING"  → freq-axis exhausted

Improvement attribution A3 vs A0:
  Improvement(k) = |rel_Δ_A0(k)| − |rel_Δ_A3(k)|
        > 0 = A3 closed the gap at band k
        < 0 = A3 widened the gap at band k

Run:
    python scripts/p13_2_spectrum_diagnostic.py
    SAVE_PLOTS=1 python scripts/p13_2_spectrum_diagnostic.py
    CONFIGS="A0 A3" SCENES="orchids leaves" python scripts/p13_2_spectrum_diagnostic.py
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
CONFIGS = os.environ.get("CONFIGS", "A3").split()
SEED = os.environ.get("SEED", "42")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "output/p13_lfcf")
ITERATION = int(os.environ.get("ITERATION", "10000"))
SAVE_PLOTS = bool(int(os.environ.get("SAVE_PLOTS", "0")))
PLOT_OUT = os.environ.get("PLOT_OUT", "logs/p13_2_diagnostic")

# Bands defined as fraction of (W/2) — Nyquist half. W=504 → half=252.
# LF: k ∈ [1, 0.10·half] ≈ [1, 25]
# MF: [25, 100]
# HF: [100, half-1]
BAND_LF_FRAC = float(os.environ.get("BAND_LF_FRAC", "0.10"))
BAND_MF_FRAC = float(os.environ.get("BAND_MF_FRAC", "0.40"))

# Classifier thresholds on band-mean rel_Δ = log10(P_render / P_gt)
THRESH_SPURIOUS = float(os.environ.get("THRESH_SPURIOUS", "0.20"))   # > +0.20 (1.6× over)
THRESH_MISSING  = float(os.environ.get("THRESH_MISSING",  "0.20"))   # < −0.20 (0.63× under)
THRESH_MATCHED  = float(os.environ.get("THRESH_MATCHED",  "0.10"))   # |Δ| < 0.10 (1.26× tol)


# ── FFT utilities (mirror p13_2_spectrum_analysis.py) ──
def load_image_as_luminance(img_path):
    """Load RGB image → grayscale Y (BT.601) ∈ [0,1]."""
    img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def compute_2d_fft_power(img_2d):
    """DC-removed 2D FFT power spectrum, centered."""
    F = np.fft.fft2(img_2d - img_2d.mean())
    return np.abs(np.fft.fftshift(F)) ** 2


def radial_average(power_2d):
    """Radially-averaged 1D spectrum P(k)."""
    H, W = power_2d.shape
    uc, vc = H // 2, W // 2
    y, x = np.indices((H, W))
    r = np.sqrt((y - uc) ** 2 + (x - vc) ** 2).astype(np.int32)
    k_max = min(uc, vc)
    p_radial = np.zeros(k_max + 1, dtype=np.float64)
    for k in range(k_max + 1):
        mask = (r == k)
        if mask.any():
            p_radial[k] = power_2d[mask].mean()
    return p_radial   # [k_max+1]


def load_view_set(dir_path):
    """Load all PNG in dir → list of luminance [H,W]. Sorted by filename.

    LLFF test count varies per scene (N_total/hold): fern=3, leaves/orchids=4,
    flower=5, fortress/room=6, trex=7, horns=8. Accept whatever exists.
    """
    paths = sorted(glob(os.path.join(dir_path, "*.png")))
    if not paths:
        return None
    return [load_image_as_luminance(p) for p in paths]


def mean_radial_spectrum(views):
    """Per-view FFT → radial → average across views. Returns P(k)."""
    specs = []
    for y in views:
        power = compute_2d_fft_power(y)
        specs.append(radial_average(power))
    max_len = max(len(r) for r in specs)
    specs = [np.pad(r, (0, max_len - len(r))) for r in specs]
    return np.mean(np.stack(specs, axis=0), axis=0)


def rel_delta(p_render, p_gt, eps=1e-12):
    """log10(P_render / P_gt) — relative signed gap. Aligns lengths."""
    L = min(len(p_render), len(p_gt))
    pr = np.maximum(p_render[:L], eps)
    pg = np.maximum(p_gt[:L], eps)
    return np.log10(pr / pg)   # [L]


def band_mean(arr, k_lo, k_hi):
    """Mean of arr over k ∈ [k_lo, k_hi)."""
    k_lo = max(1, k_lo)   # skip DC (k=0)
    k_hi = min(len(arr), k_hi)
    if k_hi <= k_lo:
        return float("nan")
    return float(np.mean(arr[k_lo:k_hi]))


def classify_failure(lf, mf, hf):
    """Per-scene classifier from band-mean rel_Δ at HF band primarily."""
    # HF is the discriminator — most freq-axis interventions target HF
    if hf > THRESH_SPURIOUS:
        return "SPURIOUS_HF", "DWTGS HF-sparsity loss"
    if hf < -THRESH_MISSING:
        return "MISSING_HF", "FALA-sharpen / FFT loss"
    if abs(lf) > THRESH_SPURIOUS or abs(mf) > THRESH_SPURIOUS:
        return "LF_MF_MISMATCH", "non-freq axis (geometry/SH/specular)"
    if max(abs(lf), abs(mf), abs(hf)) < THRESH_MATCHED:
        return "NEAR_CEILING", "freq-axis exhausted, pivot needed"
    return "WEAK_SIGNAL", "marginal — multi-seed or other axis"


# ── Scene-level analysis ──
def analyze_scene(scene):
    """Returns dict with spectrum + diagnostic per config for one scene."""
    result = {'scene': scene, 'configs': {}}
    # GT loaded once (same for both configs — comes from same test split)
    # We load from A0's gt/ folder by convention.
    gt_dir_a0 = f"{OUTPUT_ROOT}/A0_seed{SEED}_{scene}/test/ours_{ITERATION}/gt"
    gt_views = load_view_set(gt_dir_a0)
    if gt_views is None:
        return None
    p_gt = mean_radial_spectrum(gt_views)
    result['p_gt'] = p_gt
    result['H'], result['W'] = gt_views[0].shape
    result['n_views'] = len(gt_views)

    half = min(result['H'], result['W']) // 2
    k_lf = int(BAND_LF_FRAC * half)        # 0..k_lf = LF
    k_mf = int(BAND_MF_FRAC * half)        # k_lf..k_mf = MF
    # k_mf..half = HF
    result['bands'] = {'LF': (1, k_lf), 'MF': (k_lf, k_mf), 'HF': (k_mf, half)}

    for cfg in CONFIGS:
        render_dir = f"{OUTPUT_ROOT}/{cfg}_seed{SEED}_{scene}/test/ours_{ITERATION}/renders"
        renders = load_view_set(render_dir)
        if renders is None:
            result['configs'][cfg] = None
            continue
        p_render = mean_radial_spectrum(renders)
        rdelta = rel_delta(p_render, p_gt)
        lf = band_mean(rdelta, *result['bands']['LF'])
        mf = band_mean(rdelta, *result['bands']['MF'])
        hf = band_mean(rdelta, *result['bands']['HF'])
        result['configs'][cfg] = {
            'p_render': p_render,
            'rel_delta': rdelta,
            'lf': lf, 'mf': mf, 'hf': hf,
        }

    # Per-scene classifier on A3 (winner) — if A3 in CONFIGS
    if 'A3' in result['configs'] and result['configs']['A3'] is not None:
        a3 = result['configs']['A3']
        pattern, mechanism = classify_failure(a3['lf'], a3['mf'], a3['hf'])
        result['pattern'] = pattern
        result['mechanism'] = mechanism

    return result


def plot_scene_diagnostic(result, save_path):
    """Per-scene plot: 3 spectra (GT, A0, A3) + rel_Δ curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    k = np.arange(len(result['p_gt']))
    # Left: log-log P(k)
    ax[0].loglog(k[1:], result['p_gt'][1:], 'k-', label='GT', linewidth=2)
    colors = {'A0': 'r', 'A3': 'b'}
    for cfg in CONFIGS:
        c = result['configs'].get(cfg)
        if c is None:
            continue
        kk = np.arange(len(c['p_render']))
        ax[0].loglog(kk[1:], c['p_render'][1:], colors.get(cfg, 'g') + '-',
                     label=cfg, alpha=0.7)
    ax[0].set_xlabel("radial freq bin k")
    ax[0].set_ylabel("mean radial power")
    ax[0].set_title(f"{result['scene']} radial spectrum")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Right: rel_Δ(k) per config
    ax[1].axhline(0, color='k', linestyle=':', alpha=0.5)
    for cfg in CONFIGS:
        c = result['configs'].get(cfg)
        if c is None:
            continue
        rd = c['rel_delta']
        ax[1].plot(np.arange(len(rd))[1:], rd[1:], colors.get(cfg, 'g') + '-',
                   label=f"{cfg}", alpha=0.8)
    # Band lines
    for name, (lo, hi) in result['bands'].items():
        ax[1].axvline(lo, color='gray', linestyle='--', alpha=0.3)
        ax[1].text(lo, 0.4, name, fontsize=8, alpha=0.5)
    ax[1].axhline(THRESH_SPURIOUS, color='orange', linestyle=':', alpha=0.4)
    ax[1].axhline(-THRESH_MISSING, color='orange', linestyle=':', alpha=0.4)
    ax[1].set_xlabel("radial freq bin k")
    ax[1].set_ylabel("rel_Δ = log10(P_render / P_gt)")
    ax[1].set_ylim(-1.5, 1.5)
    ax[1].set_title(f"failure signature  ({result.get('pattern', '?')})")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110)
    plt.close(fig)
    return True


def main():
    print("=== Phase 13.2 Render-vs-GT Spectrum Diagnostic ===")
    print(f"OUTPUT_ROOT={OUTPUT_ROOT}  SEED={SEED}  ITERATION={ITERATION}")
    print(f"CONFIGS={CONFIGS}  SCENES={SCENES}")
    print(f"BANDS  LF=[1, {BAND_LF_FRAC}·half]  "
          f"MF=[{BAND_LF_FRAC}, {BAND_MF_FRAC}]·half  HF=[{BAND_MF_FRAC}·half, half]")
    print(f"Classifier thresholds: spurious>+{THRESH_SPURIOUS}, "
          f"missing<−{THRESH_MISSING}, matched|·|<{THRESH_MATCHED}\n")

    # Pre-flight: check first config + scene exists
    first_dir = f"{OUTPUT_ROOT}/{CONFIGS[0]}_seed{SEED}_{SCENES[0]}/test/ours_{ITERATION}"
    if not os.path.isdir(first_dir):
        print(f"[FATAL] Missing: {first_dir}")
        print("        Run render.py first (see prompt for batch command).")
        sys.exit(1)

    if SAVE_PLOTS:
        os.makedirs(PLOT_OUT, exist_ok=True)

    all_results = []
    for sc in SCENES:
        r = analyze_scene(sc)
        if r is None:
            print(f"  {sc}: SKIP (no GT folder)")
            continue
        all_results.append(r)
        if SAVE_PLOTS:
            png = os.path.join(PLOT_OUT, f"{sc}_diagnostic.png")
            if plot_scene_diagnostic(r, png):
                print(f"  {sc}: plot → {png}")

    if not all_results:
        print("\n[FATAL] No scenes analyzed.")
        sys.exit(1)

    # ── Per-scene band-mean rel_Δ table ──
    print(f"\n=== Per-scene band-mean rel_Δ = log10(P_render / P_gt) ===")
    print(f"(positive = render OVER-produces, negative = UNDER-produces)")
    hdr = f"{'scene':<10}  {'N':>3}  "
    for cfg in CONFIGS:
        hdr += f"{cfg+' LF':>8}  {cfg+' MF':>8}  {cfg+' HF':>8}  | "
    if 'A3' in CONFIGS:
        hdr += f"  {'A3 pattern':<18}"
    print(hdr)
    print("-" * len(hdr))
    for r in all_results:
        row = f"{r['scene']:<10}  {r.get('n_views', 0):>3}  "
        for cfg in CONFIGS:
            c = r['configs'].get(cfg)
            if c is None:
                row += f"{'-':>8}  {'-':>8}  {'-':>8}  | "
            else:
                row += f"{c['lf']:>+8.3f}  {c['mf']:>+8.3f}  {c['hf']:>+8.3f}  | "
        row += f"  {r.get('pattern', '-'):<18}"
        print(row)

    # ── Improvement A3 vs A0 per band ──
    if 'A0' in CONFIGS and 'A3' in CONFIGS:
        print(f"\n=== Improvement A3 vs A0 per band  (|rel_Δ_A0| − |rel_Δ_A3|) ===")
        print(f"(positive = A3 closed the gap, negative = A3 widened it)")
        print(f"{'scene':<10}  {'imp LF':>8}  {'imp MF':>8}  {'imp HF':>8}")
        print("-" * 50)
        for r in all_results:
            a0 = r['configs'].get('A0')
            a3 = r['configs'].get('A3')
            if a0 is None or a3 is None:
                continue
            imp_lf = abs(a0['lf']) - abs(a3['lf'])
            imp_mf = abs(a0['mf']) - abs(a3['mf'])
            imp_hf = abs(a0['hf']) - abs(a3['hf'])
            print(f"{r['scene']:<10}  {imp_lf:>+8.3f}  {imp_mf:>+8.3f}  {imp_hf:>+8.3f}")

    # ── Aggregate band stats ──
    print(f"\n=== Aggregate 8-scene band-mean rel_Δ ===")
    for cfg in CONFIGS:
        vals_lf = [r['configs'][cfg]['lf'] for r in all_results
                   if r['configs'].get(cfg) is not None]
        vals_mf = [r['configs'][cfg]['mf'] for r in all_results
                   if r['configs'].get(cfg) is not None]
        vals_hf = [r['configs'][cfg]['hf'] for r in all_results
                   if r['configs'].get(cfg) is not None]
        if not vals_hf:
            continue
        print(f"  {cfg}  LF mean={statistics.fmean(vals_lf):+.3f}  "
              f"MF mean={statistics.fmean(vals_mf):+.3f}  "
              f"HF mean={statistics.fmean(vals_hf):+.3f}  (N={len(vals_hf)})")

    # ── Pattern tally ──
    if 'A3' in CONFIGS:
        print(f"\n=== A3 failure-pattern tally ===")
        tally = {}
        for r in all_results:
            p = r.get('pattern', 'UNKNOWN')
            tally.setdefault(p, []).append(r['scene'])
        for p, scenes in sorted(tally.items(), key=lambda x: -len(x[1])):
            print(f"  {p:<18}  ({len(scenes)}/8)  scenes: {scenes}")

    # ── Mechanism recommendation ──
    print(f"\n=== Mechanism recommendation ===")
    if 'A3' not in CONFIGS:
        print("  (A3 not in CONFIGS — skip recommendation)")
        return
    patterns = [r.get('pattern') for r in all_results if 'pattern' in r]
    counts = {p: patterns.count(p) for p in set(patterns)}
    dominant = max(counts.items(), key=lambda x: x[1])[0] if counts else "UNKNOWN"
    n_dom = counts.get(dominant, 0)
    print(f"  Dominant pattern: {dominant} ({n_dom}/{len(patterns)} scenes)")
    print()
    if dominant == "SPURIOUS_HF":
        print(f"  → Recommend: DWTGS-style HF SPARSITY loss")
        print(f"    Rationale: A3 over-produces HF on majority scenes → penalize")
        print(f"               novel-view HH wavelet band; expect Δ +0.1~0.3")
    elif dominant == "MISSING_HF":
        print(f"  → Recommend: FALA REVERSED  (sharpen GT instead of blur)")
        print(f"    OR HF-emphasis L1 loss on high-pass(GT - render)")
        print(f"    Rationale: A3 under-produces HF → standard FALA blur HURTS;")
        print(f"               need to AMPLIFY HF supervision, not damp it")
    elif dominant == "LF_MF_MISMATCH":
        print(f"  → Recommend: PIVOT off freq-axis")
        print(f"    Try: SH-axis tweak (room hypothesis) or geometry-axis")
        print(f"    Rationale: failure not at HF — freq-domain intervention won't help")
    elif dominant == "NEAR_CEILING":
        print(f"  → Recommend: ACCEPT current ceiling on freq-axis")
        print(f"    Pivot: visibility-prune / iter-budget / architecture-axis")
        print(f"    Rationale: A3 already matches GT spectrum on majority scenes")
    else:
        print(f"  → Recommend: per-scene breakdown — no clean global pattern")
        print(f"    Inspect per-scene table above; scenes with different patterns")
        print(f"    may need different interventions (scene-conditional)")

    print()
    print(f"  Note: 8-scene single-seed (seed {SEED}). Confirm pattern stable")
    print(f"        with seed 137 + 9999 BEFORE committing to mechanism choice.")


if __name__ == "__main__":
    main()
