#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 13.2] λ_HF calibration script.
# File: scripts/p13_2_lambda_calibration.py  (TẠO MỚI)
# Mục đích: Đo ACTUAL magnitude của L_main (= L1 pixel) và L_HF
#           (= L1 trên Laplacian filter) trên Phase 13 A3 rendered
#           images đã có sẵn. Output: 3 giá trị λ_HF calibrated
#           để pilot không phải dùng heuristic số ngẫu nhiên.
# Pre-condition: render.py đã chạy cho A3 × 8 scenes (đã có data
#                từ Phase 13.2.0' diagnostic).
# Output: 3 λ_HF (gentle / moderate / strong) thay heuristic
#         {0.05, 0.10, 0.20} bằng số data-driven.
# Compute: pure CPU, ~3 phút cho 8 scene × ~5 test view trung bình.
# ============================================================
"""[CRSGaussian Phase 13.2] λ_HF calibration before pilot ablation.

Vấn đề:
  Pilot HF-emphasis loss cần choose λ_HF. Heuristic {0.05, 0.10, 0.20}
  của mình KHÔNG calibrated — có thể off 1 order of magnitude vs
  actual Laplacian L1 magnitude.

Giải pháp:
  Đo trực tiếp trên Phase 13 A3 test renders đã có:
    L_main = mean |render - gt|                          (over pixels)
    L_HF   = mean |Laplacian(render) - Laplacian(gt)|    (over pixels, per channel)
    R      = mean(L_main) / mean(L_HF)                   (across 8 scenes)

  Recommend 3 calibrated λ_HF values:
    λ_gentle   = 0.10 × R   → L_HF contributes ~10% main loss weight
    λ_moderate = 0.30 × R   → ~30% (matches λ_depth=0.05 convention scale)
    λ_strong   = 1.00 × R   → equal magnitude to main loss

  Plus per-scene cross-check vs HF deficit pattern (orchids/leaves
  should have highest L_HF if diagnostic finding is consistent).

Early-abort signal:
  Nếu mean L_HF < 0.001 → HF mismatch magnitude quá nhỏ, mechanism
                          unlikely move PSNR → consider abandon trước pilot.
  Nếu R > 100           → λ_strong > 100 unphysical, normal training
                          không support weight này → mechanism nên skip.

Run:
    python scripts/p13_2_lambda_calibration.py
    CONFIGS="A3" python scripts/p13_2_lambda_calibration.py
    OUTPUT_ROOT=output/p13_lfcf SEED=42 python scripts/p13_2_lambda_calibration.py
"""

import os
import sys
import statistics
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ── Config ──
SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex"
).split()
CONFIGS = os.environ.get("CONFIGS", "A3").split()
SEED = os.environ.get("SEED", "42")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "output/p13_lfcf")
ITERATION = int(os.environ.get("ITERATION", "10000"))

# 3 mức contribution % main loss → 3 calibrated λ_HF
CONTRIBUTION_LEVELS = [
    ("gentle",   0.10),   # 10% main loss magnitude
    ("moderate", 0.30),   # 30%
    ("strong",   1.00),   # equal magnitude
]

# Early-abort thresholds
THRESH_L_HF_TOO_SMALL = 0.001    # L_HF below this → mechanism likely useless
THRESH_RATIO_TOO_LARGE = 100.0   # ratio above → λ_strong unphysical


# ── Laplacian filter (SAME kernel as planned hf_emphasis_loss) ──
# 3×3 isotropic Laplacian — rotation-invariant 2nd-derivative
# Critical: phải dùng KERNEL GIỐNG NHAU với loss thực tế để calibration valid.
_LAPLACIAN_KERNEL = torch.tensor(
    [[0.,  1.,  0.],
     [1., -4.,  1.],
     [0.,  1.,  0.]], dtype=torch.float32
).view(1, 1, 3, 3)


def load_image_as_rgb_tensor(img_path):
    """Load RGB PNG → torch tensor [3, H, W] float ∈ [0, 1]."""
    img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
    # [H, W, 3] → [3, H, W]
    t = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()
    return t


def compute_main_l1(render, gt):
    """L1 mean pixel error (mirrors loss_photometric L1 term)."""
    return float(torch.abs(render - gt).mean().item())


def compute_hf_l1(render, gt):
    """L1 mean on Laplacian filter response. Per-channel filter, isotropic kernel.

    Critical: matches exactly the planned compute_hf_emphasis_loss formula
    so calibrated λ values transfer 1:1 to production loss.
    """
    K = _LAPLACIAN_KERNEL.expand(3, 1, 3, 3)   # per-channel filter
    # [3, H, W] → [1, 3, H, W]
    r_hf = F.conv2d(render.unsqueeze(0), K, padding=1, groups=3)
    g_hf = F.conv2d(gt.unsqueeze(0),     K, padding=1, groups=3)
    return float(torch.abs(r_hf - g_hf).mean().item())


def load_view_pair(render_dir, gt_dir):
    """Load matching pairs of render/GT PNG. Returns list of (render, gt) tensor pairs.

    Sort by filename so 00000.png pairs with 00000.png.
    """
    r_paths = sorted(glob(os.path.join(render_dir, "*.png")))
    g_paths = sorted(glob(os.path.join(gt_dir,     "*.png")))
    if not r_paths or not g_paths:
        return None
    if len(r_paths) != len(g_paths):
        print(f"  [WARN] count mismatch render={len(r_paths)} gt={len(g_paths)}")
        return None
    return [(load_image_as_rgb_tensor(rp), load_image_as_rgb_tensor(gp))
            for rp, gp in zip(r_paths, g_paths)]


def analyze_scene(scene, cfg):
    """Per-scene L_main + L_HF aggregated across all test views."""
    render_dir = f"{OUTPUT_ROOT}/{cfg}_seed{SEED}_{scene}/test/ours_{ITERATION}/renders"
    gt_dir     = f"{OUTPUT_ROOT}/{cfg}_seed{SEED}_{scene}/test/ours_{ITERATION}/gt"
    pairs = load_view_pair(render_dir, gt_dir)
    if pairs is None:
        return None
    l_mains = [compute_main_l1(r, g) for r, g in pairs]
    l_hfs   = [compute_hf_l1(r, g)   for r, g in pairs]
    return {
        'scene': scene,
        'n_views': len(pairs),
        'L_main_mean': statistics.fmean(l_mains),
        'L_HF_mean':   statistics.fmean(l_hfs),
        'ratio':       statistics.fmean(l_mains) / max(statistics.fmean(l_hfs), 1e-12),
    }


def main():
    print("=== Phase 13.2 λ_HF Calibration ===")
    print(f"OUTPUT_ROOT={OUTPUT_ROOT}  SEED={SEED}  ITERATION={ITERATION}")
    print(f"CONFIGS={CONFIGS}  SCENES={SCENES}")
    print(f"Levels: " + ", ".join(f"{n}={p:.0%}" for n, p in CONTRIBUTION_LEVELS))
    print()

    # Pre-flight: first scene + first config must have renders
    cfg0 = CONFIGS[0]
    first_dir = f"{OUTPUT_ROOT}/{cfg0}_seed{SEED}_{SCENES[0]}/test/ours_{ITERATION}"
    if not os.path.isdir(first_dir):
        print(f"[FATAL] Missing: {first_dir}")
        print("        Run render.py first — see Phase 13.2.0' diagnostic prompt.")
        sys.exit(1)

    for cfg in CONFIGS:
        print(f"\n──── Config: {cfg} ────")
        results = []
        for sc in SCENES:
            r = analyze_scene(sc, cfg)
            if r is None:
                print(f"  {sc}: SKIP (no renders or count mismatch)")
                continue
            results.append(r)

        if not results:
            print(f"  [FATAL] No scenes analyzed for {cfg}.")
            continue

        # ── Per-scene table ──
        print(f"\n  Per-scene magnitudes:")
        print(f"  {'scene':<10}  {'N':>3}  {'L_main':>10}  {'L_HF':>10}  {'ratio':>8}")
        print("  " + "-" * 50)
        for r in results:
            print(f"  {r['scene']:<10}  {r['n_views']:>3}  "
                  f"{r['L_main_mean']:>10.5f}  {r['L_HF_mean']:>10.5f}  "
                  f"{r['ratio']:>8.3f}")

        # ── Aggregate ──
        L_main_mean = statistics.fmean(r['L_main_mean'] for r in results)
        L_HF_mean   = statistics.fmean(r['L_HF_mean']   for r in results)
        R = L_main_mean / max(L_HF_mean, 1e-12)
        print(f"\n  Aggregate {len(results)} scenes:")
        print(f"    L_main  mean = {L_main_mean:.5f}")
        print(f"    L_HF    mean = {L_HF_mean:.5f}")
        print(f"    Ratio R      = L_main / L_HF = {R:.3f}")

        # ── λ recommendations ──
        print(f"\n  Calibrated λ_HF values  (so that λ × L_HF contributes X% of L_main):")
        lambda_recs = {}
        for name, pct in CONTRIBUTION_LEVELS:
            lam = pct * R
            lambda_recs[name] = lam
            print(f"    λ_{name:<9}  = {pct:.0%} × R = {lam:>9.4f}")

        # ── Cross-check vs HF-deficit pattern ──
        # Per memory project_a3_mechanism_spatial_not_spectral (Phase 13.2.0'):
        # orchids/leaves have largest HF rel_Δ deficit (rendered ~56% GT HF energy).
        # Expect they also have largest L_HF here — sanity check that diagnostic
        # finding is consistent with this calibration measurement.
        print(f"\n  Cross-check — scenes ranked by L_HF (largest = most HF mismatch):")
        for r in sorted(results, key=lambda x: -x['L_HF_mean']):
            print(f"    {r['scene']:<10}  L_HF = {r['L_HF_mean']:.5f}")
        print(f"  Expect: orchids, leaves, flower near top (HF-rich + largest deficit)")
        print(f"          fortress, room, fern near bottom (LF-dominant scenes)")

        # ── Early-abort signals ──
        print(f"\n  Pre-pilot signals:")
        if L_HF_mean < THRESH_L_HF_TOO_SMALL:
            print(f"    ⚠️  L_HF mean = {L_HF_mean:.5f} < {THRESH_L_HF_TOO_SMALL}")
            print(f"        HF mismatch magnitude rất nhỏ — mechanism unlikely move PSNR")
            print(f"        → Consider ABANDON HF direction trước khi pilot")
        elif R > THRESH_RATIO_TOO_LARGE:
            print(f"    ⚠️  Ratio R = {R:.1f} > {THRESH_RATIO_TOO_LARGE}")
            print(f"        λ_strong = {lambda_recs['strong']:.1f} unphysical")
            print(f"        → Reconsider mechanism choice or use only gentle level")
        else:
            print(f"    ✓ Calibration sane (L_HF moderate, R reasonable)")
            print(f"      Proceed to pilot với 3 λ values above on 3 scenes "
                  f"(trex/horns/orchids)")

        # ── Pilot script template ──
        print(f"\n  Suggested pilot command (3 scene × 3 λ = 9 runs):")
        print(f"    LAMBDA_GENTLE={lambda_recs['gentle']:.4f} \\")
        print(f"    LAMBDA_MODERATE={lambda_recs['moderate']:.4f} \\")
        print(f"    LAMBDA_STRONG={lambda_recs['strong']:.4f} \\")
        print(f"    bash scripts/p13_2_hf_pilot.sh    # (to be created)")


if __name__ == "__main__":
    main()
