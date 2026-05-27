#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 23] Cross-backbone ablation analyzer on RoMa v1
# File: scripts/p23_v1_ablation_analyze.py  (KEEP LOCAL — server-only)
#
# Đọc 12 configs P23 + reuse trim_full từ Phase 22 → 2 bảng:
#   1. LOO Δ-vs-trim_full (mất gì khi bỏ X)
#   2. Single-add Δ-vs-base (thêm gì khi cho riêng X)
#
# Cross-reference Phase 20 (MVS) + Phase 20b (dense PDCNet+) để xác định
# cross-backbone-stable contribution.
#
# Usage: python scripts/p23_v1_ablation_analyze.py
# ============================================================
"""Phase 23 cross-backbone ablation analyzer — LOO + single-add."""

import os
import re
import math
from pathlib import Path

import numpy as np

SCENES = "fern flower fortress horns leaves orchids room trex".split()
SEED = "42"  # 1-seed pilot
ABL_ROOT = Path(os.environ.get("ABL_ROOT", "logs/p23_v1_ablation"))
TRIM_FULL_DIR = Path(os.environ.get("TRIM_FULL_DIR", "logs/p22_pilot"))  # reuse Phase 22

# Configs in priority order
LOO_CONFIGS = [
    ("trim_no_efa",       "EFA (LFCF+AbsGS)"),
    ("trim_no_drop",      "DropAnSH"),
    ("trim_no_opacity",   "Opacity decay"),
    ("trim_no_dcycle",    "D_cycle"),
    ("trim_no_shcrs",     "SH-CRS + SH-REL"),
    ("trim_no_depthcrs",  "depth+CRS cascade"),
]
SINGLE_CONFIGS = [
    ("single_efa",        "EFA alone"),
    ("single_drop",       "DropAnSH alone"),
    ("single_opacity",    "Opacity decay alone"),
    ("single_dcycle",     "depth + D_cycle"),
    ("single_shcrs",      "depth + SH-CRS"),
    ("single_depthcrs",   "full CRS cascade"),
]

# Phase 20 / 20b reference (MVS + dense PDCNet+) — pre-computed from memory
PHASE20_MVS_LOO = {
    "trim_no_efa":       -0.12,
    "trim_no_drop":      -0.56,
    "trim_no_opacity":   -0.16,
    "trim_no_dcycle":    -0.15,
    "trim_no_shcrs":     -0.30,
    "trim_no_depthcrs":  -0.86,
}
PHASE20B_DENSE_LOO = {
    "trim_no_efa":       -0.03,
    "trim_no_drop":      -0.64,
    "trim_no_opacity":   -0.24,
    "trim_no_dcycle":    -0.01,
    "trim_no_shcrs":     -0.04,
    "trim_no_depthcrs":  -0.62,
}

PSNR_PAT = re.compile(r"Best test PSNR:\s*([\d.eE+\-]+)")
ROW_PAT = re.compile(
    r"\|\s*test\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"[^|]*\|\s*(\d+)")


def parse_log(path: Path):
    if not path.is_file():
        return None
    txt = path.read_text(encoding="utf-8", errors="ignore")
    rows = ROW_PAT.findall(txt)
    if rows:
        psnr, ssim, lpips, n = rows[-1]
        return float(psnr), float(ssim), float(lpips), int(n)
    mp = PSNR_PAT.search(txt)
    return (float(mp.group(1)), None, None, None) if mp else None


def collect_config(config_dir: Path):
    """Return per-scene (psnr, n_gauss) for seed42."""
    out = {}
    for sc in SCENES:
        f = config_dir / f"A3_seed{SEED}_{sc}.log"
        v = parse_log(f)
        if v is not None:
            out[sc] = (v[0], v[3])
    return out


def collect_trim_full_phase22():
    """Reuse Phase 22 pilot data, 3-seed mean per scene → return same format."""
    out = {}
    for sc in SCENES:
        psnrs, ns = [], []
        for sd in ["42", "137", "9999"]:
            f = TRIM_FULL_DIR / f"A3_seed{sd}_{sc}.log"
            v = parse_log(f)
            if v is not None:
                psnrs.append(v[0]); ns.append(v[3])
        if psnrs:
            out[sc] = (float(np.mean(psnrs)), int(np.mean(ns)))
    return out


def fmt(v, w=8, prec=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return f"{'?':>{w}}"
    return f"{v:{w}.{prec}f}"


def main():
    print("=" * 100)
    print(f"[P23] ABL_ROOT={ABL_ROOT}  TRIM_FULL_DIR={TRIM_FULL_DIR} (reuse Phase 22)")
    print(f"[P23] backbone = RoMa v1 init  | seed=42 (1-seed pilot, 8 scenes)")
    print("=" * 100)

    # Collect all configs
    base = collect_config(ABL_ROOT / "base")
    trim_full = collect_trim_full_phase22()  # 3-seed mean

    loo_data = {cfg: collect_config(ABL_ROOT / cfg) for cfg, _ in LOO_CONFIGS}
    single_data = {cfg: collect_config(ABL_ROOT / cfg) for cfg, _ in SINGLE_CONFIGS}

    # Counts
    print(f"  found: base={len(base)}/8  trim_full(Phase22, 3-seed)={len(trim_full)}/8")
    for cfg, _ in LOO_CONFIGS:
        n = len(loo_data[cfg])
        mark = "✓" if n == 8 else f"⚠ {n}/8"
        print(f"         {cfg:20s} = {mark}")
    for cfg, _ in SINGLE_CONFIGS:
        n = len(single_data[cfg])
        mark = "✓" if n == 8 else f"⚠ {n}/8"
        print(f"         {cfg:20s} = {mark}")
    print()

    # ─── Table 1: LOO Δ-vs-trim_full ───
    print("─" * 100)
    print(f"  TABLE 1 — LOO ablation (Δ vs trim_full, paired per scene)")
    print(f"  Negative Δ = module CONTRIBUTES (its absence hurts PSNR)")
    print("─" * 100)
    print(f"  {'Module bucket':25s}  {'Δ_v1 (P23)':>12s}  {'Δ_MVS (P20)':>12s}  {'Δ_dense (P20b)':>14s}  {'cross-stable?':>14s}")
    print("─" * 100)
    for cfg, label in LOO_CONFIGS:
        d = loo_data[cfg]
        if not d:
            print(f"  {label:25s}  {'?':>12s}  {PHASE20_MVS_LOO[cfg]:>+12.3f}  {PHASE20B_DENSE_LOO[cfg]:>+14.3f}")
            continue
        # Paired Δ: per-scene (cfg − trim_full)
        deltas = []
        for sc in SCENES:
            if sc in d and sc in trim_full:
                deltas.append(d[sc][0] - trim_full[sc][0])
        d_v1 = float(np.mean(deltas)) if deltas else float("nan")
        d_mvs = PHASE20_MVS_LOO[cfg]
        d_dense = PHASE20B_DENSE_LOO[cfg]

        # Classify cross-stable: all 3 negative significant (< -0.10)
        all_neg = all(x < -0.10 for x in [d_v1, d_mvs, d_dense] if not math.isnan(x))
        all_wash = all(abs(x) < 0.05 for x in [d_v1, d_mvs, d_dense] if not math.isnan(x))
        if all_neg:
            tag = "✓ STABLE"
        elif all_wash:
            tag = "≈ DEAD all"
        elif abs(d_mvs) > 0.10 and abs(d_v1) < 0.05:
            tag = "MVS-only"
        else:
            tag = "mixed"

        print(f"  {label:25s}  {d_v1:>+12.3f}  {d_mvs:>+12.3f}  {d_dense:>+14.3f}  {tag:>14s}")
    print("─" * 100)
    print()

    # ─── Table 2: Single-add Δ-vs-base ───
    print("─" * 100)
    print(f"  TABLE 2 — Single-add (Δ vs base, paired per scene)")
    print(f"  Positive Δ = module helps in isolation (synergy-excluded)")
    print("─" * 100)
    print(f"  {'Module bucket':25s}  {'Δ_v1 (P23)':>12s}  {'(no Phase 20 single)':>22s}")
    print("─" * 100)
    for cfg, label in SINGLE_CONFIGS:
        d = single_data[cfg]
        if not d or not base:
            print(f"  {label:25s}  {'?':>12s}  (missing base or config)")
            continue
        deltas = []
        for sc in SCENES:
            if sc in d and sc in base:
                deltas.append(d[sc][0] - base[sc][0])
        d_v1 = float(np.mean(deltas)) if deltas else float("nan")
        mark = "⭐" if d_v1 > 0.3 else ("✓" if d_v1 > 0.1 else ("~" if d_v1 > -0.1 else "✗"))
        print(f"  {label:25s}  {d_v1:>+12.3f}  {mark}")
    print("─" * 100)
    print()

    # ─── Table 3: Reference levels ───
    print("─" * 80)
    print(f"  TABLE 3 — Reference PSNR levels (v1 backbone)")
    print("─" * 80)
    if base:
        base_psnrs = [base[sc][0] for sc in SCENES if sc in base]
        if base_psnrs:
            print(f"  base (v1 init, 0 module)         : {np.mean(base_psnrs):.3f}  ({len(base_psnrs)}/8)")
    if trim_full:
        tf_psnrs = [trim_full[sc][0] for sc in SCENES if sc in trim_full]
        if tf_psnrs:
            print(f"  trim_full (Phase 22 N=24)        : {np.mean(tf_psnrs):.3f}  ({len(tf_psnrs)}/8)")
    print(f"  reference: Phase 22 v1 8-scene avg  : 21.918")
    print(f"  reference: Phase 20 MVS 8-scene avg : 21.330  (TRIM-locked)")
    print(f"  reference: 3DGS pure MVS (Phase 20) : 20.83")
    print("─" * 80)

    # ─── Contribution framing suggestion ───
    print()
    print("=" * 80)
    print("CONTRIBUTION FRAMING (auto-suggest based on LOO Δ_v1):")
    print("=" * 80)
    cross_stable = []
    backbone_dep = []
    dead_all = []
    for cfg, label in LOO_CONFIGS:
        d = loo_data[cfg]
        if not d: continue
        deltas = [d[sc][0] - trim_full[sc][0] for sc in SCENES if sc in d and sc in trim_full]
        if not deltas: continue
        d_v1 = float(np.mean(deltas))
        d_mvs = PHASE20_MVS_LOO[cfg]
        d_dense = PHASE20B_DENSE_LOO[cfg]
        if all(x < -0.10 for x in [d_v1, d_mvs, d_dense]):
            cross_stable.append(label)
        elif d_mvs < -0.10 and d_v1 > -0.05:
            backbone_dep.append(label)
        elif all(abs(x) < 0.05 for x in [d_v1, d_mvs, d_dense]):
            dead_all.append(label)

    print(f"  ⭐ Core (cross-backbone-stable, contribute on ALL 3 backbones):")
    for x in cross_stable: print(f"      - {x}")
    if not cross_stable: print(f"      (none yet — wait for full data)")
    print(f"  ⚠ Backbone-dependent (MVS-only, wash on dense+v1):")
    for x in backbone_dep: print(f"      - {x}")
    if not backbone_dep: print(f"      (none)")
    print(f"  ✗ Dead all backbones:")
    for x in dead_all: print(f"      - {x}")
    if not dead_all: print(f"      (none)")


if __name__ == "__main__":
    main()
