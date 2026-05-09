#!/usr/bin/env python3
"""[CRSGaussian Phase 10A] Dense init analyzer.

Configs:
  AUGMENT  (Phase 8 FULL backbone + DUSt3R points augmented với COLMAP)
  REPLACE  (Phase 8 FULL backbone + DUSt3R points only)

Reference (reuse logs/p8/FULL_*.log): Phase 8 FULL = 21.335 dB.

Attribution:
  Δ_AUG = AUGMENT - P8_FULL  (effect of adding DUSt3R points)
  Δ_REP = REPLACE - P8_FULL  (effect of replacing COLMAP với DUSt3R)
  Δ_AR  = AUGMENT - REPLACE  (precision value của COLMAP keypoints)

Verdict tree:
  Δ_AUG ≥ +0.85 → BREAK SOTA       (vượt ICO-GS 22.20 dB)
  Δ_AUG ≥ +0.30 → CLOSE SOTA       (significant gap closure)
  Δ_AUG ≥ +0.10 → MODEST IMPROVE   (positive but small)
  Δ_AUG <  +0.10 → FAILED          (no/negative effect)

Run:
  python scripts/p10a_analyze.py
  LOG_DIR=logs/p10a P8_LOG_DIR=logs/p8 python scripts/p10a_analyze.py
"""

import os
import re
import statistics

SCENES = ['fern', 'flower', 'fortress', 'horns',
          'leaves', 'orchids', 'room', 'trex']

CONFIGS = ['P8_FULL', 'AUGMENT', 'REPLACE']

LOG_DIR    = os.environ.get("LOG_DIR",    "logs/p10a")
P8_LOG_DIR = os.environ.get("P8_LOG_DIR", "logs/p8")

# Reference targets
PHASE_8_FULL_REF = 21.335
NO_CRS_REF       = 21.210
DOC_GS_REF       = 21.380
BINOCULAR_REF    = 21.440
ICO_GS_SOTA      = 22.200

# Verdict thresholds (vs P8_FULL)
THR_BREAK_SOTA   = ICO_GS_SOTA - PHASE_8_FULL_REF   # ≈ +0.865
THR_CLOSE_SOTA   = 0.30
THR_MODEST       = 0.10

PSNR_PAT  = re.compile(r"Best test PSNR:\s*([\d\.eE\-\+]+)")
SSIM_PAT  = re.compile(r"\bSSIM[\s:=]+([\d\.eE\-\+]+)")
LPIPS_PAT = re.compile(r"\bLPIPS[\s:=]+([\d\.eE\-\+]+)")
N_GAUSS_PAT = re.compile(r"\bN=(\d+)\b")
INIT_GAUSS_PAT = re.compile(r"\[Phase 10A\]\s+Initial Gaussians:\s*(\d+)")
TIME_PAT = re.compile(r"elapsed=([\d\.]+)s")


def parse_first(pat, text, cast=float):
    m = pat.search(text)
    return cast(m.group(1)) if m else None


def parse_last(pat, text, cast=float):
    matches = pat.findall(text)
    return cast(matches[-1]) if matches else None


def parse_log(path):
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    return {
        'psnr':         parse_first(PSNR_PAT, text),
        'ssim':         parse_last(SSIM_PAT, text),
        'lpips':        parse_last(LPIPS_PAT, text),
        'n_gauss':      parse_last(N_GAUSS_PAT, text, cast=int),
        'init_gauss':   parse_first(INIT_GAUSS_PAT, text, cast=int),
        'train_time_s': parse_last(TIME_PAT, text),
    }


def log_path(cfg, scene):
    """P8_FULL reuse từ logs/p8; AUGMENT/REPLACE từ logs/p10a."""
    if cfg == 'P8_FULL':
        return f"{P8_LOG_DIR}/FULL_{scene}.log"
    return f"{LOG_DIR}/{cfg}_{scene}.log"


def collect():
    table = {cfg: {} for cfg in CONFIGS}
    for cfg in CONFIGS:
        for sc in SCENES:
            res = parse_log(log_path(cfg, sc))
            table[cfg][sc] = res
    return table


def avg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return statistics.fmean(vals)


def fmt(v, prec=3):
    return f"{v:.{prec}f}" if v is not None else "    -"


def per_scene_table(table):
    print("\n=== Per-scene PSNR ===")
    header = "scene".ljust(10) + "".join(c.ljust(10) for c in CONFIGS) + "Δ_AUG    Δ_REP"
    print(header)
    print("-" * len(header))
    for sc in SCENES:
        row = sc.ljust(10)
        psnrs = {c: (table[c][sc] or {}).get('psnr') for c in CONFIGS}
        for c in CONFIGS:
            row += fmt(psnrs[c]).ljust(10)
        ref = psnrs.get('P8_FULL')
        if ref is not None:
            d_aug = (psnrs.get('AUGMENT') - ref) if psnrs.get('AUGMENT') is not None else None
            d_rep = (psnrs.get('REPLACE') - ref) if psnrs.get('REPLACE') is not None else None
            row += f"{fmt(d_aug):>8} {fmt(d_rep):>8}"
        print(row)


def avg_summary(table):
    print("\n=== Avg across 8 scenes ===")
    print("Config      | PSNR    | SSIM   | LPIPS  | InitGauss | FinalGauss | Time(s) | Δ_PSNR | Verdict")
    print("-" * 105)

    avgs = {}
    for cfg in CONFIGS:
        rows = list(table[cfg].values())
        avgs[cfg] = {
            'psnr':       avg(r.get('psnr') for r in rows if r),
            'ssim':       avg(r.get('ssim') for r in rows if r),
            'lpips':      avg(r.get('lpips') for r in rows if r),
            'init_gauss': avg(r.get('init_gauss') for r in rows if r),
            'n_gauss':    avg(r.get('n_gauss') for r in rows if r),
            'time':       avg(r.get('train_time_s') for r in rows if r),
        }

    p8 = avgs['P8_FULL']['psnr']
    for cfg in CONFIGS:
        a = avgs[cfg]
        d = (a['psnr'] - p8) if (a['psnr'] is not None and p8 is not None) else None
        if cfg == 'P8_FULL':
            verdict = '(reference)'
        elif d is None:
            verdict = 'NO DATA'
        elif d >= THR_BREAK_SOTA:
            verdict = '🎉 BREAK SOTA'
        elif d >= THR_CLOSE_SOTA:
            verdict = '✅ CLOSE SOTA'
        elif d >= THR_MODEST:
            verdict = '🟡 MODEST'
        else:
            verdict = '❌ FAILED'

        print(f"{cfg:11} | {fmt(a['psnr'])} | {fmt(a['ssim'])} | {fmt(a['lpips'])} | "
              f"{fmt(a['init_gauss'], 0):>9} | {fmt(a['n_gauss'], 0):>10} | "
              f"{fmt(a['time'], 1):>7} | {fmt(d, 3):>6} | {verdict}")

    return avgs


def attribution(avgs):
    print("\n=== Attribution ===")
    p8 = avgs['P8_FULL']['psnr']
    aug = avgs['AUGMENT']['psnr']
    rep = avgs['REPLACE']['psnr']

    if all(v is not None for v in [p8, aug, rep]):
        d_aug = aug - p8
        d_rep = rep - p8
        d_ar  = aug - rep
        print(f"  Δ_AUG (AUGMENT - P8_FULL)  = {d_aug:+.3f}  — DUSt3R augmentation effect")
        print(f"  Δ_REP (REPLACE - P8_FULL)  = {d_rep:+.3f}  — full DUSt3R replacement effect")
        print(f"  Δ_AR  (AUGMENT - REPLACE)  = {d_ar:+.3f}  — COLMAP keypoint precision value")

        if d_aug >= THR_BREAK_SOTA:
            print(f"\n  🎉 AUGMENT vượt ICO-GS SOTA ({ICO_GS_SOTA:.3f}). Phase 10A SUCCESS.")
        elif d_aug >= THR_CLOSE_SOTA:
            print(f"\n  ✅ AUGMENT đóng gap đáng kể ({d_aug:+.3f} ≥ {THR_CLOSE_SOTA}). Cân nhắc Phase 10B stack.")
        elif d_aug >= THR_MODEST:
            print(f"\n  🟡 AUGMENT improves but modestly ({d_aug:+.3f}). Investigate confidence threshold + dedupe radius.")
        else:
            print(f"\n  ❌ AUGMENT không cải thiện ({d_aug:+.3f}). Hypothesis dense init không phải orthogonal lever cho recipe này.")


def main():
    table = collect()
    per_scene_table(table)
    avgs = avg_summary(table)
    attribution(avgs)

    print("\n=== Reference targets ===")
    print(f"  No-CRS Tier1   = {NO_CRS_REF:.3f}")
    print(f"  Phase 8 FULL   = {PHASE_8_FULL_REF:.3f}  ⭐ project best")
    print(f"  DOC-GS         = {DOC_GS_REF:.3f}")
    print(f"  BinocularGS    = {BINOCULAR_REF:.3f}")
    print(f"  ICO-GS (SOTA)  = {ICO_GS_SOTA:.3f}  ← break this")


if __name__ == "__main__":
    main()
