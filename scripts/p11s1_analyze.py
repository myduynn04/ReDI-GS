#!/usr/bin/env python3
"""[CRSGaussian Phase 11 Step 1] Covisibility reweight analyzer.

Configs:
  A0: Phase 8 FULL ref (no covisibility)
  A1: + covisibility reweight (combined với CRS_pix, mode=min)

Decision tree (Δ_A1 vs A0, AVG cross scenes):
  ≥ +0.20 → 🎉 CONFIRM (proceed to confirm 2 scenes → scale 8)
  +0.05..+0.20 → 🟡 MARGINAL (parallel test Step 2)
  < +0.05 → ❌ REJECT (đi Step 2)

Run:
  python scripts/p11s1_analyze.py
  SCENES="orchids leaves horns" python scripts/p11s1_analyze.py
  LOG_DIR=logs/p11s1 python scripts/p11s1_analyze.py
"""

import os
import re
import statistics

# Default 1 scene (diagnostic). Override via SCENES env để scale.
SCENES = os.environ.get("SCENES", "orchids").split()
CONFIGS = ['A0', 'A1']
LOG_DIR = os.environ.get("LOG_DIR", "logs/p11s1")

# Decision thresholds
THR_CONFIRM = 0.20
THR_MARGINAL = 0.05

# Phase 8 FULL per-scene reference (orchids = 16.761 in spec; others
# từ logs/p8 nếu user re-run 2/8 scenes).
P8_FULL_REF_PER_SCENE = {
    'orchids': 16.761,
}
P8_FULL_REF_AVG = 21.335

PSNR_PAT  = re.compile(r"Best test PSNR:\s*([\d\.eE\-\+]+)")
SSIM_PAT  = re.compile(r"\bSSIM[\s:=]+([\d\.eE\-\+]+)")
LPIPS_PAT = re.compile(r"\bLPIPS[\s:=]+([\d\.eE\-\+]+)")
N_GAUSS_PAT = re.compile(r"\bN=(\d+)\b")
INIT_GAUSS_PAT = re.compile(r"Initial Gaussians:\s*(\d+)")
TIME_PAT = re.compile(r"elapsed=([\d\.]+)s")
COV_PAT = re.compile(r"covisible_pixel_frac=([\d\.]+)")
WEIGHT_PAT = re.compile(r"phase11s1/mean_weight\s*=?\s*([\d\.]+)")


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
        'cov_frac':     parse_first(COV_PAT, text),
    }


def avg(values):
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def fmt(v, prec=3):
    return f"{v:.{prec}f}" if v is not None else "    -"


def per_scene_table(table):
    print(f"\n=== Per-scene PSNR  (scenes: {', '.join(SCENES)}) ===")
    header = "scene".ljust(12) + "A0".ljust(10) + "A1".ljust(10) + "Δ_A1     P8_ref"
    print(header)
    print("-" * len(header))
    for sc in SCENES:
        row = sc.ljust(12)
        a0 = (table['A0'].get(sc) or {}).get('psnr')
        a1 = (table['A1'].get(sc) or {}).get('psnr')
        ref = P8_FULL_REF_PER_SCENE.get(sc)
        d = (a1 - a0) if (a0 is not None and a1 is not None) else None
        row += fmt(a0).ljust(10) + fmt(a1).ljust(10) + f"{fmt(d):>7}  {fmt(ref)}"
        print(row)


def avg_summary(table):
    print("\n=== Avg summary ===")
    print("Cfg | PSNR    | SSIM   | LPIPS  | InitG  | FinalG  | Time(s) | CovFrac | Δ_PSNR | Verdict")
    print("-" * 100)

    avgs = {}
    for cfg in CONFIGS:
        rows = [table[cfg].get(sc) for sc in SCENES if table[cfg].get(sc)]
        avgs[cfg] = {
            'psnr':       avg(r.get('psnr')         for r in rows),
            'ssim':       avg(r.get('ssim')         for r in rows),
            'lpips':      avg(r.get('lpips')        for r in rows),
            'init_gauss': avg(r.get('init_gauss')   for r in rows),
            'n_gauss':    avg(r.get('n_gauss')      for r in rows),
            'time':       avg(r.get('train_time_s') for r in rows),
            'cov_frac':   avg(r.get('cov_frac')     for r in rows),
        }

    a0_psnr = avgs['A0']['psnr']
    for cfg in CONFIGS:
        a = avgs[cfg]
        d = (a['psnr'] - a0_psnr) if (a['psnr'] is not None and a0_psnr is not None) else None
        if cfg == 'A0':
            verdict = '(reference)'
        elif d is None:
            verdict = 'NO DATA'
        elif d >= THR_CONFIRM:
            verdict = '🎉 CONFIRM (≥ +0.20)'
        elif d >= THR_MARGINAL:
            verdict = '🟡 MARGINAL'
        else:
            verdict = '❌ REJECT (< +0.05)'

        print(f"{cfg:3} | {fmt(a['psnr'])} | {fmt(a['ssim'])} | {fmt(a['lpips'])} | "
              f"{fmt(a['init_gauss'], 0):>6} | {fmt(a['n_gauss'], 0):>7} | "
              f"{fmt(a['time'], 1):>7} | {fmt(a['cov_frac'], 3):>7} | "
              f"{fmt(d, 3):>6} | {verdict}")

    return avgs


def decision(avgs):
    print("\n=== Decision ===")
    a0 = avgs['A0']['psnr']
    a1 = avgs['A1']['psnr']
    if a0 is None or a1 is None:
        print("  NO DATA — cần ít nhất A0 + A1 logs valid")
        return
    d = a1 - a0
    print(f"  Δ_A1 (A1 - A0) = {d:+.3f} dB  (avg over {len(SCENES)} scene(s))")

    if d >= THR_CONFIRM:
        print(f"\n  🎉 Δ ≥ {THR_CONFIRM} → CONFIRM Step 1.")
        if len(SCENES) == 1:
            print("     Next: confirm leaves + horns. Run:")
            print('       SCENES_OVERRIDE="leaves horns" bash scripts/p11s1_diagnostic.sh  # nếu hỗ trợ multi-scene')
            print("       hoặc chạy thủ công 2 lần với SCENE=leaves và SCENE=horns")
            print("     Sau đó: SCENES='orchids leaves horns' python scripts/p11s1_analyze.py")
        elif len(SCENES) <= 3:
            print("     Next: scale 8 scenes (master script)")
        else:
            print("     Step 1 ACCEPTED. Lock recipe.")
    elif d >= THR_MARGINAL:
        print(f"\n  🟡 Δ ∈ [{THR_MARGINAL}, {THR_CONFIRM}) → MARGINAL.")
        print("     Strategy: parallel test Step 2 (perceptual).")
    else:
        print(f"\n  ❌ Δ < {THR_MARGINAL} → REJECT Step 1.")
        print("     Next: đi Step 2 (perceptual loss).")


def main():
    print(f"LOG_DIR={LOG_DIR}  SCENES={SCENES}")
    table = {cfg: {sc: parse_log(f"{LOG_DIR}/{cfg}_{sc}.log") for sc in SCENES}
             for cfg in CONFIGS}

    per_scene_table(table)
    avgs = avg_summary(table)
    decision(avgs)


if __name__ == "__main__":
    main()
