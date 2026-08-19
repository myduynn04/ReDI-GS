#!/usr/bin/env python3
"""Aggregate the per-run logs produced by run.sh into PSNR / SSIM / LPIPS tables.

The script scans the log directory produced by run.sh (default ``logs/run``)
for files named ``A3_seed{SEED}_{SCENE}.log``, extracts the final test
metrics, and prints the per-scene mean across seeds together with the
overall N=24 paired mean (8 scenes x 3 seeds).
"""

import os
import re
import math
from pathlib import Path

import numpy as np

SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
SEEDS = os.environ.get("SEEDS", "42 137 9999").split()
LOG_DIR = Path(os.environ.get("LOG_DIR", "logs/run"))
# [CRSGaussian Phase 26] Tên file log không phải lúc nào cũng "A3_..." —
# scripts/p26_ablation_run.sh đặt tên theo ${CONFIG}_seed... Cho phép
# override qua PREFIX, default "A3" giữ nguyên hành vi cũ (scripts/run.sh).
PREFIX = os.environ.get("PREFIX", "A3")

PSNR_PAT = re.compile(r"Best test PSNR:\s*([\d.eE+\-]+)")
ROW_PAT = re.compile(
    r"\|\s*test\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"[^|]*\|\s*(\d+)")


def parse_log(path: Path):
    """Return ``(psnr, ssim, lpips, n_gauss)`` or ``None`` if no data."""
    if not path.is_file():
        return None
    txt = path.read_text(encoding="utf-8", errors="ignore")
    rows = ROW_PAT.findall(txt)
    if rows:
        psnr, ssim, lpips, n = rows[-1]
        return float(psnr), float(ssim), float(lpips), int(n)
    mp = PSNR_PAT.search(txt)
    return (float(mp.group(1)), None, None, None) if mp else None


def collect(root: Path):
    """Read every ``{PREFIX}_seed{SEED}_{SCENE}.log`` under ``root``."""
    out = {}
    for sc in SCENES:
        for sd in SEEDS:
            f = root / f"{PREFIX}_seed{sd}_{sc}.log"
            v = parse_log(f)
            if v is not None:
                out[(sc, sd)] = v
    return out


def fmt(v, w=8, prec=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return f"{'?':>{w}}"
    return f"{v:{w}.{prec}f}"


def main():
    print("=" * 88)
    print(f"  ReDI-GS results aggregator")
    print(f"  LOG_DIR = {LOG_DIR}")
    print(f"  SCENES  = {SCENES}")
    print(f"  SEEDS   = {SEEDS}")
    print("=" * 88)

    results = collect(LOG_DIR)
    expected = len(SCENES) * len(SEEDS)
    print(f"  Found {len(results)} / {expected} log files")
    print()

    if not results:
        print("  No logs to analyse — run scripts/run.sh first.")
        return

    # Per-scene means across seeds.
    print("-" * 88)
    print(f"  Per-scene means (averaged over seeds present)")
    print("-" * 88)
    print(f"  {'Scene':10s}  {'PSNR':>8s}  {'SSIM':>8s}  {'LPIPS':>8s}  "
          f"{'N_gauss':>10s}  {'n_seeds':>8s}")
    print("-" * 88)

    per_scene_psnr = []
    per_scene_ssim = []
    per_scene_lpips = []
    per_scene_n = []

    for sc in SCENES:
        psnrs, ssims, lpipss, ns = [], [], [], []
        for sd in SEEDS:
            if (sc, sd) in results:
                p, s, l, n = results[(sc, sd)]
                psnrs.append(p)
                if s is not None:
                    ssims.append(s)
                if l is not None:
                    lpipss.append(l)
                if n is not None:
                    ns.append(n)

        if psnrs:
            mp = float(np.mean(psnrs))
            ms = float(np.mean(ssims)) if ssims else None
            ml = float(np.mean(lpipss)) if lpipss else None
            mn = int(np.mean(ns)) if ns else None
            per_scene_psnr.append(mp)
            if ms is not None:
                per_scene_ssim.append(ms)
            if ml is not None:
                per_scene_lpips.append(ml)
            if mn is not None:
                per_scene_n.append(mn)
            print(f"  {sc:10s}  {fmt(mp)}  {fmt(ms)}  {fmt(ml)}  "
                  f"{(str(mn) if mn is not None else '?'):>10s}  "
                  f"{len(psnrs):>8d}")
        else:
            print(f"  {sc:10s}  (no data)")

    print("-" * 88)

    # Overall N=24 mean.
    if per_scene_psnr:
        print()
        print("=" * 88)
        print(f"  Final aggregate over {len(results)} runs:")
        print(f"     PSNR  = {np.mean(per_scene_psnr):.3f}")
        if per_scene_ssim:
            print(f"     SSIM  = {np.mean(per_scene_ssim):.3f}")
        if per_scene_lpips:
            print(f"     LPIPS = {np.mean(per_scene_lpips):.3f}")
        if per_scene_n:
            print(f"     N_gauss (avg per scene) = {int(np.mean(per_scene_n))}")
        print("=" * 88)


if __name__ == "__main__":
    main()
