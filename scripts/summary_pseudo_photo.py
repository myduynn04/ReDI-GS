#!/usr/bin/env python3
# ============================================================
# [CRSGaussian] Summary Pseudo Photo Ablation
# File: scripts/summary_pseudo_photo.py
# Mục đích: Đọc logs/ablation_pseudo_photo/*.log → in bảng PSNR + delta
#           Bypass awk/bash arithmetic issues.
#
# Usage: python scripts/summary_pseudo_photo.py
# ============================================================

import os
import re
import sys
from glob import glob

LOGDIR = "logs/ablation_pseudo_photo"
CONFIGS = ["B0", "AP2_05", "AP2_10", "AP2_20"]
SCENES = ["fern", "flower", "fortress", "horns",
          "leaves", "orchids", "room", "trex"]

PSNR_RE = re.compile(r"\[ITER 10000\] Evaluating test:.*PSNR ([0-9.]+)")
SSIM_RE = re.compile(r"\[ITER 10000\] Evaluating test:.*SSIM ([0-9.]+)")
LPIPS_RE = re.compile(r"\[ITER 10000\] Evaluating test:.*LPIPS ([0-9.]+)")


def extract(log_path, regex):
    """Extract last value matching regex from log file."""
    if not os.path.exists(log_path):
        return None
    val = None
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = regex.search(line)
            if m:
                try:
                    val = float(m.group(1))
                except ValueError:
                    pass
    return val


def fmt(v, w=7, prec=4):
    if v is None:
        return f"{'—':>{w}}"
    return f"{v:{w}.{prec}f}"


def fmt_delta(v, w=8, prec=3):
    if v is None:
        return f"{'—':>{w}}"
    sign = '+' if v >= 0 else ''
    return f"{sign}{v:.{prec}f}".rjust(w)


def print_table(metric_name, regex):
    print(f"\n{'=' * 100}")
    print(f"  {metric_name} @10k")
    print(f"{'=' * 100}")

    # Header
    print(f"  {'Cfg':<8}", end="")
    for s in SCENES:
        print(f" | {s:>8}", end="")
    print(f" | {'AVG':>8}")
    print(f"  {'-' * 8}", end="")
    for _ in SCENES:
        print(f"-+-{'-' * 8}", end="")
    print(f"-+-{'-' * 8}")

    # Collect all values
    all_vals = {}  # cfg → scene → value
    for cfg in CONFIGS:
        all_vals[cfg] = {}
        for s in SCENES:
            log_path = os.path.join(LOGDIR, f"{cfg}_{s}.log")
            all_vals[cfg][s] = extract(log_path, regex)

    # Print rows
    for cfg in CONFIGS:
        print(f"  {cfg:<8}", end="")
        valid = []
        for s in SCENES:
            v = all_vals[cfg][s]
            print(f" | {fmt(v, 8)}", end="")
            if v is not None:
                valid.append(v)
        avg = sum(valid) / len(valid) if valid else None
        print(f" | {fmt(avg, 8)}")

    # Delta vs B0
    print()
    print(f"  Delta vs B0:")
    b0_vals = all_vals.get("B0", {})
    for cfg in CONFIGS:
        if cfg == "B0":
            continue
        print(f"  Δ{cfg:<7}", end="")
        deltas = []
        for s in SCENES:
            v = all_vals[cfg][s]
            b = b0_vals.get(s)
            if v is not None and b is not None:
                d = v - b
                deltas.append(d)
                print(f" | {fmt_delta(d, 8)}", end="")
            else:
                print(f" | {'—':>8}", end="")
        avg_d = sum(deltas) / len(deltas) if deltas else None
        print(f" | {fmt_delta(avg_d, 8)}")


def print_completeness():
    print(f"\n{'=' * 100}")
    print(f"  COMPLETENESS — log files present")
    print(f"{'=' * 100}")
    print(f"  {'Cfg':<8}", end="")
    for s in SCENES:
        print(f" | {s:>8}", end="")
    print()
    for cfg in CONFIGS:
        print(f"  {cfg:<8}", end="")
        for s in SCENES:
            log_path = os.path.join(LOGDIR, f"{cfg}_{s}.log")
            if not os.path.exists(log_path):
                mark = "MISS"
            else:
                v = extract(log_path, PSNR_RE)
                mark = "OK" if v is not None else "INPROG"
            print(f" | {mark:>8}", end="")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        LOGDIR = sys.argv[1]
    print(f"Reading from: {LOGDIR}")
    print_completeness()
    print_table("PSNR", PSNR_RE)
    print_table("SSIM", SSIM_RE)
    print_table("LPIPS", LPIPS_RE)
