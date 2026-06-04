#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Utility] Archive metrics from training logs
# File: scripts/archive_metrics.py
#
# Mục đích: Extract PSNR/SSIM/LPIPS/L1/N_gauss/training_time từ tất cả
# log files trước khi xóa output dirs. Lưu vào 1 CSV nhẹ (~vài KB) để
# defense Q&A reproducibility ngay cả khi output bị xóa.
#
# Log format mong đợi (từ train.py SUMMARY block):
#     30000 |  test |  23.7909 |   0.7896 |   0.1571 |   0.040729 |       78751
#     <iter> | test | <psnr>   | <ssim>   | <lpips>  | <l1>       | <n_gauss>
#
# Training time: parse timestamps "[DD/MM HH:MM:SS]" → first vs last
# Best PSNR: từ dòng "Best test PSNR: X.XXXX at iter N (N=...)"
#
# Usage:
#   python scripts/archive_metrics.py
#   # Output: logs/_archive_metrics_YYYYMMDD.csv
# ============================================================

import re
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# ── Tất cả log dirs cần archive ──
# Auto-detect: scan logs/ tìm mọi dir bắt đầu p*
LOG_BASE = Path("logs")

# Regex patterns
# SUMMARY line: "   30000 |  test |  23.7909 |   0.7896 |   0.1571 |   0.040729 |       78751"
SUMMARY_RE = re.compile(
    r"\s*(\d+)\s*\|\s*test\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(\d+)"
)
# Best PSNR cross-check
BEST_RE = re.compile(r"Best test PSNR:\s*([\d.]+)\s*at iter\s*(\d+)\s*\(N=(\d+)\)")
# Timestamp pattern (train.py logger format)
TIMESTAMP_RE = re.compile(r"\[(\d{2})/(\d{2})\s+(\d{2}):(\d{2}):(\d{2})\]")


def extract_metrics(log_path):
    """Extract test PSNR/SSIM/LPIPS/L1/N_gauss + training time from a log."""
    try:
        with open(log_path, "r", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return {"error": str(e)}

    # Find last test summary line (final iter eval)
    summaries = SUMMARY_RE.findall(content)
    if not summaries:
        return None  # log incomplete or different format

    iter_, psnr, ssim, lpips, l1, n_gauss = summaries[-1]

    # Cross-check best PSNR (should match)
    best = BEST_RE.search(content)
    best_psnr = float(best.group(1)) if best else float(psnr)

    # Training time from timestamps (first vs last)
    train_min = None
    timestamps = TIMESTAMP_RE.findall(content)
    if len(timestamps) >= 2:
        try:
            # Use day from first timestamp as anchor (no year info in logs)
            d_first = datetime(2000, int(timestamps[0][1]), int(timestamps[0][0]),
                              int(timestamps[0][2]), int(timestamps[0][3]), int(timestamps[0][4]))
            d_last = datetime(2000, int(timestamps[-1][1]), int(timestamps[-1][0]),
                             int(timestamps[-1][2]), int(timestamps[-1][3]), int(timestamps[-1][4]))
            delta = (d_last - d_first).total_seconds() / 60
            # Handle day wrap (rare but possible for long runs)
            if delta < 0:
                delta += 24 * 60
            train_min = round(delta, 1)
        except Exception:
            train_min = None

    return {
        "iter": iter_,
        "psnr": psnr,
        "best_psnr": f"{best_psnr:.4f}",
        "ssim": ssim,
        "lpips": lpips,
        "l1": l1,
        "n_gauss": n_gauss,
        "train_min": train_min if train_min is not None else "",
    }


def main():
    if not LOG_BASE.is_dir():
        print(f"❌ {LOG_BASE} not found — run from CoR-GS/ root")
        sys.exit(1)

    # Auto-detect all phase dirs (p*, t2min*, etc.)
    log_dirs = sorted([d for d in LOG_BASE.iterdir() if d.is_dir() and not d.name.startswith("_")])
    print(f"Scanning {len(log_dirs)} log dirs in {LOG_BASE}/")

    out_lines = ["log_dir,file,iter,psnr,best_psnr,ssim,lpips,l1,n_gauss,train_min"]
    n_total = 0
    n_extracted = 0
    n_incomplete = 0
    per_dir_count = {}

    for log_dir in log_dirs:
        per_dir = 0
        for log_file in sorted(log_dir.rglob("*.log")):
            n_total += 1
            m = extract_metrics(log_file)
            if m and "error" not in m and m is not None:
                if all(m.get(k) for k in ["psnr", "ssim", "lpips", "n_gauss"]):
                    n_extracted += 1
                    per_dir += 1
                    rel_path = str(log_file.relative_to(LOG_BASE))
                    out_lines.append(
                        f"{log_dir.name},{log_file.name},{m['iter']},{m['psnr']},{m['best_psnr']},"
                        f"{m['ssim']},{m['lpips']},{m['l1']},{m['n_gauss']},{m['train_min']}"
                    )
                else:
                    n_incomplete += 1
            else:
                n_incomplete += 1
        per_dir_count[log_dir.name] = per_dir

    # Write CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = LOG_BASE / f"_archive_metrics_{timestamp}.csv"
    with open(backup_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")

    size_kb = os.path.getsize(backup_path) / 1024
    print()
    print("=" * 70)
    print("ARCHIVE SUMMARY")
    print("=" * 70)
    print(f"  Total log files scanned: {n_total}")
    print(f"  Extracted:               {n_extracted}")
    print(f"  Incomplete/skipped:      {n_incomplete}")
    print(f"  Output file:             {backup_path}")
    print(f"  Size:                    {size_kb:.1f} KB")
    print()
    print("=" * 70)
    print("PER-DIR COUNT (extracted)")
    print("=" * 70)
    for d, c in per_dir_count.items():
        if c > 0:
            print(f"  {d:<35} {c} runs archived")
    print()
    print(f"Verify: head -10 {backup_path}")
    print(f"Filter (vd Phase 22): grep p22_pilot {backup_path}")


if __name__ == "__main__":
    main()
