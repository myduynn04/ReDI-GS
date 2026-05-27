#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 22 — Bước 2] Place / Restore RoMa v1 dense init
# File: scripts/p22_place_romav1_init.py  (KEEP LOCAL — server-only)
#
# Analog p21_place_roma_init.py, source = .ply.romav1 (Phase 22).
#
# Backup convention CỘNG DỒN:
#   <data>/<scene>/3_views/dense/fused.ply.colmap_mvs_backup   ← Phase 18 backup (giữ nguyên)
#   <data>/<scene>/3_views/dense/fused.ply.roma                ← Phase 21 v2 (giữ nguyên)
#   <data>/<scene>/3_views/dense/fused.ply.romav1              ← Phase 22 v1 (sinh từ p22 preprocess)
#   <data>/<scene>/3_views/dense/fused.ply                     ← state hiện tại (swap được)
#
# Mode PLACE (default): swap fused.ply ← fused.ply.romav1
# Mode RESTORE (env RESTORE=1): swap fused.ply ← backup (= COLMAP-MVS gốc)
#
# Sau Phase 22 pilot, nếu muốn quay v2 cho comparison: chạy p21_place_roma_init.py
# Nếu muốn quay MVS: chạy script này hoặc p21 với RESTORE=1.
# ============================================================
"""Place / restore RoMa v1 dense init."""

import os
import shutil
import sys
from pathlib import Path

SCENES = os.environ.get("SCENES", "fern flower fortress horns leaves orchids room trex").split()
DATA_ROOT = os.environ.get("DATA_ROOT", "data/nerf_llff_data")
N_VIEWS = os.environ.get("N_VIEWS", "3")
RESTORE = os.environ.get("RESTORE", "0") == "1"
BACKUP_SUFFIX = ".colmap_mvs_backup"
V1_SUFFIX = ".romav1"


def main():
    print("=" * 64)
    print(f"Phase 22 — {'RESTORE COLMAP-MVS' if RESTORE else 'PLACE RoMa v1'} init")
    print(f"SCENES = {SCENES}")
    print("=" * 64)

    n_ok, n_skip = 0, 0
    for sc in SCENES:
        dst = Path(DATA_ROOT) / sc / f"{N_VIEWS}_views/dense/fused.ply"
        backup = Path(str(dst) + BACKUP_SUFFIX)
        v1_ply = Path(str(dst) + V1_SUFFIX)

        if RESTORE:
            if backup.is_file():
                shutil.copy(backup, dst)
                print(f"  {sc:<10} RESTORED: fused.ply ← {BACKUP_SUFFIX}")
                n_ok += 1
            else:
                print(f"  {sc:<10} ⚠ no backup ({backup}) — skip")
                n_skip += 1
            continue

        if not v1_ply.is_file():
            print(f"  {sc:<10} ❌ RoMa v1 ply MISSING: {v1_ply} — run p22_run_all_scenes.sh first")
            n_skip += 1
            continue
        if not dst.is_file():
            print(f"  {sc:<10} ❌ fused.ply MISSING: {dst}")
            n_skip += 1
            continue

        # backup CHỈ 1 LẦN (giữ nguyên Phase 18 + Phase 21 backup nếu đã có)
        if not backup.is_file():
            shutil.copy(dst, backup)
            sz_kb = backup.stat().st_size / 1024
            print(f"  {sc:<10} backup ({sz_kb:.1f} KB) → {BACKUP_SUFFIX}")
        else:
            print(f"  {sc:<10} backup đã có (giữ nguyên — từ Phase 18/21)")

        shutil.copy(v1_ply, dst)
        sz_kb = dst.stat().st_size / 1024
        print(f"  {sc:<10} ✅ placed RoMa v1 init → fused.ply ({sz_kb:.1f} KB)")
        n_ok += 1

    print()
    print(f"Done — {n_ok} ok, {n_skip} skip")
    if not RESTORE:
        print(f"⚠ Sau Phase 22 pilot: RESTORE=1 python {sys.argv[0]}  (về MVS)")
        print(f"    HOẶC: python scripts/p21_place_roma_init.py  (về RoMa v2)")


if __name__ == "__main__":
    main()
