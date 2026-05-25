#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 21 — Bước 2] Place / Restore RoMa v2 dense init
# File: scripts/p21_place_roma_init.py  (KEEP LOCAL — server-only)
#
# Analog của p18_gate2_place_dense_init.py nhưng nguồn = .ply.roma
# (sinh bởi p21_roma_preprocess.py).
#
# KHÔNG đụng code CRSGaussian. Chỉ thay 1 file dữ liệu per scene:
#   <data>/<scene>/3_views/dense/fused.ply   ← swap thành RoMa
#
# Mode PLACE (default):
#   1. Backup fused.ply gốc (COLMAP-MVS) → fused.ply.colmap_mvs_backup
#      (chỉ backup 1 LẦN — re-run không đè backup; tương thích Phase 18 backup cũ)
#   2. Copy fused.ply.roma → fused.ply
#
# Mode RESTORE (env RESTORE=1):
#   Copy backup → fused.ply (hoàn nguyên COLMAP-MVS init).
#   Chạy SAU pilot để A3-TRIM-MVS baseline nguyên vẹn.
#
# ⚠ Nếu Phase 18 backup .colmap_mvs_backup vẫn còn (từ session cũ): GIỮ NGUYÊN
#   (script không đè). Phase 18 đã RESTORE post-NO-verdict → fused.ply hiện
#   = COLMAP-MVS gốc. Phase 21 backup-once logic vẫn an toàn.
# ============================================================
"""Place / restore RoMa v2 dense init for CRSGaussian.

Server (env corgs hay romav2 đều OK — chỉ file copy):
    # PLACE init (default 8 scenes):
    python scripts/p21_place_roma_init.py

    # RESTORE COLMAP-MVS sau pilot:
    RESTORE=1 python scripts/p21_place_roma_init.py

    # 1 scene only:
    SCENES="fern" python scripts/p21_place_roma_init.py
"""

import os
import shutil
import sys
from pathlib import Path

SCENES = os.environ.get("SCENES", "fern flower fortress horns leaves orchids room trex").split()
DATA_ROOT = os.environ.get("DATA_ROOT", "data/nerf_llff_data")
N_VIEWS = os.environ.get("N_VIEWS", "3")
RESTORE = os.environ.get("RESTORE", "0") == "1"
BACKUP_SUFFIX = ".colmap_mvs_backup"
ROMA_SUFFIX = ".roma"


def main():
    print("=" * 64)
    print(f"Phase 21 — {'RESTORE COLMAP-MVS' if RESTORE else 'PLACE RoMa v2'} init")
    print(f"SCENES = {SCENES}")
    print("=" * 64)

    n_ok, n_skip = 0, 0
    for sc in SCENES:
        dst = Path(DATA_ROOT) / sc / f"{N_VIEWS}_views/dense/fused.ply"
        backup = Path(str(dst) + BACKUP_SUFFIX)
        roma = Path(str(dst) + ROMA_SUFFIX)

        if RESTORE:
            if backup.is_file():
                shutil.copy(backup, dst)
                print(f"  {sc:<10} RESTORED: fused.ply ← {BACKUP_SUFFIX}")
                n_ok += 1
            else:
                print(f"  {sc:<10} ⚠ no backup ({backup}) — skip")
                n_skip += 1
            continue

        # ── PLACE mode ──
        if not roma.is_file():
            print(f"  {sc:<10} ❌ RoMa ply MISSING: {roma} — run p21_roma_preprocess.py first")
            n_skip += 1
            continue
        if not dst.is_file():
            print(f"  {sc:<10} ❌ fused.ply MISSING: {dst} — kiểm tra DATA_ROOT")
            n_skip += 1
            continue

        # backup CHỈ 1 LẦN (tương thích Phase 18 backup nếu còn)
        if not backup.is_file():
            shutil.copy(dst, backup)
            sz_kb = backup.stat().st_size / 1024
            print(f"  {sc:<10} backup COLMAP-MVS ({sz_kb:.1f} KB) → {BACKUP_SUFFIX}")
        else:
            print(f"  {sc:<10} backup đã có (giữ nguyên)")

        shutil.copy(roma, dst)
        sz_kb = dst.stat().st_size / 1024
        print(f"  {sc:<10} ✅ placed RoMa init → fused.ply ({sz_kb:.1f} KB)")
        n_ok += 1

    print()
    print(f"Done — {n_ok} ok, {n_skip} skip")
    if not RESTORE:
        print(f"⚠ Sau pilot nhớ: RESTORE=1 python {sys.argv[0]}")


if __name__ == "__main__":
    main()
