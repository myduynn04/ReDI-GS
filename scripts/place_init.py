#!/usr/bin/env python3
"""Place or restore the RoMa v1 dense initialization.

The training code reads each scene's initial point cloud from
``<scene>/3_views/dense/fused.ply``. This helper swaps that file with
the RoMa v1 cloud produced by ``preprocess.py``:

* Default mode places the RoMa v1 cloud as the active ``fused.ply`` and,
  the first time it runs, keeps a backup of the original COLMAP MVS
  cloud at ``fused.ply.colmap_mvs_backup``.
* ``RESTORE=1 python scripts/place_init.py`` reverts ``fused.ply`` back
  to the COLMAP MVS backup, useful for sparse-init experiments.

By default the script processes all 8 LLFF scenes; override with
``SCENES="fern flower"`` etc.
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
V1_SUFFIX = ".romav1"


def main():
    print("=" * 64)
    print(f"{'RESTORE COLMAP-MVS' if RESTORE else 'PLACE RoMa v1'} init")
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
            print(f"  {sc:<10} ❌ RoMa v1 ply MISSING: {v1_ply} — run preprocess_all.sh first")
            n_skip += 1
            continue
        if not dst.is_file():
            print(f"  {sc:<10} ❌ fused.ply MISSING: {dst}")
            n_skip += 1
            continue

        # Back up only once, so a re-run never overwrites the original MVS cloud.
        if not backup.is_file():
            shutil.copy(dst, backup)
            sz_kb = backup.stat().st_size / 1024
            print(f"  {sc:<10} backup ({sz_kb:.1f} KB) → {BACKUP_SUFFIX}")
        else:
            print(f"  {sc:<10} backup already exists — kept")

        shutil.copy(v1_ply, dst)
        sz_kb = dst.stat().st_size / 1024
        print(f"  {sc:<10} ✅ placed RoMa v1 init → fused.ply ({sz_kb:.1f} KB)")
        n_ok += 1

    print()
    print(f"Done — {n_ok} ok, {n_skip} skip")
    if not RESTORE:
        print(f"To revert to the COLMAP MVS init: RESTORE=1 python {sys.argv[0]}")


if __name__ == "__main__":
    main()
