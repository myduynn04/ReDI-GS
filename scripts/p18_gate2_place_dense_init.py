#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 18 — Gate 2] Place PDCNet+ dense init as fused.ply
# File: scripts/p18_gate2_place_dense_init.py  (TẠO MỚI — keep local)
#
# KHÔNG đụng code CRSGaussian. Chỉ thay 1 file DỮ LIỆU per scene:
#   <data>/<scene>/<n>_views/dense/fused.ply
# = init point cloud mà A3 đọc vào. 100% reversible (có backup).
#
# Mode PLACE (default):
#   1. Backup fused.ply gốc (COLMAP-MVS) → fused.ply.colmap_mvs_backup
#      (chỉ backup 1 LẦN — re-run không ghi đè backup)
#   2. Đọc PDCNet+ ply (/tmp/p18_gate1/<scene>_keypoints_to_3d.ply)
#   3. Ghi đè fused.ply = PDCNet+ cloud, KÈM nx/ny/nz=0
#      (CRSGaussian fetchPly BẮT BUỘC có normal field — ply trimesh thiếu)
#
# Mode RESTORE (env RESTORE=1):
#   Copy backup → fused.ply (hoàn nguyên COLMAP-MVS init).
#   Chạy SAU Gate 2 để data dir sạch lại, A3 baseline nguyên vẹn.
#
# storePly format = EXACT copy CRSGaussian scene/dataset_readers.py:267
# (x,y,z,nx,ny,nz,red,green,blue) → fetchPly đọc khớp 100%.
#
# KHÔNG filter outlier — test as-is (faithful Binocular3DGS recipe).
# ============================================================
"""[CRSGaussian Phase 18 Gate 2] place / restore PDCNet+ dense init.

Server (env corgs, từ dir CoR-GS):
    # PLACE init mới (fortress + room):
    python scripts/p18_gate2_place_dense_init.py
    # RESTORE init gốc sau Gate 2:
    RESTORE=1 python scripts/p18_gate2_place_dense_init.py
"""

import os
import shutil
import numpy as np
from plyfile import PlyData, PlyElement

SCENES = os.environ.get("SCENES", "fortress room").split()
PDCNET_DIR = os.environ.get("PDCNET_DIR", "/tmp/p18_gate1")
DATA_ROOT = os.environ.get("DATA_ROOT", "data/nerf_llff_data")
N_VIEWS = os.environ.get("N_VIEWS", "3")
RESTORE = os.environ.get("RESTORE", "0") == "1"
BACKUP_SUFFIX = ".colmap_mvs_backup"


def store_ply_crsg(path, xyz, rgb):
    """EXACT copy CRSGaussian storePly (dataset_readers.py:267) — ghi
    x,y,z,nx,ny,nz(=0),red,green,blue để fetchPly đọc đúng format."""
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)


def read_pdcnet_ply(path):
    """Đọc PDCNet+ trimesh ply → (xyz float32 (N,3), rgb float32 (N,3) 0-255)."""
    p = PlyData.read(path)
    v = p['vertex']
    xyz = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float32)
    rgb = np.vstack([v['red'], v['green'], v['blue']]).T.astype(np.float32)
    return xyz, rgb


def main():
    print("=" * 64)
    print("Phase 18 Gate 2 — place/restore PDCNet+ dense init")
    print("=" * 64)
    print(f"mode = {'RESTORE' if RESTORE else 'PLACE'}   SCENES = {SCENES}")
    print()

    for sc in SCENES:
        dst = f"{DATA_ROOT}/{sc}/{N_VIEWS}_views/dense/fused.ply"
        backup = dst + BACKUP_SUFFIX

        if RESTORE:
            if os.path.isfile(backup):
                shutil.copy(backup, dst)
                print(f"  {sc:<10} RESTORED: fused.ply ← {BACKUP_SUFFIX}")
            else:
                print(f"  {sc:<10} no backup ({backup}) — skip")
            continue

        # ── PLACE mode ──
        src = f"{PDCNET_DIR}/{sc}_keypoints_to_3d.ply"
        if not os.path.isfile(src):
            print(f"  {sc:<10} ❌ PDCNet+ ply MISSING: {src} — skip")
            continue
        if not os.path.isfile(dst):
            print(f"  {sc:<10} ❌ fused.ply MISSING: {dst} — skip")
            continue

        # backup CHỈ 1 LẦN — nếu backup đã tồn tại, giữ nguyên (đừng đè
        # backup bằng bản PDCNet+ khi re-run)
        if not os.path.isfile(backup):
            shutil.copy(dst, backup)
            sz = os.path.getsize(backup)
            print(f"  {sc:<10} backup COLMAP-MVS fused.ply ({sz} bytes) → {BACKUP_SUFFIX}")
        else:
            print(f"  {sc:<10} backup đã có (giữ nguyên bản gốc)")

        xyz, rgb = read_pdcnet_ply(src)
        store_ply_crsg(dst, xyz, rgb)
        print(f"  {sc:<10} ✅ placed PDCNet+ init → fused.ply  "
              f"({len(xyz)} điểm, +zero-normals)")

    print()
    if RESTORE:
        print("Done RESTORE — fused.ply đã về COLMAP-MVS gốc.")
    else:
        print("Done PLACE — fused.ply giờ là PDCNet+ dense init.")
        print("⚠️  Sau Gate 2 nhớ chạy: RESTORE=1 python scripts/p18_gate2_place_dense_init.py")


if __name__ == "__main__":
    main()
