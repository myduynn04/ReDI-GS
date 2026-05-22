#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 18 — Gate 1] Measure PDCNet+ dense init
# File: scripts/p18_gate1_measure.py  (TẠO MỚI — keep local)
#
# So sánh point cloud init: PDCNet+ dense (từ triangulate.py) vs
# COLMAP-MVS `fused.ply` (init hiện tại của A3).
#
# Đo per scene:
#   - N điểm mỗi loại + ratio
#   - has_normals (PDCNet+ ply — quyết định có cần fix normal-field không)
#   - bbox diag (raw + robust 1-99 percentile) — check scale + outlier
#   - center distance — check 2 cloud cùng coordinate frame
#   - frac PDCNet+ điểm nằm trong MVS-bbox (×1.2) — coverage/alignment
#
# Diagnostic-only. Chạy env có `plyfile`+`numpy` (corgs OK), từ dir CoR-GS.
# ============================================================
"""[CRSGaussian Phase 18 Gate 1] measure PDCNet+ dense init.

Server (env corgs, từ dir CoR-GS):
    python scripts/p18_gate1_measure.py
"""

import os
import numpy as np
from plyfile import PlyData

SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
PDCNET_DIR = os.environ.get("PDCNET_DIR", "/tmp/p18_gate1")
DATA_ROOT = os.environ.get("DATA_ROOT", "data/nerf_llff_data")
N_VIEWS = os.environ.get("N_VIEWS", "3")


def read_ply_xyz(path):
    """Return (xyz (N,3) float64, has_normals bool)."""
    p = PlyData.read(path)
    v = p["vertex"]
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)
    fields = v.data.dtype.names
    has_n = all(k in fields for k in ("nx", "ny", "nz"))
    return xyz, has_n


def bbox_diag(xyz, lo=0.0, hi=100.0):
    """Bounding-box diagonal. lo/hi percentile để robust với outlier."""
    mn = np.percentile(xyz, lo, axis=0)
    mx = np.percentile(xyz, hi, axis=0)
    return float(np.linalg.norm(mx - mn)), mn, mx


def main():
    print("=" * 92)
    print("Phase 18 Gate 1 — PDCNet+ dense init vs COLMAP-MVS fused.ply")
    print("=" * 92)
    print(f"PDCNET_DIR={PDCNET_DIR}  DATA_ROOT={DATA_ROOT}  N_VIEWS={N_VIEWS}\n")

    hdr = (f"{'scene':<10} {'N_pdcnet':>10} {'N_mvs':>9} {'ratio':>7} "
           f"{'pdc_norm':>8} {'diag_pdc':>9} {'diag_mvs':>9} "
           f"{'diag_pdc_r':>11} {'ctr_dist':>9} {'in_mvs%':>8}")
    print(hdr)
    print("-" * 92)

    rows = []
    for sc in SCENES:
        pdc_path = f"{PDCNET_DIR}/{sc}_keypoints_to_3d.ply"
        mvs_path = f"{DATA_ROOT}/{sc}/{N_VIEWS}_views/dense/fused.ply"

        if not os.path.isfile(pdc_path):
            print(f"{sc:<10} (PDCNet+ ply MISSING: {pdc_path})")
            continue
        pdc, pdc_hasn = read_ply_xyz(pdc_path)

        if not os.path.isfile(mvs_path):
            print(f"{sc:<10} {len(pdc):>10}  (MVS fused.ply MISSING — A3 init khác?)")
            rows.append((sc, len(pdc), None))
            continue
        mvs, _ = read_ply_xyz(mvs_path)

        diag_pdc_raw, _, _ = bbox_diag(pdc, 0, 100)
        diag_pdc_rob, _, _ = bbox_diag(pdc, 1, 99)        # robust, bỏ outlier
        diag_mvs, mvs_mn, mvs_mx = bbox_diag(mvs, 0, 100)
        ctr_pdc = (np.percentile(pdc, 1, 0) + np.percentile(pdc, 99, 0)) / 2
        ctr_mvs = (mvs_mn + mvs_mx) / 2
        ctr_dist = float(np.linalg.norm(ctr_pdc - ctr_mvs))

        # frac PDCNet+ điểm nằm trong MVS bbox mở rộng ×1.2
        ext = (mvs_mx - mvs_mn) * 0.1
        lo, hi = mvs_mn - ext, mvs_mx + ext
        inside = np.all((pdc >= lo) & (pdc <= hi), axis=1)
        in_frac = float(inside.mean()) * 100

        ratio = len(pdc) / max(len(mvs), 1)
        print(f"{sc:<10} {len(pdc):>10} {len(mvs):>9} {ratio:>6.1f}x "
              f"{'yes' if pdc_hasn else 'NO':>8} {diag_pdc_raw:>9.2f} "
              f"{diag_mvs:>9.2f} {diag_pdc_rob:>11.2f} {ctr_dist:>9.3f} "
              f"{in_frac:>7.1f}%")
        rows.append((sc, len(pdc), len(mvs)))

    # Aggregate
    valid = [(s, p, m) for (s, p, m) in rows if m is not None]
    if valid:
        tot_pdc = sum(p for _, p, _ in valid)
        tot_mvs = sum(m for _, _, m in valid)
        print("-" * 92)
        print(f"{'TOTAL':<10} {tot_pdc:>10} {tot_mvs:>9} "
              f"{tot_pdc / max(tot_mvs, 1):>6.1f}x")
    print()
    print("Đọc kết quả:")
    print("  ratio          : PDCNet+ dày gấp mấy lần COLMAP-MVS")
    print("  pdc_norm       : 'NO' = ply thiếu nx/ny/nz → cần fix normal-field")
    print("  diag_pdc vs _r : raw >> robust ⟹ có outlier triangulation xa")
    print("  diag_pdc_r vs diag_mvs : ~bằng nhau ⟹ cùng scale (tốt)")
    print("  ctr_dist       : nhỏ so với diag ⟹ 2 cloud cùng coordinate frame")
    print("  in_mvs%        : % điểm PDCNet+ nằm trong vùng MVS — cao ⟹ aligned")


if __name__ == "__main__":
    main()
