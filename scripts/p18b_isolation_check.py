#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 18b] CRS-pruning isolation-gate verify
# File: scripts/p18b_isolation_check.py  (TẠO MỚI — keep local)
#
# KHÔNG train. Kiểm giả thuyết (user, 2026-05-22): CRS pruning isolation-gate
#   isolated = knn_dist > tau_isolated * extent      (gaussian_model.py:971)
# CHẾT trên dense init vì cloud đặc → knn_dist co nhỏ → gần như 0 điểm vượt
# ngưỡng tuyệt-đối 0.1*extent → crs_prune = crs_low & isolated ≈ toàn False.
#
# Đo % điểm "isolated" trên 4 loại cloud / scene:
#   INIT-MVS   : fused.ply.colmap_mvs_backup     (sparse init — gate THIẾT KẾ cho)
#   INIT-PDC   : <scene>_keypoints_to_3d.ply     (dense init — nghi gate chết)
#   TRAINED-sparse : output/p13_lfcf  ckpt iter 10000
#   TRAINED-dense  : output/p18_pilot ckpt iter 10000
# extent = scene.cameras_extent (camera-based → giống nhau cho mọi init).
#
# Replicate ĐÚNG code: knn_dist = distCUDA2(xyz) (squared dist, simple_knn),
# threshold = tau_isolated * extent, tau_isolated=0.1 (densify_and_prune default).
#
# Chạy (env corgs/3DGS có simple_knn, GPU, dir CoR-GS):
#   python scripts/p18b_isolation_check.py
#   PDCNET_DIR=/tmp/p18_gate1 python scripts/p18b_isolation_check.py
# ============================================================
"""[CRSGaussian Phase 18b] verify CRS-pruning isolation-gate vs init density."""

import os
import sys
import numpy as np

sys.path.insert(0, ".")

DATA_ROOT = os.environ.get("DATA_ROOT", "data/nerf_llff_data")
PDCNET_DIR = os.environ.get("PDCNET_DIR", "/tmp/p18_gate1")
SPARSE_ROOT = os.environ.get("SPARSE_ROOT", "output/p13_lfcf")
DENSE_ROOT = os.environ.get("DENSE_ROOT", "output/p18_pilot")
SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
SEED = os.environ.get("SEED", "42")
ITER = int(os.environ.get("ITERATION", "10000"))
N_VIEWS = os.environ.get("N_VIEWS", "3")
TAU_ISO = float(os.environ.get("TAU_ISOLATED", "0.1"))   # densify_and_prune default
TAU_SWEEP = [0.01, 0.02, 0.05, 0.10]                     # context sweep


def read_ply_xyz(path):
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    return np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)


def scene_extent(scene):
    """Load Scene chỉ để lấy cameras_extent (camera-based, init-independent).
    Mirror p13_2_bottleneck_decompose loading."""
    from argparse import ArgumentParser, Namespace
    from scene import Scene, GaussianModel
    from arguments import ModelParams, PipelineParams
    mp = f"{SPARSE_ROOT}/A3_seed{SEED}_{scene}"
    cfg = os.path.join(mp, "cfg_args")
    if not os.path.isfile(cfg):
        return None
    parser = ArgumentParser()
    _lp = ModelParams(parser)
    _pp = PipelineParams(parser)
    args_cmd = parser.parse_args([])
    with open(cfg) as f:
        cfgo = eval(f.read())
    merged = vars(args_cmd).copy()
    for k, v in vars(cfgo).items():
        merged[k] = v
    args = Namespace(**merged)
    args.model_path = mp
    args.iteration = ITER
    if not os.path.isdir(getattr(args, "source_path", "") or ""):
        args.source_path = os.path.join(DATA_ROOT, scene)
    g = GaussianModel(args)
    sc = Scene(args, g, load_iteration=ITER, shuffle=False)
    return float(sc.cameras_extent)


def knn_stats(xyz_np, extent):
    """distCUDA2 → knn_dist (squared, simple_knn). Trả stats + %isolated mỗi tau."""
    import torch
    from simple_knn._C import distCUDA2
    xyz = torch.tensor(xyz_np, dtype=torch.float32, device="cuda")
    knn, _ = distCUDA2(xyz)
    knn = knn.detach().cpu().numpy()
    iso = {t: 100.0 * float((knn > t * extent).mean()) for t in TAU_SWEEP}
    return dict(N=len(knn), med=float(np.median(knn)),
                p99=float(np.percentile(knn, 99)), mx=float(knn.max()), iso=iso)


def trained_ply(root, scene):
    return (f"{root}/A3_seed{SEED}_{scene}/point_cloud/"
            f"iteration_{ITER}/point_cloud.ply")


def main():
    print("=" * 92)
    print("Phase 18b — CRS-pruning isolation-gate verify (knn_dist > tau*extent)")
    print(f"tau_isolated={TAU_ISO}  PDCNET_DIR={PDCNET_DIR}  SEED={SEED}")
    print("Giả thuyết: dense init → knn co nhỏ → %isolated@0.1 ≈ 0 → CRS-prune chết")
    print("=" * 92)

    rows = []
    for sc in SCENES:
        ext = scene_extent(sc)
        if ext is None:
            print(f"{sc:<10} (no cfg_args ở {SPARSE_ROOT} — skip)")
            continue
        clouds = {
            "INIT-MVS": f"{DATA_ROOT}/{sc}/{N_VIEWS}_views/dense/fused.ply.colmap_mvs_backup",
            "INIT-PDC": f"{PDCNET_DIR}/{sc}_keypoints_to_3d.ply",
            "TRAIN-sparse": trained_ply(SPARSE_ROOT, sc),
            "TRAIN-dense": trained_ply(DENSE_ROOT, sc),
        }
        print(f"\n──── {sc}   (extent={ext:.3f}, threshold@0.1 = {0.1*ext:.4f}) ────")
        print(f"  {'cloud':<14} {'N':>9} {'med_knn':>10} {'p99_knn':>10} "
              + " ".join(f'iso@{t:g}'.rjust(9) for t in TAU_SWEEP))
        for label, path in clouds.items():
            if not os.path.isfile(path):
                print(f"  {label:<14} (MISSING: {path})")
                continue
            try:
                st = knn_stats(read_ply_xyz(path), ext)
            except Exception as e:
                print(f"  {label:<14} ERR {e}")
                continue
            isos = " ".join(f"{st['iso'][t]:>8.2f}%" for t in TAU_SWEEP)
            print(f"  {label:<14} {st['N']:>9} {st['med']:>10.5f} "
                  f"{st['p99']:>10.5f} {isos}")
            rows.append((sc, label, st))

    # ── Aggregate ──
    print("\n" + "=" * 92)
    print("AGGREGATE — mean %isolated@tau over scenes, per cloud type")
    print("=" * 92)
    print(f"  {'cloud':<14} " + " ".join(f'iso@{t:g}'.rjust(9) for t in TAU_SWEEP))
    for label in ["INIT-MVS", "INIT-PDC", "TRAIN-sparse", "TRAIN-dense"]:
        sub = [st for (s, l, st) in rows if l == label]
        if not sub:
            continue
        agg = " ".join(
            f"{np.mean([st['iso'][t] for st in sub]):>8.2f}%" for t in TAU_SWEEP)
        print(f"  {label:<14} {agg}")
    print()
    print("ĐỌC: cột iso@0.1 = % điểm gate-isolation BẮN (tau_isolated mặc định).")
    print("  INIT-MVS iso@0.1 >> INIT-PDC iso@0.1 ≈ 0  ⟹ giả thuyết ĐÚNG:")
    print("  gate calibrate cho sparse, CHẾT trên dense init.")
    print("  → fix = tau_isolated density-adaptive (percentile / ×median knn).")


if __name__ == "__main__":
    main()
