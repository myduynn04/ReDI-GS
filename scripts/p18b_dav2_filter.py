#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 18b — D2a] DAV2 q_depth hard-filter dense init
# File: scripts/p18b_dav2_filter.py  (TẠO MỚI — keep local)
#
# KHÔNG train. Đọc raw dense init cloud (PDCNet+), tính q_depth per-point
# (compute_depth_agreement — CRSGaussian có sẵn), GIỮ điểm q_depth ≥ τ_q,
# ghi cloud đã lọc → p18_gate2_place_dense_init.py đặt thành fused.ply.
#
# τ_q = 0.5 (pre-registered, decisions_log [2026-05-22] Phase 18b D2a):
#   neutral pivot của CRS framework — drop điểm "tệ hơn neutral về depth".
#
# 0 đụng code CRSGaussian — chỉ tạo file .ply init đã lọc.
#
# Chạy (env corgs, dir CoR-GS, GPU + DAV2 weights):
#   python scripts/p18b_dav2_filter.py
#   SCENES="horns trex fortress" python scripts/p18b_dav2_filter.py
# ============================================================
"""[CRSGaussian Phase 18b D2a] hard-filter dense init by DAV2 q_depth."""

import os
import sys
import numpy as np

sys.path.insert(0, ".")

DATA_ROOT = os.environ.get("DATA_ROOT", "data/nerf_llff_data")
PDCNET_DIR = os.environ.get("PDCNET_DIR", "/tmp/p18_gate1")          # raw dense
OUT_DIR = os.environ.get("OUT_DIR", "/tmp/p18b_dav2_filtered")       # filtered
SPARSE_ROOT = os.environ.get("SPARSE_ROOT", "output/p13_lfcf")       # cfg_args
SCENES = os.environ.get("SCENES", "horns trex fortress").split()
SEED = os.environ.get("SEED", "42")
ITER = int(os.environ.get("ITERATION", "10000"))
TAU_Q = float(os.environ.get("TAU_Q", "0.5"))   # pre-registered, locked


def read_ply_xyzrgb(path):
    """Đọc trimesh-style ply (x,y,z,red,green,blue) → (xyz f4, rgb u1)."""
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
    rgb = np.vstack([v["red"], v["green"], v["blue"]]).T.astype(np.uint8)
    return xyz, rgb


def write_ply_xyzrgb(path, xyz, rgb):
    """Ghi ply (x,y,z,red,green,blue) — khớp read_pdcnet_ply của p18_gate2."""
    from plyfile import PlyData, PlyElement
    dt = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
          ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    el = np.empty(xyz.shape[0], dtype=dt)
    el[:] = list(map(tuple, np.concatenate(
        [xyz.astype(np.float32), rgb.astype(np.uint8)], axis=1)))
    PlyData([PlyElement.describe(el, 'vertex')]).write(path)


def load_scene(scene):
    """Load Scene từ SPARSE_ROOT cfg_args → (train_cameras, args)."""
    from argparse import ArgumentParser, Namespace
    from scene import Scene, GaussianModel
    from arguments import ModelParams, PipelineParams
    mp = f"{SPARSE_ROOT}/A3_seed{SEED}_{scene}"
    cfg = os.path.join(mp, "cfg_args")
    if not os.path.isfile(cfg):
        return None, None
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
    return sc.getTrainCameras(), args


def main():
    import torch
    from utils.crs.crs_init import compute_depth_agreement
    from utils.depth import precompute_depth_priors, align_depth_to_colmap

    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 78)
    print(f"Phase 18b D2a — q_depth hard-filter dense init  (τ_q = {TAU_Q})")
    print(f"raw={PDCNET_DIR}  →  filtered={OUT_DIR}")
    print("=" * 78)

    for sc in SCENES:
        raw = f"{PDCNET_DIR}/{sc}_keypoints_to_3d.ply"
        if not os.path.isfile(raw):
            print(f"  {sc:<10} ❌ raw ply MISSING: {raw} — skip")
            continue
        cams, args = load_scene(sc)
        if cams is None:
            print(f"  {sc:<10} ❌ no cfg_args ({SPARSE_ROOT}) — skip")
            continue
        dp = precompute_depth_priors(
            cams, args.dav2_path,
            encoder=getattr(args, "dav2_encoder", "vitl"))
        aligned, drange = align_depth_to_colmap(
            dp, cams, args.source_path, args.n_views)

        xyz, rgb = read_ply_xyzrgb(raw)
        pts = torch.tensor(xyz, dtype=torch.float32, device="cuda")
        q = compute_depth_agreement(pts, cams, aligned, drange)
        if hasattr(q, "detach"):
            q = q.detach().cpu().numpy()
        q = np.asarray(q).reshape(-1)

        keep = q >= TAU_Q
        n_keep, n_tot = int(keep.sum()), len(keep)
        dst = f"{OUT_DIR}/{sc}_keypoints_to_3d.ply"
        write_ply_xyzrgb(dst, xyz[keep], rgb[keep])
        print(f"  {sc:<10} kept {n_keep}/{n_tot} ({100.0*n_keep/n_tot:.1f}%)  "
              f"drop {100.0*(1-n_keep/n_tot):.1f}%  → {dst}")

    print()
    print(f"Done. Filtered clouds ở {OUT_DIR}.")
    print("Tiếp: PDCNET_DIR=" + OUT_DIR + " SCENES=\"...\" "
          "python scripts/p18_gate2_place_dense_init.py")


if __name__ == "__main__":
    main()
