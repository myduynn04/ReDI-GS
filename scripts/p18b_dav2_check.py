#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 18b — D2] DAV2 depth-agreement check on dense init
# File: scripts/p18b_dav2_check.py  (TẠO MỚI — keep local)
#
# KHÔNG train. Kiểm: tín hiệu q_depth (DAV2 depth-agreement — ĐÃ có sẵn
# trong informed_crs_init) có flag được điểm xấu horns/trex KHÔNG?
#
# q_depth per-point = compute_depth_agreement(): chiếu điểm init vào mỗi
# train cam → d_point; tra DAV2 aligned depth tại pixel → d_dav2;
#   q = 1 − |d_dav2 − d_point| / depth_range   (clamp [0,1], avg qua cam)
# q THẤP = điểm lệch DAV2 = ứng viên phantom-từ-bad-match.
#
# Đo q_depth cho INIT-MVS vs INIT-PDC (dense), per scene. Câu hỏi:
#   %(q_depth thấp) có TẬP TRUNG ở horns/trex (scene thua) không?
#   - Tập trung + orchids sạch  → (B) tín hiệu đúng, lỗi downstream → D2 đáng pilot
#   - Rải đều / orchids cũng bẩn → (A) DAV2 mù thin-structure → D2 chết
#
# Tái dùng machinery CRSGaussian: precompute_depth_priors + align_depth_to_colmap
# + compute_depth_agreement (KHÔNG tự chế).
#
# Chạy (env corgs, dir CoR-GS, GPU + DAV2 weights):
#   python scripts/p18b_dav2_check.py
#   PDCNET_DIR=/tmp/p18_gate1 python scripts/p18b_dav2_check.py
# ============================================================
"""[CRSGaussian Phase 18b D2] DAV2 depth-agreement check on dense init cloud."""

import os
import sys
import numpy as np

sys.path.insert(0, ".")

DATA_ROOT = os.environ.get("DATA_ROOT", "data/nerf_llff_data")
PDCNET_DIR = os.environ.get("PDCNET_DIR", "/tmp/p18_gate1")
SPARSE_ROOT = os.environ.get("SPARSE_ROOT", "output/p13_lfcf")   # cho cfg_args
SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
SEED = os.environ.get("SEED", "42")
ITER = int(os.environ.get("ITERATION", "10000"))
N_VIEWS_DIR = os.environ.get("N_VIEWS_DIR", "3")    # cho path fused.ply

# Pilot N=24 paired Δ (PDCNet+ raw − MVS) — decisions_log [2026-05-21].
PILOT_DELTA = {
    "fern": +0.077, "flower": +0.512, "fortress": +0.996, "horns": -0.318,
    "leaves": +0.839, "orchids": +0.139, "room": +0.268, "trex": -0.357,
}
GOOD = ["fortress", "leaves", "flower"]
BAD = ["horns", "trex"]
QT = [0.3, 0.5, 0.7]   # ngưỡng "q_depth thấp"


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xm, ym = x - x.mean(), y - y.mean()
    d = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    return float((xm * ym).sum() / d) if d > 0 else float("nan")


def read_ply_xyz(path):
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    return np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)


def load_scene(scene):
    """Load Scene từ SPARSE_ROOT cfg_args → (train_cameras, args).
    Mirror p13_2_bottleneck_decompose loading."""
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


def q_depth_stats(xyz_np, cams, aligned, drange):
    """compute_depth_agreement trên cloud → stats q_depth."""
    import torch
    from utils.crs.crs_init import compute_depth_agreement
    pts = torch.tensor(xyz_np, dtype=torch.float32, device="cuda")
    q = compute_depth_agreement(pts, cams, aligned, drange)
    if hasattr(q, "detach"):          # torch tensor → numpy
        q = q.detach().cpu().numpy()
    q = np.asarray(q).reshape(-1)
    return dict(N=len(q), med=float(np.median(q)), mean=float(q.mean()),
                lo={t: 100.0 * float((q < t).mean()) for t in QT},
                neutral=100.0 * float((q == 0.5).mean()))


def main():
    print("=" * 90)
    print("Phase 18b D2 — DAV2 depth-agreement (q_depth) trên init cloud")
    print(f"PDCNET_DIR={PDCNET_DIR}  SEED={SEED}")
    print("q_depth THẤP = điểm lệch DAV2. Hỏi: %thấp có tập trung horns/trex?")
    print("=" * 90)

    rows = {}
    for sc in SCENES:
        cams, args = load_scene(sc)
        if cams is None:
            print(f"{sc:<10} (no cfg_args ở {SPARSE_ROOT} — skip)")
            continue
        from utils.depth import precompute_depth_priors, align_depth_to_colmap
        dp = precompute_depth_priors(
            cams, args.dav2_path,
            encoder=getattr(args, "dav2_encoder", "vitl"))
        aligned, drange = align_depth_to_colmap(
            dp, cams, args.source_path, args.n_views)

        clouds = {
            "INIT-MVS": f"{DATA_ROOT}/{sc}/{N_VIEWS_DIR}_views/dense/"
                        f"fused.ply.colmap_mvs_backup",
            "INIT-PDC": f"{PDCNET_DIR}/{sc}_keypoints_to_3d.ply",
        }
        print(f"\n──── {sc}  (depth_range={drange:.3f}, pilotΔ={PILOT_DELTA[sc]:+.3f}) ────")
        print(f"  {'cloud':<12} {'N':>9} {'q_med':>7} {'q_mean':>7} "
              + " ".join(f'%<{t:g}'.rjust(8) for t in QT) + f" {'neutral':>8}")
        for label, path in clouds.items():
            if not os.path.isfile(path):
                print(f"  {label:<12} (MISSING: {path})")
                continue
            st = q_depth_stats(read_ply_xyz(path), cams, aligned, drange)
            los = " ".join(f"{st['lo'][t]:>7.2f}%" for t in QT)
            print(f"  {label:<12} {st['N']:>9} {st['med']:>7.3f} "
                  f"{st['mean']:>7.3f} {los} {st['neutral']:>7.1f}%")
            rows.setdefault(label, {})[sc] = st

    # ── DISCRIMINATION — INIT-PDC %low-q_depth vs pilot Δ ──
    pdc = rows.get("INIT-PDC", {})
    if pdc:
        print("\n" + "=" * 90)
        print("DISCRIMINATION — INIT-PDC: %(q_depth<0.5) per scene vs pilot Δ")
        print("=" * 90)
        scs = [s for s in SCENES if s in pdc]
        badf = {s: pdc[s]["lo"][0.5] for s in scs}
        dl = [PILOT_DELTA[s] for s in scs]
        print(f"  r(%low-q_depth, Δ) Pearson = {pearson([badf[s] for s in scs], dl):+.3f}"
              "   (D2 có substrate ⟹ ÂM mạnh)")
        print("\n  scene xếp theo pilot Δ:")
        for s in sorted(scs, key=lambda z: PILOT_DELTA[z]):
            tag = ("  <--BAD" if s in BAD else
                   ("  <--good" if s in GOOD else
                    ("  <--orchids" if s == "orchids" else "")))
            print(f"    {s:<10} Δ={PILOT_DELTA[s]:+.3f}   "
                  f"%low-q_depth={badf[s]:6.2f}%{tag}")
        gm = np.mean([badf[s] for s in BAD if s in badf])
        wm = np.mean([badf[s] for s in GOOD if s in badf])
        print(f"\n  BAD (horns,trex) mean %low = {gm:.2f}%   "
              f"GOOD (fortress,leaves,flower) mean %low = {wm:.2f}%")
        print(f"  ratio BAD/GOOD = {gm/wm:.2f}x"
              if wm > 0 else "  (GOOD mean 0)")
        print("\n  ĐỌC: ratio ≥ ~1.5 ∧ orchids sạch ⟹ (B) D2 có substrate → pilot.")
        print("       ratio ≈ 1 / orchids cũng bẩn  ⟹ (A) D2 chết, dừng.")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
