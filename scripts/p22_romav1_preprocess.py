#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 22 — Bước 1] RoMa v1 dense init preprocess
# File: scripts/p22_romav1_preprocess.py  (KEEP LOCAL — server-only)
#
# Pre-requisite (server):
#   conda activate roma_v1
#   cd ~/workspace/representation-3d/duyen/CoR-GS
#
# Usage:
#   SCENE=fern python scripts/p22_romav1_preprocess.py
#
# Output:
#   <DATA>/<scene>/3_views/dense/fused.ply.romav1   ← KHÔNG đè .ply.roma (v2)
#
# Khác Phase 21 (v2):
#   - API: `roma_outdoor(use_custom_corr=False)` thay `RoMaV2()`
#   - Match returns `(warp, certainty)` thay preds dict
#   - Sample returns `(matches, certainty)` (2-tuple) thay 4-tuple
#   - KHÔNG có per-pixel precision → chỉ dùng sampler tự filter
#   - Fallback pure-PyTorch kernel (use_custom_corr=False) — đã verify smoke
#     1.11s/pair + 4.4GB VRAM (NHANH HƠN + VRAM THẤP HƠN v2 trên H100)
#
# Pre-registered (Phase 22 cùng Phase 21):
#   N_SAMPLE       = 10000
#   TAU_REPROJ     = 2.0 px
#   KHÔNG filter thêm (sampler tự lọc)
# ============================================================
"""RoMa v1 preprocess — 1 scene → dense init ply for CRSGaussian Phase 22."""

import os
import sys
import time
import struct
import collections
from pathlib import Path

import numpy as np
import cv2
from PIL import Image as PILImage

# ─────────────────────────────────────────────────────────────
SCENE       = os.environ.get("SCENE", "fern")
DATA_ROOT   = os.environ.get("DATA_ROOT", "data/nerf_llff_data")
IMG_DIR     = os.environ.get("IMG_DIR", "images")
N_VIEWS     = int(os.environ.get("N_VIEWS", "3"))
LLFFHOLD    = int(os.environ.get("LLFFHOLD", "8"))
N_SAMPLE    = int(os.environ.get("N_SAMPLE", "10000"))
TAU_REPROJ  = float(os.environ.get("TAU_REPROJ", "2.0"))


# ─────────────────────────────────────────────────────────────
# [CRSGaussian P22] COLMAP binary readers — verbatim copy từ p21
# ─────────────────────────────────────────────────────────────
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Img    = collections.namedtuple("Img",    ["id", "qvec", "tvec", "camera_id", "name"])

CAMERA_MODEL_NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}
CAMERA_MODEL_NAMES = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL", 4: "OPENCV"}


def _read_bytes(fid, n, fmt):
    return struct.unpack("<" + fmt, fid.read(n))


def read_intrinsics_binary(path):
    cams = {}
    with open(path, "rb") as fid:
        (n_cam,) = _read_bytes(fid, 8, "Q")
        for _ in range(n_cam):
            cam_id, model_id, w, h = _read_bytes(fid, 24, "iiQQ")
            n_p = CAMERA_MODEL_NUM_PARAMS[model_id]
            params = _read_bytes(fid, 8 * n_p, "d" * n_p)
            cams[cam_id] = Camera(cam_id, CAMERA_MODEL_NAMES[model_id], w, h, np.array(params))
    return cams


def read_extrinsics_binary(path):
    imgs = {}
    with open(path, "rb") as fid:
        (n_img,) = _read_bytes(fid, 8, "Q")
        for _ in range(n_img):
            props = _read_bytes(fid, 64, "idddddddi")
            img_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            cam_id = props[8]
            name = ""
            while True:
                (c,) = _read_bytes(fid, 1, "c")
                if c == b"\x00":
                    break
                name += c.decode("utf-8")
            (n_pts2D,) = _read_bytes(fid, 8, "Q")
            _read_bytes(fid, 24 * n_pts2D, "ddq" * n_pts2D)
            imgs[img_id] = Img(img_id, qvec, tvec, cam_id, name)
    return imgs


def qvec2rotmat(q):
    return np.array([
        [1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[3]*q[1] + 2*q[0]*q[2]],
        [2*q[1]*q[2] + 2*q[0]*q[3], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[0]*q[1]],
        [2*q[3]*q[1] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], 1 - 2*q[1]**2 - 2*q[2]**2],
    ])


def cam_K(cam: Camera) -> np.ndarray:
    if cam.model == "PINHOLE":
        fx, fy, cx, cy = cam.params
    elif cam.model == "SIMPLE_PINHOLE":
        f, cx, cy = cam.params; fx = fy = f
    elif cam.model == "SIMPLE_RADIAL":
        f, cx, cy = cam.params[:3]; fx = fy = f
    else:
        fx, fy, cx, cy = cam.params[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


# ─────────────────────────────────────────────────────────────
def select_training_views(extr_by_name: dict, llffhold: int, n_views: int):
    sorted_names = sorted(extr_by_name.keys())
    train_pool = [n for idx, n in enumerate(sorted_names) if idx % llffhold != 0]
    if n_views > 0:
        idx_sub = np.linspace(0, len(train_pool) - 1, n_views)
        idx_sub = [int(round(i)) for i in idx_sub]
        sel_names = [train_pool[i] for i in idx_sub]
    else:
        sel_names = train_pool
    return [extr_by_name[n] for n in sel_names]


def triangulate_pair(K_A, R_A, t_A, K_B, R_B, t_B, kp_A, kp_B):
    # Đoạn DLT để chuyển từ 2d -> 3d
    P_A = K_A @ np.hstack([R_A, t_A.reshape(3, 1)])
    P_B = K_B @ np.hstack([R_B, t_B.reshape(3, 1)])
    pts4D = cv2.triangulatePoints(P_A, P_B, kp_A.T, kp_B.T)
    xyz = (pts4D[:3] / pts4D[3:4]).T

    z_A = (R_A @ xyz.T + t_A.reshape(3, 1))[2]
    z_B = (R_B @ xyz.T + t_B.reshape(3, 1))[2]
    chir = (z_A > 0) & (z_B > 0)

    proj_A = P_A @ np.hstack([xyz, np.ones((len(xyz), 1))]).T
    proj_A = (proj_A[:2] / proj_A[2:3]).T
    proj_B = P_B @ np.hstack([xyz, np.ones((len(xyz), 1))]).T
    proj_B = (proj_B[:2] / proj_B[2:3]).T
    err_A = np.linalg.norm(proj_A - kp_A, axis=1)
    err_B = np.linalg.norm(proj_B - kp_B, axis=1)
    rep_ok = (err_A <= TAU_REPROJ) & (err_B <= TAU_REPROJ)

    mask = chir & rep_ok
    return xyz, mask, err_A, err_B


def sample_rgb_bilinear(img_np: np.ndarray, kp: np.ndarray) -> np.ndarray:
    H, W = img_np.shape[:2]
    x = np.clip(kp[:, 0], 0, W - 1)
    y = np.clip(kp[:, 1], 0, H - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.clip(x0 + 1, 0, W - 1), np.clip(y0 + 1, 0, H - 1)
    wx, wy = x - x0, y - y0
    rgb = (
        img_np[y0, x0].astype(np.float32) * ((1 - wx) * (1 - wy))[:, None] +
        img_np[y0, x1].astype(np.float32) * (wx       * (1 - wy))[:, None] +
        img_np[y1, x0].astype(np.float32) * ((1 - wx) * wy      )[:, None] +
        img_np[y1, x1].astype(np.float32) * (wx       * wy      )[:, None]
    )
    return np.clip(rgb, 0, 255).astype(np.uint8)


def store_ply(path, xyz, rgb):
    from plyfile import PlyData, PlyElement
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(xyz, dtype=np.float32)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate(
        (xyz.astype(np.float32), normals, rgb.astype(np.uint8)),
        axis=1,
    )
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)


# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(f"[P22 v1 preprocess] scene={SCENE}  N_VIEWS={N_VIEWS}  N_SAMPLE={N_SAMPLE}  τ_reproj={TAU_REPROJ}")
    print("=" * 70)

    scene_dir = Path(DATA_ROOT) / SCENE
    sparse_dir = scene_dir / "sparse/0"
    img_dir = scene_dir / IMG_DIR
    out_dir = scene_dir / f"{N_VIEWS}_views/dense"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ply = out_dir / "fused.ply.romav1"

    # COLMAP load
    cams = read_intrinsics_binary(sparse_dir / "cameras.bin")
    extrs = read_extrinsics_binary(sparse_dir / "images.bin")
    extr_by_name = {im.name: im for im in extrs.values()}
    print(f"[P22] COLMAP: {len(cams)} cameras, {len(extrs)} images")

    # View selection — same logic Phase 21
    sel = select_training_views(extr_by_name, LLFFHOLD, N_VIEWS)
    print(f"[P22] selected {N_VIEWS} training views:")
    for i, im in enumerate(sel):
        print(f"  view {i}: {im.name}")

    # Load images
    imgs_np, paths = [], []
    for im in sel:
        p = img_dir / im.name
        if not p.is_file():
            stem = Path(im.name).stem
            for ext in (".png", ".jpg", ".JPG", ".PNG", ".jpeg"):
                alt = img_dir / (stem + ext)
                if alt.is_file():
                    p = alt; break
        if not p.is_file():
            raise SystemExit(f"[P22] image MISSING: {p}")
        pil = PILImage.open(p).convert("RGB")
        imgs_np.append(np.array(pil))
        paths.append(p)
        cam = cams[im.camera_id]
        if pil.size != (cam.width, cam.height):
            print(f"  ⚠ {im.name}: PIL size {pil.size} ≠ COLMAP {(cam.width, cam.height)} — check IMG_DIR")
    print(f"[P22] images loaded: size={imgs_np[0].shape[:2][::-1]} (W,H)")

    # ── Load RoMa v1 — fallback PyTorch kernel ──
    import torch
    from romatch import roma_outdoor

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    # use_custom_corr=False → pure-PyTorch fallback (đã verify smoke 1.11s/4.4GB)
    model = roma_outdoor(device=device, use_custom_corr=False)
    print(f"[P22] RoMa v1 init: {time.time()-t0:.1f}s")
    torch.cuda.reset_peak_memory_stats()

    # ── All pairs C(N_VIEWS, 2) — dynamic, work với mọi N ──
    # N=3 → 3 pairs (same as hardcode cũ).
    # N=6 → 15 pairs. N=9 → 36 pairs. Cost ~1.11s/pair on H100.
    from itertools import combinations
    pairs = list(combinations(range(N_VIEWS), 2))
    print(f"[P22] N_VIEWS={N_VIEWS} → {len(pairs)} pairs (C({N_VIEWS},2))")
    all_xyz, all_rgb, totals = [], [], []
    t_pair = time.time()
    for ia, ib in pairs:
        im_a, im_b = sel[ia], sel[ib]
        path_a, path_b = paths[ia], paths[ib]
        cam_a, cam_b = cams[im_a.camera_id], cams[im_b.camera_id]
        H_a, W_a = imgs_np[ia].shape[:2]
        H_b, W_b = imgs_np[ib].shape[:2]

        # v1 match returns (warp, certainty) — KHÁC v2 (preds dict)
        # Đoạn dùng certainty map từ roma 
        warp, certainty = model.match(str(path_a), str(path_b), device=device)
        # v1 sample returns (matches, certainty) — 2-tuple KHÁC v2 (4-tuple)
        matches, sampled_certainty = model.sample(warp, certainty, num=N_SAMPLE)
        kp_a, kp_b = model.to_pixel_coordinates(matches, H_a, W_a, H_b, W_b)
        kp_a = kp_a.detach().cpu().numpy().astype(np.float64)
        kp_b = kp_b.detach().cpu().numpy().astype(np.float64)

        # Triangulate (identical Phase 21)
        K_a, K_b = cam_K(cam_a), cam_K(cam_b)
        R_a, t_a = qvec2rotmat(im_a.qvec), im_a.tvec
        R_b, t_b = qvec2rotmat(im_b.qvec), im_b.tvec
        xyz, mask, err_a, err_b = triangulate_pair(K_a, R_a, t_a, K_b, R_b, t_b, kp_a, kp_b)
        # ⭐ LẤY MÀU — đọc ảnh A tại pixel kp_a
        rgb = sample_rgb_bilinear(imgs_np[ia], kp_a)

        kept = int(mask.sum())
        med_err = np.median(np.maximum(err_a, err_b)[mask]) if kept > 0 else float("nan")
        cert_mean = float(sampled_certainty.mean().item())
        totals.append(kept)
        print(f"  pair ({ia},{ib}) [{im_a.name} ↔ {im_b.name}]: "
              f"{kept}/{len(kp_a)} kept ({100*kept/max(len(kp_a),1):.1f}%)  "
              f"med reproj={med_err:.3f}px  mean certainty={cert_mean:.3f}")

        all_xyz.append(xyz[mask])
        all_rgb.append(rgb[mask])

    print(f"[P22] all pairs done: {time.time()-t_pair:.1f}s")

    xyz_all = np.vstack(all_xyz)
    rgb_all = np.vstack(all_rgb)
    print(f"[P22] total points: {len(xyz_all)} (= sum {totals})")
    store_ply(str(out_ply), xyz_all, rgb_all)
    print(f"[P22] wrote {out_ply} ({out_ply.stat().st_size/1024:.1f} KB)")

    print(f"[P22] xyz bbox:")
    print(f"  min  = {xyz_all.min(axis=0)}")
    print(f"  max  = {xyz_all.max(axis=0)}")
    print(f"  mean = {xyz_all.mean(axis=0)}")
    print(f"[P22] VRAM peak: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
    print(f"[P22] DONE — {SCENE} → {out_ply}")


if __name__ == "__main__":
    main()
