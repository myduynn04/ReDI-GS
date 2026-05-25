#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 21 — Bước 1] RoMa v2 dense init preprocess
# File: scripts/p21_roma_preprocess.py  (KEEP LOCAL — server-only)
#
# Pre-requisite (server):
#   conda activate romav2
#   cd ~/workspace/representation-3d/duyen/CoR-GS
#
# Usage:
#   SCENE=fern python scripts/p21_roma_preprocess.py            # 1 scene
#   for s in fern flower fortress horns leaves orchids room trex; do
#     SCENE=$s python scripts/p21_roma_preprocess.py
#   done
#
# Output:
#   <DATA>/<scene>/3_views/dense/fused.ply.roma   ← KHÔNG ghi đè fused.ply gốc
#   Sau verify, dùng p21_place_roma_init.py để swap (analog p18_gate2).
#
# Pipeline:
#   1. Đọc FULL COLMAP <scene>/sparse/0/{cameras.bin, images.bin}
#   2. Replicate CRSGaussian view-selection logic (LLFF eval + n_views=3 linspace)
#      → tên 3 training images
#   3. Load RoMa v2 once
#   4. Per pair (01, 02, 12):
#      - match → sample N_SAMPLE matches
#      - to_pixel_coordinates → COLMAP pixel coord system
#      - triangulate via cv2.triangulatePoints
#      - filter: cheirality (z>0 cả 2 cam) + reproj error ≤ TAU_REPROJ
#   5. Concat 3 pairs → 3D points + RGB từ ảnh A bilinear sample
#   6. Write .ply (format khớp CRSGaussian storePly: x,y,z,nx,ny,nz,r,g,b)
#
# Pre-registered (Phase 21 Bước 1):
#   N_SAMPLE       = 10000  (match Phase 18 PDCNet+ default)
#   TAU_REPROJ     = 2.0 px (standard COLMAP threshold)
#   KHÔNG τ_std manual filter (smoke verify sampler đã filter trước)
#   KHÔNG geometric verification ngoài cheirality + reproj
# ============================================================
"""RoMa v2 preprocess — 1 scene → dense init ply for CRSGaussian."""

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
# Env vars + constants
# ─────────────────────────────────────────────────────────────
SCENE       = os.environ.get("SCENE", "fern")
DATA_ROOT   = os.environ.get("DATA_ROOT", "data/nerf_llff_data")
IMG_DIR     = os.environ.get("IMG_DIR", "images")   # COLMAP-native res (match cameras.bin)
N_VIEWS     = int(os.environ.get("N_VIEWS", "3"))
LLFFHOLD    = int(os.environ.get("LLFFHOLD", "8"))
N_SAMPLE    = int(os.environ.get("N_SAMPLE", "10000"))
TAU_REPROJ  = float(os.environ.get("TAU_REPROJ", "2.0"))   # pixels


# ─────────────────────────────────────────────────────────────
# [CRSGaussian P21] COLMAP binary readers — copied from scene/colmap_loader.py
# (env romav2 không có CRSGaussian source path — copy minimal cho self-contained)
# ─────────────────────────────────────────────────────────────
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Img    = collections.namedtuple("Img",    ["id", "qvec", "tvec", "camera_id", "name"])

CAMERA_MODEL_NUM_PARAMS = {
    0: 3,   # SIMPLE_PINHOLE     (f, cx, cy)
    1: 4,   # PINHOLE            (fx, fy, cx, cy)
    2: 4,   # SIMPLE_RADIAL      (f, cx, cy, k)
    3: 5,   # RADIAL
    4: 8,   # OPENCV
}
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
            # Skip xys + point3D_ids
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
    """COLMAP cam.params → 3×3 K (PINHOLE / SIMPLE_PINHOLE only — assert other types)."""
    if cam.model == "PINHOLE":
        fx, fy, cx, cy = cam.params
    elif cam.model == "SIMPLE_PINHOLE":
        f, cx, cy = cam.params
        fx = fy = f
    else:
        # SIMPLE_RADIAL / RADIAL / OPENCV — ignore distortion (LLFF undistorted by COLMAP)
        # cam.params[0:3] = (f, cx, cy) hoặc (fx, fy, cx, cy) tuỳ model
        if cam.model == "SIMPLE_RADIAL":
            f, cx, cy = cam.params[:3]; fx = fy = f
        else:
            fx, fy, cx, cy = cam.params[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


# ─────────────────────────────────────────────────────────────
# [CRSGaussian P21] View selection — replicate readColmapSceneInfo logic
# scene/dataset_readers.py:355-365 (LLFF eval + n_views linspace)
# ─────────────────────────────────────────────────────────────
def select_training_views(extr_by_name: dict, llffhold: int, n_views: int):
    """Return list of n_views Img objects, ordered by image_name."""
    sorted_names = sorted(extr_by_name.keys())
    # eval mode: drop every llffhold-th as test
    train_pool = [n for idx, n in enumerate(sorted_names) if idx % llffhold != 0]
    if n_views > 0:
        idx_sub = np.linspace(0, len(train_pool) - 1, n_views)
        idx_sub = [int(round(i)) for i in idx_sub]
        sel_names = [train_pool[i] for i in idx_sub]
    else:
        sel_names = train_pool
    return [extr_by_name[n] for n in sel_names]


# ─────────────────────────────────────────────────────────────
# [CRSGaussian P21] Triangulation + filter
# ─────────────────────────────────────────────────────────────
def triangulate_pair(K_A, R_A, t_A, K_B, R_B, t_B, kp_A, kp_B):
    """Return (xyz Nx3, mask N) after cheirality + reproj filter."""
    P_A = K_A @ np.hstack([R_A, t_A.reshape(3, 1)])
    P_B = K_B @ np.hstack([R_B, t_B.reshape(3, 1)])
    pts4D = cv2.triangulatePoints(P_A, P_B, kp_A.T, kp_B.T)   # 4 x N
    xyz = (pts4D[:3] / pts4D[3:4]).T                          # N x 3

    # Cheirality — z > 0 trong cả 2 camera frames
    z_A = (R_A @ xyz.T + t_A.reshape(3, 1))[2]
    z_B = (R_B @ xyz.T + t_B.reshape(3, 1))[2]
    chir = (z_A > 0) & (z_B > 0)

    # Reproj error — mỗi camera
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
    """img (H, W, 3) uint8; kp (N, 2) pixel xy → rgb (N, 3) uint8."""
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


# ─────────────────────────────────────────────────────────────
# [CRSGaussian P21] storePly — EXACT copy CRSGaussian dataset_readers.py
# format khớp fetchPly: x,y,z,nx=0,ny=0,nz=0,red,green,blue
# ─────────────────────────────────────────────────────────────
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
# Main
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(f"[P21 preprocess] scene={SCENE}  N_VIEWS={N_VIEWS}  N_SAMPLE={N_SAMPLE}  τ_reproj={TAU_REPROJ}")
    print("=" * 70)

    scene_dir = Path(DATA_ROOT) / SCENE
    sparse_dir = scene_dir / "sparse/0"
    img_dir = scene_dir / IMG_DIR
    out_dir = scene_dir / f"{N_VIEWS}_views/dense"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ply = out_dir / "fused.ply.roma"

    # 1. COLMAP load
    cams = read_intrinsics_binary(sparse_dir / "cameras.bin")
    extrs = read_extrinsics_binary(sparse_dir / "images.bin")
    extr_by_name = {im.name: im for im in extrs.values()}
    print(f"[P21] COLMAP: {len(cams)} cameras, {len(extrs)} images")

    # 2. View selection — replicate CRSGaussian logic
    sel = select_training_views(extr_by_name, LLFFHOLD, N_VIEWS)
    print(f"[P21] selected {N_VIEWS} training views:")
    for i, im in enumerate(sel):
        print(f"  view {i}: {im.name}")

    # Verify images exist + size matches COLMAP cam
    imgs_np, paths = [], []
    for im in sel:
        p = img_dir / im.name
        if not p.is_file():
            # try other extensions
            stem = Path(im.name).stem
            for ext in (".png", ".jpg", ".JPG", ".PNG", ".jpeg"):
                alt = img_dir / (stem + ext)
                if alt.is_file():
                    p = alt; break
        if not p.is_file():
            raise SystemExit(f"[P21] image KHÔNG tồn tại: {p}")
        pil = PILImage.open(p).convert("RGB")
        imgs_np.append(np.array(pil))
        paths.append(p)
        cam = cams[im.camera_id]
        if pil.size != (cam.width, cam.height):
            print(f"  ⚠ view {im.name}: PIL size {pil.size} ≠ COLMAP {(cam.width, cam.height)} "
                  f"→ kiểm tra IMG_DIR (hiện = {IMG_DIR})")
    print(f"[P21] images loaded: {[p.name for p in paths]}  size={imgs_np[0].shape[:2][::-1]} (W,H)")

    # 3. Load RoMa v2 once
    import torch
    from romav2 import RoMaV2
    t0 = time.time()
    model = RoMaV2()
    model.apply_setting("precise")
    print(f"[P21] RoMa v2 init: {time.time()-t0:.1f}s")
    torch.cuda.reset_peak_memory_stats()

    # 4. 3 pairs: (0,1), (0,2), (1,2)
    pairs = [(0, 1), (0, 2), (1, 2)]
    all_xyz, all_rgb, totals = [], [], []
    t_pair = time.time()
    for ia, ib in pairs:
        im_a, im_b = sel[ia], sel[ib]
        path_a, path_b = paths[ia], paths[ib]
        cam_a, cam_b = cams[im_a.camera_id], cams[im_b.camera_id]
        H_a, W_a = imgs_np[ia].shape[:2]
        H_b, W_b = imgs_np[ib].shape[:2]

        # RoMa match + sample
        preds = model.match(str(path_a), str(path_b))
        m, ov, pAB, pBA = model.sample(preds, N_SAMPLE)
        kp_a, kp_b = model.to_pixel_coordinates(m, H_a, W_a, H_b, W_b)
        kp_a = kp_a.detach().cpu().numpy().astype(np.float64)
        kp_b = kp_b.detach().cpu().numpy().astype(np.float64)

        # Triangulate
        K_a, K_b = cam_K(cam_a), cam_K(cam_b)
        R_a, t_a = qvec2rotmat(im_a.qvec), im_a.tvec
        R_b, t_b = qvec2rotmat(im_b.qvec), im_b.tvec
        xyz, mask, err_a, err_b = triangulate_pair(K_a, R_a, t_a, K_b, R_b, t_b, kp_a, kp_b)

        # RGB từ ảnh A
        rgb = sample_rgb_bilinear(imgs_np[ia], kp_a)

        kept = int(mask.sum())
        med_err = np.median(np.maximum(err_a, err_b)[mask]) if kept > 0 else float("nan")
        totals.append(kept)
        print(f"  pair ({ia},{ib}) [{im_a.name} ↔ {im_b.name}]: "
              f"{kept}/{N_SAMPLE} kept ({100*kept/N_SAMPLE:.1f}%)  "
              f"median reproj err={med_err:.3f}px")

        all_xyz.append(xyz[mask])
        all_rgb.append(rgb[mask])

    print(f"[P21] all pairs done: {time.time()-t_pair:.1f}s")

    # 5. Concat + write
    xyz_all = np.vstack(all_xyz)
    rgb_all = np.vstack(all_rgb)
    print(f"[P21] total points: {len(xyz_all)} (= sum {totals})")
    store_ply(str(out_ply), xyz_all, rgb_all)
    print(f"[P21] wrote {out_ply} ({out_ply.stat().st_size/1024:.1f} KB)")

    # 6. Stats
    print(f"[P21] xyz bbox:")
    print(f"  min  = {xyz_all.min(axis=0)}")
    print(f"  max  = {xyz_all.max(axis=0)}")
    print(f"  mean = {xyz_all.mean(axis=0)}")
    print(f"[P21] VRAM peak: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
    print(f"[P21] DONE — {SCENE} → {out_ply}")


if __name__ == "__main__":
    main()
