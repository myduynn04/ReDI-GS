#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 37 — S1] RoMa v1 preprocess + GIỮ LẠI Q_init
# File: scripts/p37_romav1_preprocess_qinit.py  (KEEP LOCAL — server-only)
#
# Mục đích:
#   Bản sao p22_romav1_preprocess.py, KHÁC DUY NHẤT ở chỗ KHÔNG vứt
#   hai đại lượng mà RoMa + triangulation đã tính ra:
#     c_i = sampled_certainty[i]        ∈ [0,1]  độ chắc chắn của match
#     e_i = max(err_a[i], err_b[i])     px       reprojection error
#   → ghi ra sidecar .npy để CRS dùng sau (H1b).
#
# ⚠ KHÔNG SỬA p22_romav1_preprocess.py — giữ nguyên để reproduce Phase 22.
#   Script này sinh ra ply BYTE-IDENTICAL với p22 (cùng seed, cùng logic),
#   chỉ THÊM file .npy bên cạnh. Verify bằng cmp nếu cần.
#
# Pre-requisite (server):
#   conda activate roma_v1
#   cd ~/workspace/representation-3d/duyen/CoR-GS
#
# Usage:
#   SCENE=fern python scripts/p37_romav1_preprocess_qinit.py
#
# Output:
#   <DATA>/<scene>/3_views/dense/fused.ply.romav1_p37       ← ply (giống p22)
#   <DATA>/<scene>/3_views/dense/fused.romav1.qinit.npz     ← MỚI, sidecar
#
#   Sidecar npz chứa (mọi mảng CÙNG THỨ TỰ với vertex trong ply):
#     q_cert   (N,) float32  certainty đã lọc theo mask
#     q_rep    (N,) float32  1 - clip(e/TAU_REPROJ, 0, 1)
#     q_init   (N,) float32  w_c*q_cert + w_e*q_rep
#     raw_err  (N,) float32  e_i thô (px) — giữ để chẩn đoán
#     cert_all (M,) float32  certainty TRƯỚC mask (M ≥ N) — để tách nguyên nhân
#                            phương sai bị giết bởi sampler hay bởi mask triangulation
#
# Pre-registered (Phase 37 S1), khớp Phase 22:
#   N_SAMPLE   = 10000
#   TAU_REPROJ = 2.0 px
#   W_CERT     = 0.5
#   W_REPROJ   = 0.5
# ============================================================
"""RoMa v1 preprocess + Q_init sidecar — Phase 37 S1 data gate."""

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

# [CRSGaussian P37] Trọng số gộp Q_init — pre-registered 0.5/0.5
W_CERT      = float(os.environ.get("W_CERT", "0.5"))
W_REPROJ    = float(os.environ.get("W_REPROJ", "0.5"))


# ─────────────────────────────────────────────────────────────
# COLMAP binary readers — verbatim copy từ p22 (KHÔNG sửa logic)
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
    """DLT triangulation — verbatim copy p22. Trả về xyz, mask, err_A, err_B."""
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


# ============================================================
# [CRSGaussian P37] describe_dist — thống kê gọn cho 1 mảng
# Mục đích: in nhanh phân phối để đánh giá cửa S1 ngay tại chỗ,
#           không cần chờ script tổng hợp.
# ============================================================
def describe_dist(name: str, a: np.ndarray) -> dict:
    if a.size == 0:
        print(f"    {name:<10} EMPTY")
        return {}
    q = np.percentile(a, [1, 5, 25, 50, 75, 95, 99])
    d = dict(n=int(a.size), mean=float(a.mean()), std=float(a.std()),
             p01=float(q[0]), p05=float(q[1]), p25=float(q[2]), p50=float(q[3]),
             p75=float(q[4]), p95=float(q[5]), p99=float(q[6]),
             vmin=float(a.min()), vmax=float(a.max()))
    print(f"    {name:<10} n={d['n']:<7} mean={d['mean']:.4f} std={d['std']:.4f} "
          f"| p05={d['p05']:.3f} p50={d['p50']:.3f} p95={d['p95']:.3f} "
          f"| range=[{d['vmin']:.3f}, {d['vmax']:.3f}]")
    return d


# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print(f"[P37 S1] scene={SCENE}  N_VIEWS={N_VIEWS}  N_SAMPLE={N_SAMPLE}  "
          f"τ_reproj={TAU_REPROJ}  w_cert={W_CERT}  w_rep={W_REPROJ}")
    print("=" * 78)

    scene_dir = Path(DATA_ROOT) / SCENE
    sparse_dir = scene_dir / "sparse/0"
    img_dir = scene_dir / IMG_DIR
    out_dir = scene_dir / f"{N_VIEWS}_views/dense"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ply = out_dir / "fused.ply.romav1_p37"
    out_npz = out_dir / "fused.romav1.qinit.npz"

    # COLMAP load
    cams = read_intrinsics_binary(sparse_dir / "cameras.bin")
    extrs = read_extrinsics_binary(sparse_dir / "images.bin")
    extr_by_name = {im.name: im for im in extrs.values()}
    print(f"[P37] COLMAP: {len(cams)} cameras, {len(extrs)} images")

    sel = select_training_views(extr_by_name, LLFFHOLD, N_VIEWS)
    print(f"[P37] selected {N_VIEWS} training views:")
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
            raise SystemExit(f"[P37] image MISSING: {p}")
        pil = PILImage.open(p).convert("RGB")
        imgs_np.append(np.array(pil))
        paths.append(p)
    print(f"[P37] images loaded: size={imgs_np[0].shape[:2][::-1]} (W,H)")

    # ── Load RoMa v1 — fallback PyTorch kernel (giống p22) ──
    import torch
    from romatch import roma_outdoor

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    model = roma_outdoor(device=device, use_custom_corr=False)
    print(f"[P37] RoMa v1 init: {time.time()-t0:.1f}s")
    torch.cuda.reset_peak_memory_stats()

    from itertools import combinations
    pairs = list(combinations(range(N_VIEWS), 2))
    print(f"[P37] N_VIEWS={N_VIEWS} → {len(pairs)} pairs")

    # [CRSGaussian P37] Accumulator — MỌI list phải được append CÙNG mask,
    # CÙNG thứ tự, để index của npz khớp chính xác index vertex của ply.
    all_xyz, all_rgb = [], []
    all_qcert, all_qrep, all_rawerr = [], [], []
    all_cert_prefilter = []            # certainty TRƯỚC mask — chẩn đoán nguyên nhân
    totals = []

    t_pair = time.time()
    for ia, ib in pairs:
        im_a, im_b = sel[ia], sel[ib]
        path_a, path_b = paths[ia], paths[ib]
        cam_a, cam_b = cams[im_a.camera_id], cams[im_b.camera_id]
        H_a, W_a = imgs_np[ia].shape[:2]
        H_b, W_b = imgs_np[ib].shape[:2]

        warp, certainty = model.match(str(path_a), str(path_b), device=device)
        matches, sampled_certainty = model.sample(warp, certainty, num=N_SAMPLE)
        kp_a, kp_b = model.to_pixel_coordinates(matches, H_a, W_a, H_b, W_b)
        kp_a = kp_a.detach().cpu().numpy().astype(np.float64)
        kp_b = kp_b.detach().cpu().numpy().astype(np.float64)

        # ── [CRSGaussian P37] ĐÂY LÀ THAY ĐỔI DUY NHẤT vs p22 ──
        # p22 chỉ lấy .mean() rồi in. Ở đây giữ nguyên vector per-point.
        # .reshape(-1) phòng trường hợp RoMa trả (N,1) thay vì (N,).
        cert_np = sampled_certainty.detach().cpu().numpy().astype(np.float32).reshape(-1)
        if cert_np.shape[0] != kp_a.shape[0]:
            raise SystemExit(
                f"[P37] FATAL: certainty len {cert_np.shape[0]} != matches len {kp_a.shape[0]}. "
                f"API RoMa đã đổi — dừng thay vì ghi sidecar lệch thứ tự."
            )

        K_a, K_b = cam_K(cam_a), cam_K(cam_b)
        R_a, t_a = qvec2rotmat(im_a.qvec), im_a.tvec
        R_b, t_b = qvec2rotmat(im_b.qvec), im_b.tvec
        xyz, mask, err_a, err_b = triangulate_pair(K_a, R_a, t_a, K_b, R_b, t_b, kp_a, kp_b)
        rgb = sample_rgb_bilinear(imgs_np[ia], kp_a)

        # e_i = lỗi tệ nhất trong hai view (conservative)
        err_max = np.maximum(err_a, err_b).astype(np.float32)
        # q_rep: chuẩn hoá về [0,1] theo cùng công thức q_reproj cũ của dự án.
        # LƯU Ý: điểm sống sót đều có err ≤ TAU_REPROJ (mask đã lọc), nên
        #        q_rep ∈ [0,1] tự động — phương sai phụ thuộc phân bố err trong [0, τ].
        q_rep = 1.0 - np.clip(err_max / TAU_REPROJ, 0.0, 1.0)

        kept = int(mask.sum())
        totals.append(kept)

        all_xyz.append(xyz[mask])
        all_rgb.append(rgb[mask])
        all_qcert.append(cert_np[mask])
        all_qrep.append(q_rep[mask])
        all_rawerr.append(err_max[mask])
        all_cert_prefilter.append(cert_np)          # TRƯỚC mask, để so sánh

        med_err = np.median(err_max[mask]) if kept > 0 else float("nan")
        print(f"  pair ({ia},{ib}) [{im_a.name} ↔ {im_b.name}]: "
              f"{kept}/{len(kp_a)} kept ({100*kept/max(len(kp_a),1):.1f}%)  "
              f"med reproj={med_err:.3f}px  mean certainty={cert_np[mask].mean():.4f}")

    print(f"[P37] all pairs done: {time.time()-t_pair:.1f}s")

    # ── Gộp ──
    xyz_all  = np.vstack(all_xyz)
    rgb_all  = np.vstack(all_rgb)
    qcert    = np.concatenate(all_qcert).astype(np.float32)
    qrep     = np.concatenate(all_qrep).astype(np.float32)
    rawerr   = np.concatenate(all_rawerr).astype(np.float32)
    certpre  = np.concatenate(all_cert_prefilter).astype(np.float32)

    # Q_init = tổ hợp tuyến tính, pre-registered 0.5/0.5
    qinit = (W_CERT * qcert + W_REPROJ * qrep).astype(np.float32)

    # Sanity: sidecar phải khớp chiều dài ply, nếu không thì index sẽ lệch
    assert len(qinit) == len(xyz_all), \
        f"[P37] FATAL: qinit {len(qinit)} != xyz {len(xyz_all)}"

    print(f"[P37] total points: {len(xyz_all)} (= sum {totals})")

    # ── Ghi ply (giống p22) + sidecar npz (MỚI) ──
    store_ply(str(out_ply), xyz_all, rgb_all)
    np.savez_compressed(
        str(out_npz),
        q_cert=qcert, q_rep=qrep, q_init=qinit,
        raw_err=rawerr, cert_all=certpre,
        w_cert=np.float32(W_CERT), w_reproj=np.float32(W_REPROJ),
        tau_reproj=np.float32(TAU_REPROJ), n_sample=np.int32(N_SAMPLE),
    )
    print(f"[P37] wrote {out_ply} ({out_ply.stat().st_size/1024:.1f} KB)")
    print(f"[P37] wrote {out_npz} ({out_npz.stat().st_size/1024:.1f} KB)")

    # ── S1 GATE — in phân phối ngay, không cần chờ script tổng hợp ──
    print()
    print(f"[P37 S1 STATS] scene={SCENE}")
    describe_dist("cert_pre",  certpre)   # trước mask triangulation
    describe_dist("q_cert",    qcert)     # sau mask — đây là cái CRS sẽ dùng
    describe_dist("q_rep",     qrep)
    describe_dist("q_init",    qinit)
    describe_dist("raw_err_px", rawerr)

    # Cảnh báo tại chỗ theo tiêu chí đăng ký trước
    if float(qinit.std()) < 0.05:
        print(f"  ⚠️  [S1 WARN] std(q_init)={qinit.std():.4f} < 0.05 — NGUY CƠ SUY BIẾN")
    if float(qcert.std()) < 0.05:
        print(f"  ⚠️  [S1 WARN] std(q_cert)={qcert.std():.4f} < 0.05 — sampler đã lọc hết phương sai?")

    print(f"[P37] VRAM peak: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
    print(f"[P37] DONE — {SCENE}")


if __name__ == "__main__":
    main()
