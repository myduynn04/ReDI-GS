#!/usr/bin/env python
# ============================================================
# [CRSGaussian Phase 10A] Pre-compute DUSt3R dense point cloud per scene.
# File: scripts/precompute_dust3r.py (NEW)
#
# Mục đích: Chạy 1 lần trên server có DUSt3R env (PyTorch + dust3r repo).
#           Output: cache/dust3r_init/<scene>.npz (points/colors/confidence)
#           cho mỗi LLFF scene. Training-time sẽ chỉ load cache, không
#           inference DUSt3R nữa → no per-iter overhead.
#
# Pipeline cho mỗi scene:
#   1. Read COLMAP poses (extrinsics + intrinsics)
#   2. Re-create train split (eval + n_views subsample) — match
#      đúng logic của readColmapSceneInfo trong scene/dataset_readers.py
#   3. Load 3 train images at size=512 qua DUSt3R load_images
#   4. Build complete pair graph (3C2 = 3 pairs, symmetrize → 6)
#   5. inference(pairs, model, cuda)
#   6. global_aligner(out, mode=PointCloudOptimizer)
#   7. preset_pose(C2W list từ COLMAP)
#   8. compute_global_alignment(init='known_poses', niter=300, lr=0.01)
#   9. Extract get_pts3d() + imgs + im_conf, mask theo conf > min_conf
#  10. Save npz
#
# Run example (server-side):
#   conda activate dust3r
#   cd <CRSGaussian_root>
#   python scripts/precompute_dust3r.py \
#       --dust3r_path external/dust3r \
#       --dust3r_ckpt external/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
#       --data_root data/nerf_llff_data \
#       --scenes fern flower fortress horns leaves orchids room trex \
#       --output cache/dust3r_init
# ============================================================

import argparse
import glob
import os
import sys
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Precompute DUSt3R dense PC per LLFF scene.")
    p.add_argument("--dust3r_path", type=str, required=True,
                   help="Path tới DUSt3R repo (vd. external/dust3r). Sẽ thêm vào sys.path.")
    p.add_argument("--dust3r_ckpt", type=str, required=True,
                   help="Path tới DUSt3R checkpoint .pth.")
    p.add_argument("--data_root", type=str, required=True,
                   help="Root của LLFF data (vd. data/nerf_llff_data).")
    p.add_argument("--scenes", type=str, nargs="+", required=True,
                   help="List scene names (vd. fern flower ...).")
    p.add_argument("--output", type=str, required=True,
                   help="Output cache dir (vd. cache/dust3r_init).")
    p.add_argument("--n_views", type=int, default=3,
                   help="Số train views (match với train.py setting).")
    p.add_argument("--llffhold", type=int, default=8,
                   help="Eval split holdout (match dataset_readers).")
    p.add_argument("--image_size", type=int, default=512,
                   help="DUSt3R input size (long edge).")
    p.add_argument("--niter", type=int, default=300,
                   help="Global alignment iterations.")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--min_conf_thr", type=float, default=1.5,
                   help="Mask points có log-conf < threshold (lưu vẫn lưu, lọc khi load).")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def setup_paths(dust3r_path):
    """Thêm DUSt3R repo vào sys.path. KHÔNG add CRSGaussian root vì
    `scene/__init__.py` import `dataset_readers` (cần imageio + open3d
    + nhiều deps heavy) — dust3r env không có. Thay vào đó dùng
    importlib.util để load `scene/colmap_loader.py` trực tiếp ở
    `_load_colmap_loader()`."""
    dust3r_path = os.path.abspath(dust3r_path)
    if not os.path.isdir(dust3r_path):
        raise FileNotFoundError(f"--dust3r_path không tồn tại: {dust3r_path}")
    sys.path.insert(0, dust3r_path)


def _load_colmap_loader():
    """Load `scene/colmap_loader.py` standalone, bypass `scene/__init__.py`
    (vốn drag heavy CRSGaussian deps không có trong dust3r env)."""
    import importlib.util
    crs_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cl_path = os.path.join(crs_root, "scene", "colmap_loader.py")
    if not os.path.isfile(cl_path):
        raise FileNotFoundError(f"colmap_loader không tồn tại: {cl_path}")
    spec = importlib.util.spec_from_file_location("_colmap_loader_standalone", cl_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_colmap_train_cams(scene_dir, n_views, llffhold, images_subdir="images"):
    """Tái tạo logic của readColmapSceneInfo:
       1. Read COLMAP cams sorted by image_name
       2. eval split: train if idx % llffhold != 0
       3. n_views subsample qua linspace

    Returns: list các dict {image_path, R_w2c (3x3), T (3,), focal_x, focal_y, h, w}
    """
    cl = _load_colmap_loader()
    qvec2rotmat = cl.qvec2rotmat

    sparse = os.path.join(scene_dir, "sparse", "0")
    try:
        ext = cl.read_extrinsics_binary(os.path.join(sparse, "images.bin"))
        intr = cl.read_intrinsics_binary(os.path.join(sparse, "cameras.bin"))
    except Exception:
        ext = cl.read_extrinsics_text(os.path.join(sparse, "images.txt"))
        intr = cl.read_intrinsics_text(os.path.join(sparse, "cameras.txt"))

    images_dir = os.path.join(scene_dir, images_subdir)
    rgb_files = sorted(glob.glob(os.path.join(images_dir, "*")))
    rgb_files = [f for f in rgb_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # Match readColmapCameras: cam list theo extrinsic key order, rồi sort by image_name.
    cams = []
    for idx, key in enumerate(sorted(ext.keys())):
        e = ext[key]
        i = intr[e.camera_id]
        R_w2c = qvec2rotmat(e.qvec)               # (3,3) world→cam
        T = np.array(e.tvec, dtype=np.float64)    # world→cam translation
        if i.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            fx = fy = float(i.params[0])
        elif i.model == "PINHOLE":
            fx = float(i.params[0]); fy = float(i.params[1])
        else:
            raise ValueError(f"Unsupported COLMAP model: {i.model}")
        image_path = os.path.join(images_dir, os.path.basename(e.name))
        cams.append({
            "uid": idx, "image_path": image_path,
            "image_name": os.path.basename(image_path).split(".")[0],
            "R_w2c": R_w2c, "T": T,
            "focal_x": fx, "focal_y": fy,
            "height": int(i.height), "width": int(i.width),
        })
    cams = sorted(cams, key=lambda c: c["image_name"])

    # Eval split — train cams = idx % llffhold != 0
    train_cams = [c for idx, c in enumerate(cams) if idx % llffhold != 0]

    # n_views subsample
    if n_views > 0:
        idx_sub = np.linspace(0, len(train_cams) - 1, n_views)
        idx_sub = [round(i) for i in idx_sub]
        train_cams = [c for idx, c in enumerate(train_cams) if idx in idx_sub]
        assert len(train_cams) == n_views, \
            f"Expected {n_views} train cams sau subsample, got {len(train_cams)}"

    return train_cams


def colmap_to_c2w(R_w2c, T_w2c):
    """COLMAP stores world→cam. DUSt3R preset_pose cần cam→world.

    W2C = [[R, T], [0,1]] → C2W = inv(W2C). Equivalent:
        R_c2w = R_w2c.T
        T_c2w = -R_w2c.T @ T_w2c
    """
    W2C = np.eye(4, dtype=np.float64)
    W2C[:3, :3] = R_w2c
    W2C[:3, 3] = T_w2c
    return np.linalg.inv(W2C)


def process_scene(scene_name, args, model):
    """Run DUSt3R cho 1 scene, save npz."""
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    scene_dir = os.path.join(args.data_root, scene_name)
    if not os.path.isdir(scene_dir):
        print(f"[skip] {scene_name}: không tồn tại {scene_dir}")
        return

    print(f"\n=== Processing scene: {scene_name} ===")
    train_cams = load_colmap_train_cams(scene_dir, args.n_views, args.llffhold)
    image_paths = [c["image_path"] for c in train_cams]
    print(f"Train images ({len(image_paths)}):")
    for p in image_paths:
        print(f"  {p}")

    # 1. Load images at DUSt3R input size.
    images = load_images(image_paths, size=args.image_size, verbose=True)

    # 2. Build complete pair graph (n_views=3 → 3 pairs × symmetrize = 6).
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
    print(f"Pairs: {len(pairs)}")

    # 3. Inference.
    out = inference(pairs, model, args.device, batch_size=1, verbose=True)

    # 4. Global alignment với COLMAP poses + focals preset.
    # init='known_poses' yêu cầu CẢ poses VÀ focals đều preset (assertion
    # nkf == n_imgs trong init_from_known_poses). preset_pose CHỈ fix poses,
    # nên thêm preset_focal sau khi scale theo tỉ lệ resize của load_images.
    scene = global_aligner(out, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer)
    c2w_list = [torch.from_numpy(colmap_to_c2w(c["R_w2c"], c["T"])).float() for c in train_cams]
    scene.preset_pose(c2w_list)

    # Scale COLMAP focal → DUSt3R focal theo image-size ratio sau load_images.
    # DUSt3R single focal/image (square pixel) → average fx, fy đã scale.
    # imshapes order match input order của load_images = order của train_cams.
    imshapes = scene.imshapes  # list[(H_new, W_new)]
    focals_dust3r = []
    for c, (H_new, W_new) in zip(train_cams, imshapes):
        sx = W_new / c["width"]
        sy = H_new / c["height"]
        f_new = 0.5 * (c["focal_x"] * sx + c["focal_y"] * sy)
        focals_dust3r.append(float(f_new))
    print(f"Focals (DUSt3R): {[f'{f:.1f}' for f in focals_dust3r]}")
    scene.preset_focal(focals_dust3r)

    loss = scene.compute_global_alignment(init="known_poses", niter=args.niter, lr=args.lr)
    print(f"Final alignment loss: {float(loss):.6f}")

    # 5. Extract pts3d + colors + conf, mask + concat.
    pts3d_list = [p.detach().cpu().numpy() for p in scene.get_pts3d()]    # list[H,W,3]
    imgs_list = scene.imgs                                                 # list[H,W,3] in [0,1]
    conf_list = [c.detach().cpu().numpy() for c in scene.im_conf]          # list[H,W]

    all_pts, all_cols, all_conf = [], [], []
    for pts, img, conf in zip(pts3d_list, imgs_list, conf_list):
        H, W = conf.shape
        pts = pts.reshape(-1, 3)
        if isinstance(img, np.ndarray):
            cols = img.reshape(-1, 3)
        else:
            cols = np.asarray(img).reshape(-1, 3)
        conf_flat = conf.reshape(-1)
        # Mask: keep all (filtering by threshold xảy ra ở training-time wrapper).
        # Vẫn drop NaN/Inf.
        finite = np.isfinite(pts).all(axis=1) & np.isfinite(conf_flat)
        all_pts.append(pts[finite])
        all_cols.append(cols[finite])
        all_conf.append(conf_flat[finite])

    points = np.concatenate(all_pts, axis=0).astype(np.float32)
    colors = np.concatenate(all_cols, axis=0).astype(np.float32)
    confidence = np.concatenate(all_conf, axis=0).astype(np.float32)
    print(f"Total dense points: {len(points)} (before training-time filter)")
    print(f"  Conf percentiles: p10={np.percentile(confidence, 10):.3f} "
          f"p50={np.percentile(confidence, 50):.3f} "
          f"p90={np.percentile(confidence, 90):.3f}")

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f"{scene_name}.npz")
    np.savez_compressed(out_path, points=points, colors=colors, confidence=confidence)
    print(f"Saved: {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)")


def main():
    args = parse_args()
    setup_paths(args.dust3r_path)

    # Load model một lần, reuse cho mọi scene.
    from dust3r.model import AsymmetricCroCo3DStereo
    print(f"Loading DUSt3R model từ {args.dust3r_ckpt} ...")
    model = AsymmetricCroCo3DStereo.from_pretrained(args.dust3r_ckpt).to(args.device)
    model.eval()

    os.makedirs(args.output, exist_ok=True)
    for scene in args.scenes:
        try:
            process_scene(scene, args, model)
        except Exception as e:
            print(f"[ERROR] {scene}: {e}")
            import traceback; traceback.print_exc()

    print("\n=== Done. Cache saved to:", args.output, "===")


if __name__ == "__main__":
    main()
