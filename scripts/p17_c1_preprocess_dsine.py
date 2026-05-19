#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 17 — C1] DSINE normal PREPROCESS (no-train).
# File: scripts/p17_c1_preprocess_dsine.py  (TẠO MỚI — keep local)
#
# Chạy DSINE trên 3 TRAIN-view/scene (giống L_depth dùng aligned_depth
# trên train-cam), lưu normal-map camera-frame [0,1] vào
#   <source_path>/<C1_DIR>/<image_stem>.npy   (H,W,3) float32.
# train.py (--use_c1_normal) load 1 lần, key theo image stem.
#
# Recipe = MIRROR CHÍNH XÁC mono-target của dn-splatter (verified):
#   run_monocular_dsine (normals_from_pretrain.py:127-146):
#     normal_b3hw = model(rgb) ; (b,h,w,3) ; @ diag([-1,1,1]) (LUF→RUF) ;
#     (normal+1)/2 → [0,1] png.
#   ⟹ ta lưu: DSINE_raw @ diag([-1,1,1]) → (x+1)/2, camera-frame.
#   KHÔNG world/wvt/orient (loss C1a cũng camera-[0,1] — xem
#   utils/loss/c1_normal.py). Self-consistent vì 2 recipe surface/mono
#   của dn-splatter dùng chung camera-[0,1] convention.
#
# DSINE = reuse scripts/dsine_pkg/ (vendored + audited verbatim, KHÔNG
#   import dn_splatter — tránh nerfstudio). Weights HF auto hoặc env
#   DSINE_CKPT=/path. KHÔNG chạy local (feedback_no_local_execution).
# ============================================================
"""[CRSGaussian Phase 17] C1 DSINE normal preprocess.

Server (no-train, ~phút/scene; cần geffnet jaxtyping + internet/DSINE_CKPT):
    cd CRSGaussian
    python -c "import ast; ast.parse(open('scripts/p17_c1_preprocess_dsine.py',encoding='utf-8').read()); print('syntax OK')"
    SCENES="fern flower fortress horns leaves orchids room trex" \
        python scripts/p17_c1_preprocess_dsine.py
  → ghi <data>/<scene>/c1_dsine_normals/<stem>.npy cho mỗi train view.
"""

import os
import sys
import math

import numpy as np

sys.path.insert(0, ".")

DATA_ROOT = os.environ.get("DATA_ROOT", "data/nerf_llff_data")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "output/p13_lfcf")
SEED = os.environ.get("SEED", "42")
ITERATION = int(os.environ.get("ITERATION", "10000"))
SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
C1_DIR = os.environ.get("C1_DIR", "c1_dsine_normals")
DSINE_CKPT = os.environ.get("DSINE_CKPT", "")


def _stem(name):
    return os.path.basename(name).split(".")[0]


def _load_dsine_predictor(dev):
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)            # so `import dsine_pkg` resolves
    try:
        from dsine_pkg import dsine_predictor as dpm
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{e}\n  → cần 'geffnet' + 'jaxtyping' (pip install geffnet "
            f"jaxtyping); và scripts/dsine_pkg/ phải cạnh script này.") from e
    if DSINE_CKPT:
        import torch as _t
        if not os.path.isfile(DSINE_CKPT):
            raise FileNotFoundError(f"DSINE_CKPT not found: {DSINE_CKPT}")

        def _local_sd(local_file_path=None, _p=DSINE_CKPT):
            return _t.load(_p, map_location=_t.device("cpu"))["model"]
        dpm._load_state_dict = _local_sd
        print(f"  [DSINE] offline weights: {DSINE_CKPT}")
    else:
        print("  [DSINE] HF auto-download (camenduru/DSINE/dsine.pt)")
    return dpm.DSinePredictor(device=dev)


def preprocess_scene(scene_name, predictor):
    import torch
    from argparse import ArgumentParser, Namespace
    from scene import Scene, GaussianModel
    from arguments import ModelParams, PipelineParams

    model_path = f"{OUTPUT_ROOT}/A3_seed{SEED}_{scene_name}"
    cfg_path = os.path.join(model_path, "cfg_args")
    ply = f"{model_path}/point_cloud/iteration_{ITERATION}/point_cloud.ply"
    if not os.path.isfile(cfg_path) or not os.path.isfile(ply):
        print(f"  SKIP {scene_name}: no cfg_args/ckpt at {model_path}")
        return
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    _ = lp
    args_cmdline = parser.parse_args([])
    with open(cfg_path) as f:
        cfg = eval(f.read())
    merged = vars(args_cmdline).copy()
    for k, v in vars(cfg).items():
        merged[k] = v
    args = Namespace(**merged)
    args.model_path = model_path
    if not os.path.isdir(getattr(args, "source_path", "") or ""):
        args.source_path = os.path.join(DATA_ROOT, scene_name)
    src = args.source_path

    out_dir = os.path.join(src, C1_DIR)
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        gaussians = GaussianModel(args)
        # Mirror p13_2 VERIFIED pattern: load_iteration=ITERATION → Scene
        # nạp pcd ckpt sẵn, KHÔNG vào create_from_pcd (cái này cần
        # args.train_bg — arg train.py thêm ở __main__, ParamGroup không
        # có → cfg_args thiếu → AttributeError nếu không load_iteration).
        # Preprocess chỉ cần train-cam ảnh+intrinsics; pcd load vô hại.
        scene = Scene(args, gaussians, load_iteration=ITERATION,
                      shuffle=False)
        train_cams = scene.getTrainCameras()
        for c in train_cams:
            W, H = c.image_width, c.image_height
            fx = W / (2.0 * math.tan(c.FoVx * 0.5))
            fy = H / (2.0 * math.tan(c.FoVy * 0.5))
            K = np.array([[fx, 0, W / 2.0],
                          [0, fy, H / 2.0],
                          [0, 0, 1]], np.float32)
            img = c.original_image[:3].clamp(0, 1).detach().cpu().numpy()
            rgb_u8 = (np.transpose(img, (1, 2, 0)) * 255.0).astype(np.uint8)
            n_b3hw = predictor(rgb_u8, K)                 # (1,3,h,w) raw cam
            n = n_b3hw[0].permute(1, 2, 0).numpy(
                force=True).astype(np.float32)            # (h,w,3)
            # dn-splatter mono recipe: LUF→RUF then [0,1] (verbatim).
            n = n @ np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
            n01 = ((n + 1.0) * 0.5).astype(np.float32)    # [0,1] camera frame
            stem = _stem(c.image_name)
            np.save(os.path.join(out_dir, f"{stem}.npy"), n01)
            print(f"  {scene_name}/{stem}  {n01.shape}  "
                  f"min={n01.min():.3f} max={n01.max():.3f}")
    del gaussians, scene
    torch.cuda.empty_cache()


def main():
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Phase 17 — C1 DSINE normal preprocess ===")
    print(f"SCENES={SCENES}  C1_DIR={C1_DIR}  (train-view only)")
    predictor = _load_dsine_predictor(dev)
    for sc in SCENES:
        print(f"──── {sc} ────")
        try:
            preprocess_scene(sc, predictor)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERR {sc}: {e}")
    print("Done. Per-train-view DSINE normals saved under "
          f"<source_path>/{C1_DIR}/.")


if __name__ == "__main__":
    main()
