#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 16 — C1 Gate] Normal-vs-Depth REDUNDANCY GATE.
# File: scripts/p16_normal_redundancy_gate.py  (TẠO MỚI — diagnostic, KEEP local)
#
# Mục đích: phép thử RẺ, NO-TRAIN, quyết định C1 (dn-splatter DSINE
#   monocular-normal external prior) SỐNG hay CHẾT *trước khi* tốn
#   DSINE-preprocess 8-scene + pilot 24-run.
#
# Câu hỏi cốt lõi (user 2026-05-19): "chỗ 2 bản đồ normal khác nhau là
#   chỗ A3 ĐANG SAI (DSINE tri-thức-mới), hay chỉ vegetation phức tạp mà
#   CẢ HAI đều không chắc (thêm DSINE = bơm nhiễu)?"
#
# Trả lời bằng 3 trục/pixel-test:
#   (1) θ      = góc(n_DSINE, n_A3depth)        — 2 bản đồ lệch bao nhiêu
#   (2) θ_self = self-consistency DSINE qua các train-view (Mẹo 1)
#                → DSINE có ĐÁNG TIN ở đó không (vs vegetation-mù)
#   (3) err    = |A3_render − GT_test| (Mẹo 2 / test-key)
#                → A3 có SAI THẬT ở đó không
#
# VERDICT (binary only):
#   θ nhỏ khắp           → NO  (redundant — DSINE lặp lại depth A3 đã có)
#   θ lớn, θ_self lớn    → NO  (trap — DSINE tự mâu thuẫn = vegetation noise)
#   θ lớn, tin, A3 đúng  → NO  (useless — DSINE khác nhưng A3 không cần)
#   θ lớn, tin, A3 sai   → GO  (DSINE tự-tin chỉ đúng chỗ A3 sai = +info mới)
#   số quyết định = lift = P(A3-sai | DSINE-confident-disagree) / P(A3-sai)
#
# 3 GUARDRAILS (test-for-triage KHÔNG test-for-tuning — memory đã chốt):
#   1. Script CHỈ in 1 verdict nhị phân. KHÔNG sinh hyperparam/checkpoint/
#      scene từ test. 2. KHÔNG đụng A3. 3. Bằng chứng cuối VẪN = full-8
#      multi-seed chuẩn — Gate chỉ là triage tiết kiệm compute, KHÔNG proof.
#
# CHỐT chống-lỗi-của-chính-tôi (audit 2026-05-19, diagnostic dự án đã
#   misfire: dense-init false-GO, L_consist false-reject):
#   • convention-sanity 2-tier (flat-med + glob-med) → coord-map sai =
#     UNRELIABLE, KHÔNG phán bừa.
#   • ASYMMETRY: θ_self warps bằng A3-depth (circularity) → GO tự-tin-cậy
#     / NO = WEAK (KHÔNG auto-đóng C1, escalate lý luận).
#   • τ DEFAULT PRE-REGISTERED đóng băng; env override → verdict tự gắn
#     "NON-CANONICAL, sensitivity-only" (chống cherry-pick borderline).
#   • RNG subsample seed cứng (reproducibility).
#
# Reuse VERIFIED (p13_2_bottleneck_decompose.py): Scene/GaussianModel/render
#   load (camera convention khớp training), cfg_args merge, _stem, cam_intr,
#   backproject (view=[(u-cx)/fx·d,(v-cy)/fy·d,d,1] @ wvt_inv ; wvt=W2C^T),
#   depth alpha-correct dp/(al+1e-6). DUAL-ANCHOR: n_dep from A3-render
#   -depth AND from DAV2-depth (utils.depth.precompute_depth_priors,
#   mirror train.py) + dcmp = "2 depth có cùng bản chất không".
# Reuse VERIFIED (dn-splatter normal_utils.pcd_to_normal + run_monocular_dsine):
#   ∇depth→normal = cross(right−left, top−bottom)·normalize ; DSINE out =
#   camera-space LUF [-1,1] → LUF→RUF diag([-1,1,1]).
#
# DSINE (VERIFIED-FROM-CODE 2026-05-19): dn_splatter/__init__.py:1-7 import
#   nerfstudio ⟹ KHÔNG `import dn_splatter`. Giải pháp: 4 file dsine
#   TỰ-CHỨA đã được VENDOR vào scripts/dsine_pkg/ (copy verbatim, CHỈ
#   sửa import → relative). Script chỉ `from dsine_pkg.dsine_predictor
#   import DSinePredictor` — KHÔNG đụng dn_splatter, KHÔNG git trên server.
#   Weights: HF camenduru/DSINE/dsine.pt (auto, cần internet) HOẶC env
#   DSINE_CKPT=/path (offline; monkeypatch module ĐÃ-import vì
#   dsine_predictor.load_model hardcode _load_state_dict(None)=HF, KHÔNG
#   sửa file vendor).
# SERVER SETUP (1 lần): pip install geffnet jaxtyping ; internet 1 lần
#   (DSINE wts + EfficientNet-B5 backbone). Upload: script + dsine_pkg/.
#   KHÔNG cần clone/đụng dn-splatter trên server. Chi tiết: docstring.
# ============================================================
"""[CRSGaussian Phase 16] C1 normal-redundancy Gate (NO-TRAIN, GPU inference).

SERVER SETUP (1 lần — KHÔNG chạy local; mọi lệnh trên Linux GPU server):
    # Upload: scripts/p16_normal_redundancy_gate.py + scripts/dsine_pkg/
    #   (4 file DSINE đã VENDOR sẵn — KHÔNG cần clone dn-splatter).
    # 1. cài 2 dep nhẹ vào CHÍNH env 3DGS đang dùng:
    pip install geffnet jaxtyping
    # 2. internet cần 1 lần (tự tải): DSINE wts (HF) + EfficientNet-B5
    #    (geffnet pretrained). Nếu server KHÔNG internet: tải sẵn dsine.pt
    #    rồi `export DSINE_CKPT=/abs/path/dsine.pt` (EfficientNet vẫn cần
    #    1 lần internet HOẶC timm cache sẵn).

Run (GPU SERVER — 3DGS env + A3 ckpt output/p13_lfcf/A3_seed42_<scene>):
    cd CRSGaussian
    # 0. syntax + smoke math (rẻ, KHÔNG GPU/DSINE — chạy trước):
    python -c "import ast; ast.parse(open('scripts/p16_normal_redundancy_gate.py',encoding='utf-8').read()); print('syntax OK')"
    python scripts/p16_normal_redundancy_gate.py --smoke
    # 1. full 8-scene, split 2 GPU (feedback_use_both_gpus). DSINE_AXIS_M
    #    = dn-splatter verified recipe diag([-1,1,1]); the [CONVENTION
    #    CHECK] block validates it (flat_med small ⟹ OK; large ⟹ deeper
    #    issue, escalate — do NOT guess):
    CUDA_VISIBLE_DEVICES=0 SCENES="fern flower fortress horns" \
        python scripts/p16_normal_redundancy_gate.py &
    CUDA_VISIBLE_DEVICES=1 SCENES="leaves orchids room trex" \
        python scripts/p16_normal_redundancy_gate.py &
    wait
    # 2. aggregate verdict (CPU, instant):
    python scripts/p16_normal_redundancy_gate.py --aggregate
"""

import os
import sys
import json
import math
import glob
import statistics

import numpy as np

sys.path.insert(0, ".")

# ── Config ──
DATA_ROOT = os.environ.get("DATA_ROOT", "data/nerf_llff_data")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "output/p13_lfcf")
SEED = os.environ.get("SEED", "42")
ITERATION = int(os.environ.get("ITERATION", "10000"))
SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
RESULT_DIR = os.environ.get("RESULT_DIR", "logs/p16_normal_gate")
DSINE_CKPT = os.environ.get("DSINE_CKPT", "")  # "" → torch.hub HF download

# τ self-calibration (percentile-based — KHÔNG magic number).
# ⚠ DEFAULTS = PRE-REGISTERED, ĐÓNG BĂNG 2026-05-19 (audit #2). Env
# override CHỈ để report SENSITIVITY (verdict có ổn định qua dải τ?) —
# TUYỆT ĐỐI KHÔNG dùng env lật NO→GO trên 1 lần chạy borderline. Đó
# CHÍNH là lỗi dense-init-R8 / L_consist-MARGIN_ABS mà header tự trích.
# Verdict CÔNG BỐ = luôn ở DEFAULT; chạy env khác → script tự gắn nhãn
# "NON-CANONICAL — sensitivity only" (xem _tau_nondefault()).
_TAU_DEF = dict(TAU_DIS_PCT="60", TAU_SELF_DEG="20", TAU_ERR_PCT="75",
                LIFT_GO="1.5", ERRMASS_GO="0.15", SELF_WIN_RAD="2")
TAU_DIS_PCT = float(os.environ.get("TAU_DIS_PCT", _TAU_DEF["TAU_DIS_PCT"]))
TAU_SELF_DEG = float(os.environ.get("TAU_SELF_DEG", _TAU_DEF["TAU_SELF_DEG"]))
TAU_ERR_PCT = float(os.environ.get("TAU_ERR_PCT", _TAU_DEF["TAU_ERR_PCT"]))
LIFT_GO = float(os.environ.get("LIFT_GO", _TAU_DEF["LIFT_GO"]))
ERRMASS_GO = float(os.environ.get("ERRMASS_GO", _TAU_DEF["ERRMASS_GO"]))
# audit #1a: θ_self windowed-min radius (px). Robust to small backproject
# misalignment from A3-depth error (the circularity). 0=single-pixel
# (no mitigation), 2=5×5 (default — tolerate ~2px, không quá lớn kẻo
# DSINE luôn "tự nhất quán" → mất bộ lọc vegetation).
SELF_WIN_RAD = int(os.environ.get("SELF_WIN_RAD", _TAU_DEF["SELF_WIN_RAD"]))


def _tau_nondefault():
    """List τ knobs overridden via env → verdict is NON-CANONICAL."""
    return [k for k in _TAU_DEF if os.environ.get(k) not in (None, _TAU_DEF[k])]
# Convention-sanity (2 tầng — audit 2026-05-19):
#  (1) SANITY_DEG: median θ trên vùng fronto-parallel (depth phẳng) —
#      mù với lỗi hoán X/Y (normal phẳng ≈ thuần Z) nên KHÔNG đủ một mình.
#  (2) GLOB_SANITY_DEG: median θ TOÀN ảnh. Map đúng ⟹ easy nhỏ + hard
#      tail ⟹ global median thấp hẳn; ≳ 75° = phân phối random = map hỏng.
# Quá 1 trong 2 → UNRELIABLE (KHÔNG phán verdict). Số convention LUÔN
# được in để người soi mắt trước khi tin GO/NO.
SANITY_DEG = float(os.environ.get("SANITY_DEG", "55"))
GLOB_SANITY_DEG = float(os.environ.get("GLOB_SANITY_DEG", "75"))


# ── CPU utils (mirror bottleneck) ──
def _stem(name):
    return os.path.basename(name).split(".")[0]


def _normalize(v, axis=-1, eps=1e-8):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def _angle_deg(a, b, axis=-1):
    """Per-element angle (deg) between unit-ish vectors a,b. Sign-canonical
    handled by caller (camera-facing orient) — here straight arccos."""
    a = _normalize(a, axis); b = _normalize(b, axis)
    d = np.clip(np.sum(a * b, axis=axis), -1.0, 1.0)
    return np.degrees(np.arccos(d))


# ── ∇depth → normal (world space) — reimpl dn-splatter pcd_to_normal +
#    backproject, dùng ĐÚNG convention CRSGaussian (wvt=W2C^T, verified
#    p13_2_bottleneck backproject). KHÔNG import dn_splatter (tránh kéo
#    nerfstudio); công thức đã verify-from-code. ──
def _backproj_grid(depth, fx, fy, cx, cy, wvt_inv):
    """depth (H,W) → world points (H,W,3). 3DGS: world = view @ wvt_inv,
    view=[(u-cx)/fx·d,(v-cy)/fy·d,d,1]."""
    H, W = depth.shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    u = (uu + 0.5).astype(np.float64)
    v = (vv + 0.5).astype(np.float64)
    d = depth.astype(np.float64)
    vx = (u - cx) / fx * d
    vy = (v - cy) / fy * d
    ones = np.ones_like(d)
    view = np.stack([vx, vy, d, ones], axis=-1)          # (H,W,4)
    world = view @ wvt_inv                                # (H,W,4)
    return world[..., :3]


def normal_from_depth_world(depth, fx, fy, cx, cy, wvt_inv):
    """World-space normal map (H,W,3) via finite-diff cross product
    (= dn-splatter pcd_to_normal). NaN where depth invalid."""
    xyz = _backproj_grid(depth, fx, fy, cx, cy, wvt_inv)  # (H,W,3) world
    # central neighbours (mirror normal_utils.pcd_to_normal indexing)
    left = xyz[1:-1, 0:-2, :]
    right = xyz[1:-1, 2:, :]
    top = xyz[0:-2, 1:-1, :]
    bottom = xyz[2:, 1:-1, :]
    n = np.cross(right - left, top - bottom)
    n = _normalize(n, axis=-1)
    out = np.full_like(xyz, np.nan)
    out[1:-1, 1:-1, :] = n
    return out


# SMOOTH_K = validity-weighted box-blur kernel applied to depth BEFORE
# the finite-diff normal. Robustifies against 1-pixel noise (the diagnosed
# A3-render-depth-noise problem). Code-grounded: dn-splatter
# normal_utils.normal_from_depth_image has a `smooth` (Gaussian-blur)
# option for exactly this; open3d-KNN=200 in depth_to_normal.py is the
# same "robust neighbourhood" intent. NOT a verdict knob (preprocessing)
# — frozen default; env only for robustness sweep.
SMOOTH_K = int(os.environ.get("SMOOTH_K", "11"))


def _smooth_depth(depth, k=None):
    """Validity-weighted box blur: blur(d·valid)/blur(valid). Keeps
    invalid (NaN/≤0) from smearing in; result NaN where the window is
    <30% valid (don't trust heavily-missing neighbourhoods)."""
    import cv2
    k = SMOOTH_K if k is None else k
    if k <= 1:
        return depth.astype(np.float64)
    valid = np.isfinite(depth) & (depth > 0)
    vd = np.where(valid, depth, 0.0).astype(np.float64)
    num = cv2.blur(vd, (k, k))
    den = cv2.blur(valid.astype(np.float64), (k, k))
    out = np.where(den > 0.3, num / np.maximum(den, 1e-9), np.nan)
    return out


def robust_normal_from_depth_world(depth, fx, fy, cx, cy, wvt_inv):
    """K×K-robust world normal: smooth depth (kills A3-render 1-pixel
    noise) then the SAME verified cross-product normal. Used for BOTH
    anchors so the A3 vs DAV2 comparison isolates the depth SOURCE,
    not the operator."""
    return normal_from_depth_world(_smooth_depth(depth), fx, fy, cx, cy,
                                   wvt_inv)


def orient_to_camera(normal_w, depth, fx, fy, cx, cy, wvt_inv, cam_center_w):
    """Flip normals to face the camera (dn-splatter normal_dir_not_correct).
    Removes ±n ambiguity AND makes n_DSINE vs n_depth comparison sign-safe."""
    xyz = _backproj_grid(depth, fx, fy, cx, cy, wvt_inv)         # (H,W,3)
    ray = xyz - cam_center_w.reshape(1, 1, 3)                    # surf - cam
    flip = np.sum(ray * normal_w, axis=-1) > 0                   # same dir
    out = normal_w.copy()
    out[flip] = -out[flip]
    return out


# ── DSINE (camera-space LUF) → world, oriented-to-camera ──
def _K33(c):
    """Camera intrinsic 3×3 from a cam dict."""
    return np.array([[c["fx"], 0, c["cx"]],
                     [0, c["fy"], c["cy"]],
                     [0, 0, 1]], np.float32)


def dsine_raw_cam(predictor, rgb_uint8, K33):
    """DSINE raw output → (h,w,3) numpy in DSINE's CAMERA frame.
    NO axis transform, NO world rotation, NO orient. The axis-convention
    fix (DSINE_AXIS_M) is applied separately by dsine_normal_world."""
    n_b3hw = predictor(rgb_uint8, K33)                    # (1,3,h,w)
    return n_b3hw[0].permute(1, 2, 0).numpy(force=True).astype(np.float64)


# DSINE camera-frame → CRSGaussian OpenCV camera-frame.
# = dn-splatter's VERIFIED recipe, EXACTLY (verify-from-code 2026-05-19,
#   user-double-checked claims A–D):
#   • run_monocular_dsine (normals_from_pretrain.py:130-133): the ONLY
#     transform is `normal @ diag([-1,1,1])` (LUF→RUF).
#   • depth_to_normal.py:197-204 then treats that directly as an
#     OpenCV-camera normal, rotating to world by R_c2w.
#   • Algebra-proven: our cam→world `n @ wvt_inv[:3,:3]` == dn-splatter
#     `n @ R_w2c` (identical). CRSGaussian cam frame = COLMAP/OpenCV
#     x-right/y-down/z-forward (getWorld2View2 + bottleneck depth↔
#     COLMAP-Z corr>0.9). ⟹ the single diag([-1,1,1]) is the WHOLE map.
# The earlier extra diag([1,-1,1]) ("RUF→OpenCV" guess) was the bug
#   (flipped Y → convention-sanity 7/8 FAIL). NOT provisional anymore —
#   this mirrors the source recipe; the Gate [CONVENTION CHECK] (n_dep
#   anchor, full-pipeline) is the validator.
DSINE_AXIS_M = np.diag([-1.0, 1.0, 1.0])


def dsine_normal_world(predictor, rgb_uint8, K33, depth, fx, fy, cx, cy,
                       wvt_inv, cam_center_w):
    """raw DSINE-cam → DSINE_AXIS_M (→OpenCV cam) → world → orient-to-cam.
    Both DSINE_AXIS_M and cam→world `n @ wvt_inv[:3,:3]` mirror
    dn-splatter's verified recipe exactly (see DSINE_AXIS_M comment);
    the Gate [CONVENTION CHECK] block is the runtime validator."""
    n = dsine_raw_cam(predictor, rgb_uint8, K33)          # (h,w,3) cam
    n = n @ DSINE_AXIS_M                                   # DSINE → OpenCV cam
    n = n @ wvt_inv[:3, :3]                                # cam → world
    n = _normalize(n, axis=-1)
    return orient_to_camera(n, depth, fx, fy, cx, cy, wvt_inv, cam_center_w)


# ── GPU loader: A3 ckpt → train+test renders + cams (mirror bottleneck) ──
def load_scene(scene_name):
    """Returns dict with test_cams, train_cams. Each cam dict:
      {name, depth(H,W), wvt, wvt_inv, fx,fy,cx,cy,W,H, cam_center(3,),
       rgb_uint8(H,W,3)}   (+ test_cams also: render(H,W,3), gt(H,W,3))
    GPU inference only (no_grad). None if checkpoint missing."""
    import torch
    from argparse import ArgumentParser, Namespace
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from arguments import ModelParams, PipelineParams

    model_path = f"{OUTPUT_ROOT}/A3_seed{SEED}_{scene_name}"
    cfg_path = os.path.join(model_path, "cfg_args")
    ply = f"{model_path}/point_cloud/iteration_{ITERATION}/point_cloud.ply"
    if not os.path.isfile(cfg_path) or not os.path.isfile(ply):
        print(f"  ERR missing cfg_args/ckpt at {model_path}")
        return None

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
    args.iteration = ITERATION
    if not os.path.isdir(getattr(args, "source_path", "") or ""):
        args.source_path = os.path.join(DATA_ROOT, scene_name)
    pipe = pp.extract(args)

    def cam_intr(c):
        W, H = c.image_width, c.image_height
        fx = W / (2.0 * math.tan(c.FoVx * 0.5))
        fy = H / (2.0 * math.tan(c.FoVy * 0.5))
        return fx, fy, W / 2.0, H / 2.0, W, H

    def cam_pack(c, with_render):
        fx, fy, cx, cy, W, H = cam_intr(c)
        wvt = c.world_view_transform.detach().cpu().numpy()
        wvt_inv = np.linalg.inv(wvt)
        # camera center in world: row-vec convention, origin@wvt_inv
        cc = (np.array([0.0, 0.0, 0.0, 1.0]) @ wvt_inv)[:3]
        img = c.original_image[:3].clamp(0, 1).detach().cpu().numpy()  # 3,H,W
        rgb_u8 = (np.transpose(img, (1, 2, 0)) * 255.0).astype(np.uint8)
        out = dict(name=_stem(c.image_name), uid=c.uid,
                   wvt=wvt, wvt_inv=wvt_inv,
                   fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H,
                   cam_center=cc.astype(np.float64), rgb_uint8=rgb_u8)
        if with_render:
            pkg = render(c, gaussians, pipe, bg)
            al = np.squeeze(pkg["alpha"].detach().cpu().numpy())
            dp = np.squeeze(pkg["depth"].detach().cpu().numpy())
            out["depth"] = (dp / (al + 1e-6)).astype(np.float32)
            rd = pkg["render"].clamp(0, 1).detach().cpu().numpy()
            out["render"] = np.transpose(rd, (1, 2, 0)).astype(np.float32)
            out["gt"] = np.transpose(img, (1, 2, 0)).astype(np.float32)
        else:
            pkg = render(c, gaussians, pipe, bg)
            al = np.squeeze(pkg["alpha"].detach().cpu().numpy())
            dp = np.squeeze(pkg["depth"].detach().cpu().numpy())
            out["depth"] = (dp / (al + 1e-6)).astype(np.float32)
        return out

    torch.cuda.empty_cache()
    with torch.no_grad():
        gaussians = GaussianModel(args)
        scene = Scene(args, gaussians, load_iteration=ITERATION, shuffle=False)
        bg_color = [1., 1., 1.] if getattr(args, "white_background", False) \
            else [0., 0., 0.]
        bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        train_cams = [cam_pack(c, False) for c in scene.getTrainCameras()]
        test_cam_objs = scene.getTestCameras()
        test_cams = [cam_pack(c, True) for c in test_cam_objs]

        # ── DAV2 raw depth for test cams (dual-anchor — mirror train.py
        #    exact call: precompute_depth_priors(cams, dav2_path,
        #    encoder=dav2_encoder)). RAW relative (NO align_to_colmap:
        #    test cams lack n_views COLMAP sparse; normals are
        #    scale-invariant + local-shift 2nd-order → raw = the DAV2
        #    ORIENTATION = exactly the C1-redundancy reference). Guarded:
        #    any failure → depth_dav2=None → DAV2 anchor reported N/A
        #    (A3 anchor still runs). ──
        dav2_path = getattr(args, "dav2_path", "") or ""
        dav2_enc = getattr(args, "dav2_encoder", "vitl")
        if getattr(args, "use_depth_prior", False) and dav2_path:
            try:
                from utils.depth import precompute_depth_priors
                dav2 = precompute_depth_priors(
                    test_cam_objs, dav2_path, encoder=dav2_enc)
                for tc in test_cams:
                    dd = dav2.get(tc["uid"])
                    tc["depth_dav2"] = (None if dd is None
                                        else dd.numpy().astype(np.float32))
            except Exception as e:
                print(f"  [DAV2] precompute FAILED ({e}) → DAV2 anchor N/A")
                for tc in test_cams:
                    tc["depth_dav2"] = None
        else:
            print("  [DAV2] use_depth_prior off / no dav2_path → DAV2 N/A")
            for tc in test_cams:
                tc["depth_dav2"] = None
    del gaussians, scene
    torch.cuda.empty_cache()
    return dict(test_cams=test_cams, train_cams=train_cams)


# ── Mẹo 1: DSINE cross-view self-consistency ──
def dsine_self_consistency(tc, n_dsine_t, ref_cams, dsine_world_cache):
    """For each pixel of test cam tc: backproject via A3-depth → world →
    project into each REF (train) cam → sample its n_DSINE_world →
    UNDIRECTED angle vs n_dsine_t. θ_self (H,W) = MIN over reachable refs
    (best-case agreement; min = lenient → conservative GO). NaN where no
    ref sees the point.

    FIX 2026-05-19 (audit): (1) ref_cams = TRAIN cams only — using other
    TEST cams hit a cache-ordering KeyError (their DSINE not yet computed)
    AND is conceptually wrong (the 3 train views are exactly C1's overlap
    set). .get() guard kept as belt-and-braces. (2) UNDIRECTED angle
    (fold ±n: min(θ,180−θ)) — n_dsine_t & n_o are each oriented to their
    OWN camera; a directed compare across views with different camera
    directions inflates θ_self spuriously. Plane orientation is the
    physically meaningful quantity (±n equivalent).

    AUDIT #1 (2026-05-19, CIRCULARITY — known, partially mitigated): the
    warp uses tc A3-depth. Where A3-depth is wrong (correlates with the
    region C1 substrate would live) the projected pixel is off → θ_self
    samples the wrong surface → θ_self inflated EXACTLY in the region we
    care about → pixel drops 'confident' → conf_dis under-counts →
    Gate biased to false-NO there. Magnitude bounded (bottleneck H3≈0.5%
    = A3-geometry largely solved) + windowed-MIN below tolerates small
    misalignment. RESIDUAL handled by ASYMMETRIC verdict reading: GO =
    trustworthy/act; NO = WEAK (Gate may be self-blinded) → do NOT
    auto-close C1 on NO, escalate to reasoning. See _verdict()."""
    H, W = tc["depth"].shape
    xyz = _backproj_grid(tc["depth"], tc["fx"], tc["fy"], tc["cx"], tc["cy"],
                         tc["wvt_inv"])                         # (H,W,3) world
    best = np.full((H, W), np.nan)
    for oc in ref_cams:
        n_o = dsine_world_cache.get(oc["name"])                 # (Ho,Wo,3)
        if n_o is None:
            continue
        wvt = oc["wvt"]
        ones = np.ones(xyz.shape[:2] + (1,))
        pw = np.concatenate([xyz, ones], axis=-1)               # (H,W,4)
        pv = pw @ wvt                                           # world→view
        z = pv[..., 2]
        valid = z > 1e-4
        u = oc["fx"] * np.where(valid, pv[..., 0], 0) / np.where(valid, z, 1) \
            + oc["cx"]
        v = oc["fy"] * np.where(valid, pv[..., 1], 0) / np.where(valid, z, 1) \
            + oc["cy"]
        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        inb = valid & (ui >= 0) & (ui < oc["W"]) & (vi >= 0) & (vi < oc["H"])
        if not inb.any():
            continue
        # audit #1a: windowed-MIN over ±SELF_WIN_RAD around the projected
        # pixel → tolerate small backproject misalignment caused by
        # A3-depth error (the θ_self circularity). MIN keeps it lenient
        # (best-case agreement) = pushes AGAINST the false-NO bias.
        ang = np.full((H, W), np.inf)
        for dv in range(-SELF_WIN_RAD, SELF_WIN_RAD + 1):
            for du in range(-SELF_WIN_RAD, SELF_WIN_RAD + 1):
                uic = np.clip(ui + du, 0, oc["W"] - 1)
                vic = np.clip(vi + dv, 0, oc["H"] - 1)
                no_s = n_o[vic, uic, :]                          # (H,W,3)
                aa = _angle_deg(n_dsine_t, no_s)
                aa = np.minimum(aa, 180.0 - aa)   # UNDIRECTED (±n equiv)
                ang = np.minimum(ang, aa)
        ang[~inb] = np.nan
        best = np.where(np.isnan(best), ang,
                        np.where(np.isnan(ang), best,
                                 np.minimum(best, ang)))
    return best


# NOTE (audit 2026-05-19): an earlier design used a covis-count proxy for
# the "A3-weak region" (Mẹo 2). The implemented Gate instead uses the
# REAL held-out test photometric error (er > τ_err) as the "A3 wrong"
# axis — strictly stronger than a covis proxy — so the covis-count helper
# was removed (was dead code, never called). Mẹo 2 = test-key error.


# ── DSINE import — from VENDORED scripts/dsine_pkg/ (verbatim copy of
#    dn-splatter dsine, imports rewritten relative). Avoids touching
#    dn_splatter/__init__.py (nerfstudio). Verified-from-code 2026-05-19. ──
def _load_dsine_predictor(dev):
    """Return DSinePredictor on dev from scripts/dsine_pkg/. Honors env
    DSINE_CKPT (offline) by monkeypatching the imported module's
    _load_state_dict (load_model hardcodes _load_state_dict(None)=HF — we
    patch the IMPORTED module object at runtime, NOT the vendor file)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)            # so `import dsine_pkg` resolves
    try:
        from dsine_pkg import dsine_predictor as dpm
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{e}\n  → cần 'geffnet' + 'jaxtyping' trong env 3DGS "
            f"(pip install geffnet jaxtyping); và scripts/dsine_pkg/ phải "
            f"nằm cạnh script này.") from e
    if DSINE_CKPT:
        import torch as _t
        if not os.path.isfile(DSINE_CKPT):
            raise FileNotFoundError(f"DSINE_CKPT not found: {DSINE_CKPT}")

        def _local_sd(local_file_path=None, _p=DSINE_CKPT):
            return _t.load(_p, map_location=_t.device("cpu"))["model"]
        dpm._load_state_dict = _local_sd
        print(f"  [DSINE] offline weights: {DSINE_CKPT}")
    else:
        print("  [DSINE] weights via HF auto-download "
              "(camenduru/DSINE/dsine.pt) — server needs internet")
    return dpm.DSinePredictor(device=dev)


# ── Per-anchor metrics (conv-sanity + 3-axis contingency) ──
# Factored so it runs ONCE per depth anchor (A3-render vs DAV2). th =
# angle(DSINE, n_dep_<anchor>); ths/er shared (θ_self, test-err don't
# depend on the anchor); fl = flat-mask from THAT anchor's depth.
def _scene_metrics(th, ths, er, fl):
    m = np.isfinite(th)
    if int(m.sum()) < 100:
        return dict(status="NO_PIXELS", flat_med=float("nan"),
                    flat_p10=float("nan"), glob_med=float("nan"), flat_n=0)
    th, ths, er, fl = th[m], ths[m], er[m], fl[m]
    flat_th = th[fl]
    flat_med = float(np.median(flat_th)) if flat_th.size else float("nan")
    flat_p10 = (float(np.percentile(flat_th, 10))
                if flat_th.size else float("nan"))
    glob_med = float(np.median(th))
    conv = dict(flat_med=flat_med, flat_p10=flat_p10, glob_med=glob_med,
                flat_n=int(flat_th.size))
    if (not np.isfinite(flat_med)) or (flat_med > SANITY_DEG) \
            or (glob_med > GLOB_SANITY_DEG):
        return dict(status="UNRELIABLE", **conv,
                    note=(f"coord-sanity FAIL (flat_med={flat_med:.1f}"
                          f">{SANITY_DEG} OR glob_med={glob_med:.1f}"
                          f">{GLOB_SANITY_DEG})"))
    tau_dis = float(np.percentile(th, TAU_DIS_PCT))
    tau_err = float(np.percentile(er, TAU_ERR_PCT))
    disagree = th > tau_dis
    confident = np.isfinite(ths) & (ths < TAU_SELF_DEG)
    a3_wrong = er > tau_err
    conf_dis = disagree & confident
    p_wrong = float(a3_wrong.mean())
    p_wrong_given_cd = (float(a3_wrong[conf_dis].mean())
                        if conf_dis.any() else 0.0)
    lift = (p_wrong_given_cd / p_wrong) if p_wrong > 1e-6 else 0.0
    go_cell = conf_dis & a3_wrong
    errmass_go = (float(er[go_cell].sum()) / float(er.sum())
                  if er.sum() > 1e-9 else 0.0)
    return dict(
        status="OK", **conv,
        tau_dis=tau_dis, tau_err=tau_err,
        frac_disagree=float(disagree.mean()),
        frac_conf_of_dis=(float(confident[disagree].mean())
                          if disagree.any() else 0.0),
        p_wrong=p_wrong, p_wrong_given_cd=p_wrong_given_cd,
        lift=lift, errmass_go=errmass_go)


# ── Per-scene Gate (DUAL-ANCHOR: A3-render-depth vs DAV2-depth) ──
def gate_scene(scene_name):
    data = load_scene(scene_name)
    if data is None:
        return None
    test_cams = data["test_cams"]
    train_cams = data["train_cams"]
    has_dav2 = any(tc.get("depth_dav2") is not None for tc in test_cams)

    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = _load_dsine_predictor(dev)

    dsine_world_cache = {}
    for c in train_cams:
        dsine_world_cache[c["name"]] = dsine_normal_world(
            predictor, c["rgb_uint8"], _K33(c), c["depth"],
            c["fx"], c["fy"], c["cx"], c["cy"], c["wvt_inv"], c["cam_center"])

    def _norm(depth, tc):  # robust K×K normal → world, oriented-to-cam
        nd = robust_normal_from_depth_world(
            depth, tc["fx"], tc["fy"], tc["cx"], tc["cy"], tc["wvt_inv"])
        return orient_to_camera(nd, depth, tc["fx"], tc["fy"], tc["cx"],
                                tc["cy"], tc["wvt_inv"], tc["cam_center"])

    def _flatmask(depth, base):
        gy, gx = np.gradient(np.nan_to_num(depth.astype(np.float64)))
        g = np.hypot(gx, gy)
        bb = base & np.isfinite(depth) & (np.asarray(depth) > 0)
        if not bb.any():
            return bb
        return bb & (g < np.nanpercentile(g[bb], 25))

    # rows cols: thA3, thDAV2, dcmp, ths, err, flatA3, flatDAV2
    rows = []
    for tc in test_cams:
        valid = np.isfinite(tc["depth"]) & (tc["depth"] > 0)
        n_ds = dsine_normal_world(predictor, tc["rgb_uint8"], _K33(tc),
                                  tc["depth"], tc["fx"], tc["fy"], tc["cx"],
                                  tc["cy"], tc["wvt_inv"], tc["cam_center"])
        n_a3 = _norm(tc["depth"], tc)
        th_a3 = _angle_deg(n_ds, n_a3)
        if tc.get("depth_dav2") is not None:
            n_dv = _norm(tc["depth_dav2"], tc)
            th_dv = _angle_deg(n_ds, n_dv)
            dcmp = _angle_deg(n_a3, n_dv)            # A3-normal vs DAV2-normal
            flat_dv = _flatmask(tc["depth_dav2"], valid)
        else:
            th_dv = np.full_like(th_a3, np.nan)
            dcmp = np.full_like(th_a3, np.nan)
            flat_dv = np.zeros_like(valid)
        theta_self = dsine_self_consistency(
            tc, n_ds, train_cams, dsine_world_cache)
        err = np.mean(np.abs(tc["render"] - tc["gt"]), axis=2)
        flat_a3 = _flatmask(tc["depth"], valid)

        idx = np.argwhere(valid)
        if len(idx) > 80000:
            # FIX audit #3: fixed-seed RNG (reproducibility).
            sel = np.random.default_rng(0).choice(
                len(idx), 80000, replace=False)
            idx = idx[sel]
        for (yi, xi) in idx:
            ts = theta_self[yi, xi]
            rows.append((float(th_a3[yi, xi]), float(th_dv[yi, xi]),
                         float(dcmp[yi, xi]),
                         float(ts) if np.isfinite(ts) else np.nan,
                         float(err[yi, xi]),
                         bool(flat_a3[yi, xi]), bool(flat_dv[yi, xi])))

    if not rows:
        return dict(scene=scene_name, status="NO_PIXELS")
    a = np.array(rows, dtype=np.float64)
    thA3, thDV, dcmp, ths, er = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    flA3, flDV = a[:, 5].astype(bool), a[:, 6].astype(bool)

    mA3 = _scene_metrics(thA3, ths, er, flA3)
    mDV = (_scene_metrics(thDV, ths, er, flDV) if has_dav2
           else dict(status="NA"))
    # dcmp = "are A3-depth-normal & DAV2-depth-normal the same nature?"
    # measured on pixels flat in BOTH (reliable geometry on both sides).
    dboth = dcmp[flA3 & flDV & np.isfinite(dcmp)]
    dcmp_flatmed = float(np.median(dboth)) if dboth.size else float("nan")

    out = dict(scene=scene_name, n_px=int(len(a)),
               dcmp_flatmed=dcmp_flatmed, has_dav2=bool(has_dav2))
    for pfx, mm in (("A3", mA3), ("DV", mDV)):
        for k, v in mm.items():
            out[f"{pfx}_{k}"] = v
    return out


# ── Verdict aggregation (per anchor: "A3" or "DV") ──
def _verdict(results, anchor):
    S = lambda r, k: r.get(f"{anchor}_{k}")
    ok = [r for r in results if S(r, "status") == "OK"]
    unrel = [r for r in results if S(r, "status") == "UNRELIABLE"]
    na = [r for r in results if S(r, "status") == "NA"]
    if na and not ok and not unrel:
        return ("NA", f"{anchor} anchor unavailable (DAV2 not computed).")
    if not ok:
        return ("UNRELIABLE",
                f"0 scene reliable ({anchor}; coord-sanity failed all) "
                f"[{len(unrel)} UNRELIABLE]")
    mean_lift = statistics.fmean(S(r, "lift") for r in ok)
    mean_em = statistics.fmean(S(r, "errmass_go") for r in ok)
    mean_fd = statistics.fmean(S(r, "frac_disagree") for r in ok)
    n_go = sum(1 for r in ok
               if S(r, "lift") >= LIFT_GO and S(r, "errmass_go") >= ERRMASS_GO)
    if mean_fd < 0.10:
        v = ("NO_REDUNDANT",
             f"θ disagreement tiny (mean frac={mean_fd:.2f}<0.10) → DSINE ≈ "
             f"derivable from A3 depth. C1a redundant, close cheap.")
    elif mean_lift >= LIFT_GO and mean_em >= ERRMASS_GO and n_go >= len(ok) / 2:
        v = ("GO",
             f"lift={mean_lift:.2f}≥{LIFT_GO} & err-mass={mean_em:.2f}≥"
             f"{ERRMASS_GO} on ≥half scenes ({n_go}/{len(ok)}) → DSINE "
             f"confidently flags REAL A3 errors = genuine +info. Pursue C1a.")
    elif mean_lift < 1.2:
        v = ("NO_TRAP",
             f"lift={mean_lift:.2f}≈1 → DSINE-confident-disagreement NOT "
             f"aligned with real A3 error (smeared on foliage regardless) → "
             f"adding DSINE = inject noise. Same failure class.")
    else:
        v = ("NO_WEAK",
             f"lift={mean_lift:.2f}/err-mass={mean_em:.2f} below GO bar "
             f"({LIFT_GO}/{ERRMASS_GO}) or <half scenes → signal too weak "
             f"to justify heavy DSINE preprocessing.")
    tag, why = v
    # audit #1b — ASYMMETRY baked in: GO is self-trustworthy; any NO is a
    # WEAK-NO (θ_self circularity may self-blind the Gate exactly in the
    # high-A3-error region where C1 substrate would live) → NO must NOT
    # auto-close C1.
    if tag != "GO":
        why += ("  ⚠ WEAK-NO: θ_self warps bằng A3-depth (circularity) → "
                "Gate có thể tự-mù ở vùng A3-error cao = ĐÚNG chỗ substrate "
                "C1. NO ≠ đóng C1 — phải escalate lý luận. CHỈ GO tự-tin-cậy.")
    if unrel:
        why += f"  [{len(unrel)} scene UNRELIABLE-excluded]"
    # audit #2 — pre-registration enforcement: env-overridden τ ⟹ verdict
    # is NON-CANONICAL (sensitivity only), can never be the published call.
    nd = _tau_nondefault()
    if nd:
        tag += "*"
        why += (f"  ⚠⚠ NON-CANONICAL: τ env override {nd} → CHỈ để "
                f"sensitivity, KHÔNG phải phán quyết. Verdict công bố phải "
                f"chạy DEFAULT (mọi env τ unset).")
    return tag, why


def _save(r):
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(os.path.join(RESULT_DIR, f"{r['scene']}.json"), "w") as f:
        json.dump(r, f, indent=2)


def aggregate_and_print():
    files = sorted(glob.glob(os.path.join(RESULT_DIR, "*.json")))
    results = []
    for fp in files:
        with open(fp) as f:
            results.append(json.load(f))
    if not results:
        print(f"No per-scene json in {RESULT_DIR}"); sys.exit(1)
    R = sorted(results, key=lambda x: x["scene"])
    print("\n=== Phase 16 — C1 Normal-Redundancy Gate (DUAL-ANCHOR) ===")

    # ── CONVENTION CHECK + dcmp FIRST (đọc TRƯỚC verdict) ──
    # Correct map ⟹ flat_med ≲30, glob_med «90. dcmp = góc giữa
    # normal-từ-A3-depth và normal-từ-DAV2-depth trên pixel reliable cả 2
    # = "2 depth có CÙNG BẢN CHẤT không" (nhỏ ⟹ A3≈DAV2, swap hợp lệ).
    print("\n[CONVENTION CHECK + dcmp]  (A3=A3-render-depth, DV=DAV2-depth)")
    print(f"{'scene':<9} {'A3flatM':>8} {'A3globM':>8} {'A3stat':>11} | "
          f"{'DVflatM':>8} {'DVglobM':>8} {'DVstat':>11} | {'dcmp°':>6}")
    for r in R:
        print(f"{r['scene']:<9} "
              f"{r.get('A3_flat_med', float('nan')):>8.1f} "
              f"{r.get('A3_glob_med', float('nan')):>8.1f} "
              f"{str(r.get('A3_status', '?')):>11} | "
              f"{r.get('DV_flat_med', float('nan')):>8.1f} "
              f"{r.get('DV_glob_med', float('nan')):>8.1f} "
              f"{str(r.get('DV_status', '?')):>11} | "
              f"{r.get('dcmp_flatmed', float('nan')):>6.1f}")

    for anc, name in (("DV", "DAV2-depth anchor (decision)"),
                      ("A3", "A3-render-depth anchor")):
        print(f"\n[GATE METRICS — {name}]")
        print(f"{'scene':<9} {'stat':>11} {'fr_dis':>7} {'cf|dis':>7} "
              f"{'lift':>6} {'errM_GO':>8}")
        for r in R:
            st = r.get(f"{anc}_status", "?")
            if st != "OK":
                print(f"{r['scene']:<9} {str(st):>11}")
                continue
            print(f"{r['scene']:<9} {'OK':>11} "
                  f"{r[f'{anc}_frac_disagree']:>7.2f} "
                  f"{r[f'{anc}_frac_conf_of_dis']:>7.2f} "
                  f"{r[f'{anc}_lift']:>6.2f} {r[f'{anc}_errmass_go']:>8.2f}")
        v, w = _verdict(results, anc)
        print(f"  → {name}: {v}\n    {w}")

    # ── Pre-registered interpretation (chốt TRƯỚC chạy — không cãi sau) ──
    dvals = [r["dcmp_flatmed"] for r in R
             if isinstance(r.get("dcmp_flatmed"), (int, float))
             and np.isfinite(r.get("dcmp_flatmed"))]
    mdc = statistics.fmean(dvals) if dvals else float("nan")
    dv_ok = [r for r in R if r.get("DV_status") == "OK"]
    dv_flat = statistics.fmean(
        r["DV_flat_med"] for r in dv_ok) if dv_ok else float("nan")
    print("\n=== PRE-REGISTERED INTERPRETATION ===")
    print(f"  mean dcmp(A3-normal vs DAV2-normal) = {mdc:.1f}°  | "
          f"DAV2-anchor mean flat_med = {dv_flat:.1f}° "
          f"({len(dv_ok)}/{len(R)} OK)")
    if not np.isfinite(mdc):
        concl = ("DAV2 anchor N/A (precompute off/failed) → chỉ có A3 anchor"
                 " (nhiễu) → KHÔNG đủ kết luận; bật DAV2 (use_depth_prior +"
                 " dav2_path) rồi chạy lại.")
    elif mdc <= 20 and dv_ok and dv_flat <= SANITY_DEG:
        concl = ("dcmp NHỎ (2 depth CÙNG bản chất) + DAV2 flat_med nhỏ →"
                 " A3-render chỉ thêm nhiễu, DAV2 = anchor sạch ĐÚNG →"
                 " ĐỌC verdict DAV2-anchor ở trên, hành động theo nó.")
    elif mdc <= 20:
        concl = ("dcmp NHỎ (2 depth giống nhau) NHƯNG DAV2 flat_med vẫn lớn"
                 " → KHÔNG phải lỗi nguồn-depth/nhiễu: hoặc convention vẫn"
                 " sai, HOẶC DSINE tự kém trên LLFF → ESCALATE (điều kiện"
                 " dừng): quyết C1 bằng lý luận+literature, KHÔNG patch Gate.")
    else:
        concl = ("dcmp LỚN (≳20°) → normal-từ-A3-render ĐÃ phân kỳ khỏi"
                 " normal-từ-DAV2-prior dù A3 train theo DAV2: A3-fit ≠"
                 " prior. Anchor đúng cho câu C1 = DAV2 (verdict DAV2 ở"
                 " trên); ghi nhận A3-geometry-sparse-view không tin được.")
    print(f"  → {concl}")

    print("\n  Guardrails: binary-only · A3 untouched · final proof = "
          "standard full-8 multi-seed.")
    print("  ASYMMETRY (audit #1): θ_self warps via A3-depth → GO = "
          "trustworthy/act; NO = WEAK, không tự đóng C1.")
    print("  STOP-CONDITION (pre-commit): đây là lần lặp Gate CUỐI. Rơi"
          " nhánh ESCALATE → quyết C1 bằng lý luận, KHÔNG patch Gate nữa.")
    nd = _tau_nondefault()
    print(f"  τ = {'NON-DEFAULT '+str(nd)+' → NON-CANONICAL' if nd else 'PRE-REGISTERED DEFAULT (canonical)'}.")


# ── Smoke (CPU, no GPU/DSINE) — coord + normal math sanity ──
def smoke():
    print("=== Smoke: coord/normal unit math (CPU) ===")
    # identity W2C^T: cam==world. A fronto-parallel plane at z=5 facing -? :
    wvt = np.eye(4); wvt_inv = np.eye(4)
    H = Wd = 16
    depth = np.full((H, Wd), 5.0, np.float32)
    fx = fy = 16.0; cx = cy = 8.0
    cc = (np.array([0, 0, 0, 1.0]) @ wvt_inv)[:3]
    n = normal_from_depth_world(depth, fx, fy, cx, cy, wvt_inv)
    n = orient_to_camera(n, depth, fx, fy, cx, cy, wvt_inv, cc)
    nv = n[1:-1, 1:-1, :].reshape(-1, 3)
    # plane ⟂ optical axis → normal ≈ ±z; oriented-to-cam (cam at origin
    # looking +z, surface at z=5) → ray·n>0 flips → n points to -z (toward cam)
    med = np.median(nv, axis=0)
    print(f"  flat-plane normal median = {med.round(3)} (expect ≈ [0,0,-1])")
    assert abs(med[2]) > 0.9 and abs(med[0]) < 0.1 and abs(med[1]) < 0.1, \
        "normal-from-depth math broken"
    # angle of identical vectors = 0 ; opposite (post-orient same plane) small
    a = _angle_deg(np.array([[0, 0, 1.0]]), np.array([[0, 0, 1.0]]))
    assert a[0] < 1e-3, "angle self != 0"
    print("  Smoke PASS\n")


def main():
    if "--aggregate" in sys.argv:
        aggregate_and_print(); return
    if "--smoke" in sys.argv:
        smoke(); return
    print("=== Phase 16 C1 Gate — per-scene (NO-TRAIN GPU inference) ===")
    print(f"OUTPUT_ROOT={OUTPUT_ROOT} SEED={SEED} ITER={ITERATION}")
    print(f"SCENES={SCENES}  DSINE=vendored(scripts/dsine_pkg)")
    print(f"τ_dis=pct{TAU_DIS_PCT}(θ) τ_self={TAU_SELF_DEG}° "
          f"τ_err=pct{TAU_ERR_PCT} | GO: lift≥{LIFT_GO} & errM≥{ERRMASS_GO}")
    smoke()
    for sc in SCENES:
        print(f"──── {sc} ────")
        try:
            r = gate_scene(sc)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERR {sc}: {e}\n")
            continue
        if r is None:
            print(f"  SKIP {sc} (no checkpoint)\n"); continue
        _save(r)
        if r.get("status") == "NO_PIXELS":
            print(f"  NO_PIXELS\n"); continue
        print(f"  A3:{r.get('A3_status','?')} "
              f"flatM={r.get('A3_flat_med', float('nan')):.1f} "
              f"lift={r.get('A3_lift', float('nan')):.2f}  |  "
              f"DV:{r.get('DV_status','?')} "
              f"flatM={r.get('DV_flat_med', float('nan')):.1f} "
              f"lift={r.get('DV_lift', float('nan')):.2f}  |  "
              f"dcmp={r.get('dcmp_flatmed', float('nan')):.1f}°\n")
    print("Per-scene done. Aggregate verdict:")
    print("    python scripts/p16_normal_redundancy_gate.py --aggregate")


if __name__ == "__main__":
    main()
