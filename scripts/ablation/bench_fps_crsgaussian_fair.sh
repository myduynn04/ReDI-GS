#!/usr/bin/env bash
# ============================================================
# FAIR FPS bench cho CRSGaussian (ours) — đo CẢ HAI để lộ overhead (a):
#   RAW  = render(...disable_dropout=True)            → = số cũ 175.4 (có pass color train-only)
#   FAIR = render_infer(...)                          → bỏ pass SH→RGB Python train-only (a),
#                                                        GIỮ rasterizer-confidence (chi phí (b) thật)
#
# render_infer() sao y ĐÚNG đường infer của render() (disable_dropout=True: dropout skip hết,
# apply_dropout=False, confidence không slice, opacity-gate RNRC giữ nguyên) — CHỈ bỏ block
# color (gaussian_renderer/__init__.py:224-228) vốn chỉ phục vụ CRS/co-reg lúc train, vô dụng infer.
# → KHÔNG đụng production render. Ảnh ["render"] giống hệt (color không ảnh hưởng rendered_image).
#
# Protocol GIỐNG các bench khác: WARMUP=50 + TIMED=300, cuda.synchronize, -r8, no_grad, cùng test cams/GPU.
# Output: CoR-GS/output/ablation/crsgaussian_fair/<scene>.json
# Dùng (env corgs):  GPU=0 bash scripts/ablation/bench_fps_crsgaussian_fair.sh
# ============================================================
set -u
CRS_REPO="/home/aidev/workspace/representation-3d/duyen/CoR-GS"
CRS_OUT="$CRS_REPO/output/p22_pilot"          # Phase 22 pilot, seed42 representative
CRS_SCENE_DIR='A3_seed42_${scene}'
ABL_OUT="$CRS_REPO/output/ablation/crsgaussian_fair"
CRS_PY="/home/aidev/miniconda3/envs/corgs/bin/python"
SCENES=(fern flower fortress horns leaves orchids room trex)
GPU="${GPU:-0}"; WARMUP=50; TIMED=300
mkdir -p "$ABL_OUT"

cat > "$CRS_REPO/bench_fps_fair.py" <<PYEOF
import torch, os, time, json, math
from scene import Scene
from gaussian_renderer import render, GaussianModel       # render thật (raw)
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
WARMUP=$WARMUP; TIMED=$TIMED

# ── render_infer: SAO Y đường infer của render() (disable_dropout=True) TRỪ block color (a) ──
def render_infer(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, device="cuda")
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    bg = bg_color
    if min(pc.bg_color.shape) != 0:
        bg = torch.tensor([0., 0., 0.]).cuda()
    # disable_dropout=True ⇒ apply_dropout=False ⇒ confidence KHÔNG slice
    confidence = pc.confidence if pipe.use_confidence else torch.ones_like(pc.confidence)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx, tanfovy=tanfovy, bg=bg, scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree, campos=viewpoint_camera.camera_center,
        prefiltered=False, debug=pipe.debug, confidence=confidence)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_xyz; means2D = screenspace_points; opacity = pc.get_opacity
    # opacity-gate RNRC (giữ nguyên như render thật — no-op nếu flag off)
    if (getattr(pc, "_rnrc_l3_active", False) and hasattr(pc, "_crs_rnrc")
            and pc._crs_rnrc.shape[0] == opacity.shape[0]):
        opacity = opacity * pc._crs_rnrc.detach().unsqueeze(-1)
    scales = pc.get_scaling; rotations = pc.get_rotation; shs = pc.get_features
    rendered_image, radii, depth, alpha = rasterizer(
        means3D=means3D, means2D=means2D, shs=shs, colors_precomp=None,
        opacities=opacity, scales=scales, rotations=rotations, cov3D_precomp=None)
    if min(pc.bg_color.shape) != 0:
        rendered_image = rendered_image + (1 - alpha) * torch.sigmoid(pc.bg_color)
    # ⬅ KHÔNG có block eval_sh/color (a) → đây là điểm khác duy nhất
    return {"render": rendered_image}

def fps_of(fn, cams, n):
    for i in range(WARMUP): fn(cams[i % n])
    torch.cuda.synchronize(); t0 = time.time()
    for i in range(TIMED): fn(cams[i % n])["render"]
    torch.cuda.synchronize(); return time.time() - t0

def bench(args, pipeline, out_path):
    with torch.no_grad():
        g = GaussianModel(args)
        scene = Scene(args, g, load_iteration=args.iteration, shuffle=False)
        bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        cams = scene.getTestCameras(); n = len(cams); ng = int(g.get_xyz.shape[0])
        # sanity: ảnh raw vs fair phải GIỐNG nhau (color không ảnh hưởng rendered_image)
        a = render(cams[0], g, pipeline, bg, disable_dropout=True)["render"]
        b = render_infer(cams[0], g, pipeline, bg)["render"]
        max_abs_diff = float((a - b).abs().max().item())
        dt_raw  = fps_of(lambda c: render(c, g, pipeline, bg, disable_dropout=True), cams, n)
        dt_fair = fps_of(lambda c: render_infer(c, g, pipeline, bg), cams, n)
        res = {"fps_raw": round(TIMED/dt_raw, 2),  "ms_raw":  round(1000*dt_raw/TIMED, 3),
               "fps_fair": round(TIMED/dt_fair, 2), "ms_fair": round(1000*dt_fair/TIMED, 3),
               "n_gauss": ng, "n_test_views": n, "warmup": WARMUP, "timed": TIMED,
               "img_max_abs_diff": max_abs_diff}
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(res, open(out_path, "w"), indent=2)
        print(f"{os.path.basename(out_path)}: RAW={res['fps_raw']} FAIR={res['fps_fair']} "
              f"N={ng} img_diff={max_abs_diff:.2e}")
if __name__ == "__main__":
    p = ArgumentParser(); model = ModelParams(p, sentinel=True); pipe = PipelineParams(p)
    p.add_argument("--iteration", default=-1, type=int); p.add_argument("--quiet", action="store_true")
    p.add_argument("--bench_out", required=True, type=str)
    a = get_combined_args(p); safe_state(a.quiet); bench(a, pipe.extract(a), a.bench_out)
PYEOF

echo ">> CRSGaussian FAIR bench: warmup=$WARMUP timed=$TIMED GPU=$GPU → $ABL_OUT"
for s in "${SCENES[@]}"; do
  scene="$s"; eval "sub=\"$CRS_SCENE_DIR\""
  M="$CRS_OUT/$sub"
  [ -d "$M/point_cloud" ] || { echo "  skip $s (no model @ $M)"; continue; }
  ( cd "$CRS_REPO" && CUDA_VISIBLE_DEVICES=$GPU "$CRS_PY" bench_fps_fair.py -m "$M" --bench_out "$ABL_OUT/$s.json" --quiet )
done

# bảng RAW vs FAIR
ABL_OUT="$ABL_OUT" "$CRS_PY" - <<'PY'
import json, os, statistics as st
a=os.environ["ABL_OUT"]; scenes=["fern","flower","fortress","horns","leaves","orchids","room","trex"]
print(f"\n{'scene':9s}{'RAW_fps':>9s}{'FAIR_fps':>10s}{'Δfps':>8s}{'N_gauss':>10s}{'img_diff':>11s}")
print("-"*57)
r=[];f=[]
for s in scenes:
    p=f"{a}/{s}.json"
    if not os.path.exists(p): print(f"{s:9s}{'-':>9s}"); continue
    d=json.load(open(p))
    print(f"{s:9s}{d['fps_raw']:9.1f}{d['fps_fair']:10.1f}{d['fps_fair']-d['fps_raw']:+8.1f}{d['n_gauss']:>10d}{d['img_max_abs_diff']:>11.1e}")
    r.append(d['fps_raw']); f.append(d['fps_fair'])
if r:
    print("-"*57)
    print(f"{'MEAN':9s}{st.mean(r):9.1f}{st.mean(f):10.1f}{st.mean(f)-st.mean(r):+8.1f}")
print("\nRef FPS (unified): DNGaussian 305.1 · FSGS 281.2 · Binocular 266.6")
print("img_diff = max|raw-fair| trên ảnh: phải ~0 (≤1e-5) → fair render KHÔNG đổi ảnh, chỉ bỏ (a)")
PY
echo ">> DONE CRSGaussian FAIR FPS"
