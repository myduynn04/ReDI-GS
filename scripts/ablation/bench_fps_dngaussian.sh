#!/usr/bin/env bash
# ============================================================
# Unified FPS bench cho DNGaussian — CÙNG protocol Binocular/ours/CoR-GS/FSGS
# (WARMUP=50 + TIMED=300 render cycle test cams, cuda.synchronize, -r8, no_grad)
# → FPS DNGaussian so thẳng được với 266.6 (Bino) / 281.2 (FSGS) / 175.4 (ours).
#
# LƯU Ý DNGaussian khác các baseline khác:
#   - Load model = restore từ chkpnt_latest.pth (KHÔNG chỉ load_ply) → khôi phục neural_renderer
#   - render(cam, g, pipe, bg, inference=True) — neural_renderer (hashgrid-MLP) chạy mỗi frame
#     → FPS dự kiến CHẬM hơn nhóm GS-SH thuần rasterization (đây là điểm so sánh cho paper)
#   - PHẢI cd $REPO (gridencoder/shencoder torch-ngp chỉ import được từ repo dir)
#
# Output json: CoR-GS/output/ablation/dngaussian/<scene>.json
# Dùng (env dngaussian):  GPU=0 bash scripts/ablation/bench_fps_dngaussian.sh
# ============================================================
set -u
REPO="/home/aidev/workspace/representation-3d/duyen/DNGaussian"
DNG_OUT="$REPO/output/LLFF_ablation"
ABL_OUT="/home/aidev/workspace/representation-3d/duyen/CoR-GS/output/ablation/dngaussian"
DNG_PY="/home/aidev/miniconda3/envs/dngaussian/bin/python"
SCENES=(fern flower fortress horns leaves orchids room trex)
GPU="${GPU:-0}"; WARMUP=50; TIMED=300
mkdir -p "$ABL_OUT"

# bench_fps.py cho DNGaussian — load y hệt render.py (restore + keep_sigma), render inference=True
cat > "$REPO/bench_fps.py" <<PYEOF
import torch, os, time, json
from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
WARMUP=$WARMUP; TIMED=$TIMED
def bench(dataset, pipeline, iteration, out_path):
    with torch.no_grad():
        # --- load model GIỐNG render.py: restore chkpnt → khôi phục neural_renderer ---
        g = GaussianModel(dataset.sh_degree)
        (model_params, _) = torch.load(os.path.join(dataset.model_path, "chkpnt_latest.pth"))
        g.restore(model_params)
        g.neural_renderer.keep_sigma = True
        scene = Scene(dataset, g, load_iteration=iteration, shuffle=False)
        bg = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
        cams = scene.getTestCameras(); n=len(cams); ng=int(g.get_xyz.shape[0])
        # test render path = KHÔNG near-prune (giống render_set "test" near=0) → khớp ảnh đã eval
        for i in range(WARMUP): render(cams[i%n], g, pipeline, bg, inference=True)
        torch.cuda.synchronize(); t0=time.time()
        for i in range(TIMED): render(cams[i%n], g, pipeline, bg, inference=True)["render"]
        torch.cuda.synchronize(); dt=time.time()-t0
        res={"fps":round(TIMED/dt,2),"ms_per_frame":round(1000*dt/TIMED,3),"n_gauss":ng,
             "n_test_views":n,"warmup":WARMUP,"timed":TIMED}
        os.makedirs(os.path.dirname(out_path),exist_ok=True)
        json.dump(res, open(out_path,"w"), indent=2)
        print(f"{os.path.basename(out_path)}: FPS={res['fps']} ms={res['ms_per_frame']} N={ng}")
if __name__=="__main__":
    p=ArgumentParser(); model=ModelParams(p,sentinel=True); pipe=PipelineParams(p)
    p.add_argument("--iteration",default=-1,type=int); p.add_argument("--quiet",action="store_true")
    p.add_argument("--bench_out",required=True,type=str)
    a=get_combined_args(p); safe_state(a.quiet)
    bench(model.extract(a), pipe.extract(a), a.iteration, a.bench_out)
PYEOF

echo ">> DNGaussian bench: warmup=$WARMUP timed=$TIMED GPU=$GPU → $ABL_OUT"
for s in "${SCENES[@]}"; do
  M="$DNG_OUT/$s"
  [ -f "$M/chkpnt_latest.pth" ] || { echo "  skip $s (no chkpnt_latest.pth)"; continue; }
  ( cd "$REPO" && CUDA_VISIBLE_DEVICES=$GPU "$DNG_PY" bench_fps.py -m "$M" --bench_out "$ABL_OUT/$s.json" --quiet )
done

# bảng FPS DNGaussian
ABL_OUT="$ABL_OUT" "$DNG_PY" - <<'PY'
import json, os, statistics as st
a=os.environ["ABL_OUT"]; scenes=["fern","flower","fortress","horns","leaves","orchids","room","trex"]
print(f"\n{'scene':9s}{'FPS':>9s}{'ms':>8s}{'N_gauss':>10s}"); print("-"*36)
fp=[]
for s in scenes:
    p=f"{a}/{s}.json"
    if not os.path.exists(p): print(f"{s:9s}{'-':>9s}"); continue
    d=json.load(open(p)); print(f"{s:9s}{d['fps']:9.1f}{d['ms_per_frame']:8.2f}{d['n_gauss']:>10d}"); fp.append(d['fps'])
if fp: print("-"*36); print(f"{'MEAN':9s}{st.mean(fp):9.1f}")
print("\nRef FPS (unified): FSGS 281.2 · Binocular 266.6 · ours 175.4")
PY
echo ">> DONE DNGaussian FPS"
