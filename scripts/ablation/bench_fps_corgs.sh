#!/usr/bin/env bash
# ============================================================
# Unified FPS bench cho CoR-GS (baseline 2-field) — CÙNG protocol.
# Inference render field gs0 (point_cloud/) = single-field → FPS fair, comparable.
# Repo: CoR-GS_goc (baseline gốc). Models: output/llff/<scene>/.
# Output json: CoR-GS/output/ablation/corgs/<scene>.json
# Dùng:  GPU=0 bash scripts/ablation/bench_fps_corgs.sh
# (đổi CORGS_ENV nếu CoR-GS_goc dùng env khác)
# ============================================================
set -u
GOC_REPO="/home/aidev/workspace/representation-3d/duyen/CoR-GS_goc"
GOC_OUT="$GOC_REPO/output/llff"
ABL_OUT="/home/aidev/workspace/representation-3d/duyen/CoR-GS/output/ablation/corgs"
CORGS_ENV="${CORGS_ENV:-corgs}"
GOC_PY="/home/aidev/miniconda3/envs/$CORGS_ENV/bin/python"
SCENES=(fern flower fortress horns leaves orchids room trex)
GPU="${GPU:-0}"; WARMUP=50; TIMED=300
mkdir -p "$ABL_OUT"

# bench_fps.py: GaussianModel(args), Scene(args,...), render(cam,g,pipe,bg) [render gs0 single-field]
cat > "$GOC_REPO/bench_fps.py" <<PYEOF
import torch, os, time, json
from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
WARMUP=$WARMUP; TIMED=$TIMED
def bench(args, pipeline, out_path):
    with torch.no_grad():
        g = GaussianModel(args)
        scene = Scene(args, g, load_iteration=args.iteration, shuffle=False)
        bg = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
        cams = scene.getTestCameras(); n=len(cams); ng=int(g.get_xyz.shape[0])
        for i in range(WARMUP): render(cams[i%n], g, pipeline, bg)
        torch.cuda.synchronize(); t0=time.time()
        for i in range(TIMED): render(cams[i%n], g, pipeline, bg)["render"]
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
    a=get_combined_args(p); safe_state(a.quiet); bench(a, pipe.extract(a), a.bench_out)
PYEOF

echo ">> CoR-GS bench: warmup=$WARMUP timed=$TIMED GPU=$GPU env=$CORGS_ENV → $ABL_OUT"
for s in "${SCENES[@]}"; do
  M="$GOC_OUT/$s"; [ -d "$M/point_cloud" ] || { echo "  skip $s (no model @ $M)"; continue; }
  ( cd "$GOC_REPO" && CUDA_VISIBLE_DEVICES=$GPU "$GOC_PY" bench_fps.py -m "$M" --bench_out "$ABL_OUT/$s.json" --quiet )
done

# bảng FPS CoR-GS
ABL_OUT="$ABL_OUT" "$GOC_PY" - <<'PY'
import json, os, statistics as st
a=os.environ["ABL_OUT"]; scenes=["fern","flower","fortress","horns","leaves","orchids","room","trex"]
print(f"\n{'scene':9s}{'FPS':>9s}{'ms':>8s}{'N_gauss':>10s}"); print("-"*36)
fp=[]
for s in scenes:
    p=f"{a}/{s}.json"
    if not os.path.exists(p): print(f"{s:9s}{'-':>9s}"); continue
    d=json.load(open(p)); print(f"{s:9s}{d['fps']:9.1f}{d['ms_per_frame']:8.2f}{d['n_gauss']:>10d}"); fp.append(d['fps'])
if fp: print("-"*36); print(f"{'MEAN':9s}{st.mean(fp):9.1f}")
print("\nRef FPS (unified): Binocular 266.6 · ours 175.4 · (FSGS đo riêng)")
PY
echo ">> DONE CoR-GS FPS"
