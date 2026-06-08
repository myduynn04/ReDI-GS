#!/usr/bin/env bash
# ============================================================
# Unified FPS bench — đo CÙNG 1 protocol trên Binocular3DGS + CRSGaussian
# để FPS infer SO THẲNG được (khác với speed.json mỗi repo đo kiểu khác nhau).
#
# Protocol (cố định cả 2): WARMUP=50 + TIMED=300 render (cycle test cams),
#   cuda.synchronize quanh timed loop, chỉ time render()["render"], -r8, no_grad.
#
# Output: $CRS_REPO/output/ablation/{binocular3dgs,crsgaussian}/<scene>.json
#         (KHÔNG đụng vào model_path để output sạch, dễ aggregate)
#
# Dùng:
#   chmod +x scripts/ablation/bench_fps_compare.sh
#   bash scripts/ablation/bench_fps_compare.sh
# ============================================================
set -u

RES="${RES:-8}"; NV=3
SCENES=(fern flower fortress horns leaves orchids room trex)

# ---- Paths ----
BINO_REPO="/home/aidev/workspace/representation-3d/duyen/Binocular3DGS"
BINO_OUT="$BINO_REPO/output/LLFF_ablation"            # <scene>_3views

CRS_REPO="/home/aidev/workspace/representation-3d/duyen/CoR-GS"

# CRSGaussian Phase 22 pilot — seed 42 representative (Option A: single-seed all metrics)
CRS_OUT="$CRS_REPO/output/p22_pilot"
CRS_SCENE_DIR='A3_seed42_${scene}'

# ── ABLATION OUTPUT dir — KHÔNG mix với training output dirs ──
ABLATION_OUT="$CRS_REPO/output/ablation"
BINO_ABL="$ABLATION_OUT/binocular3dgs"
CRS_ABL="$ABLATION_OUT/crsgaussian"
mkdir -p "$BINO_ABL" "$CRS_ABL"

# ── LOG dirs per-repo (siblings: binocular + crsgaussian) ──
# Mỗi repo có dir riêng, KHÔNG mix train logs với bench_fps logs.
# Existing: logs/ablation/binocular/ (đã có train logs của user)
# Mới:      logs/ablation/crsgaussian/ (mirror cấu trúc)
ABLATION_LOG="$CRS_REPO/logs/ablation"
BINO_LOG_DIR="$ABLATION_LOG/binocular"
CRS_LOG_DIR="$ABLATION_LOG/crsgaussian"
mkdir -p "$BINO_LOG_DIR" "$CRS_LOG_DIR"

TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$ABLATION_LOG/bench_fps_master_${TS}.log"
echo "[bench_fps_compare] start $(date)" | tee "$MASTER_LOG"

ENVS_DIR="/home/aidev/miniconda3/envs"
BINO_PY="$ENVS_DIR/binocular3dgs/bin/python"
CRS_PY="$ENVS_DIR/corgs/bin/python"

GPU="${GPU:-0}"
WARMUP=50; TIMED=300

# ============================================================
# bench_fps.py cho Binocular3DGS  (render(cam,g,pipe,bg))
# ============================================================
cat > "$BINO_REPO/bench_fps.py" <<PYEOF
import torch, os, time, json
from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
WARMUP=$WARMUP; TIMED=$TIMED
def bench(dataset, iteration, pipeline, out_path):
    with torch.no_grad():
        g = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, g, load_iteration=iteration, shuffle=False)
        bg = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
        cams = scene.getTestCameras(); n=len(cams); ng=int(g.get_xyz.shape[0])
        for i in range(WARMUP): render(cams[i%n], g, pipeline, bg)
        torch.cuda.synchronize(); t0=time.time()
        for i in range(TIMED): render(cams[i%n], g, pipeline, bg)["render"]
        torch.cuda.synchronize(); dt=time.time()-t0
        res={"fps":round(TIMED/dt,2),"ms_per_frame":round(1000*dt/TIMED,3),"n_gauss":ng,
             "n_test_views":n,"warmup":WARMUP,"timed":TIMED,"model_path":dataset.model_path}
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(res, open(out_path,"w"), indent=2)
        print(f"{os.path.basename(out_path)}: FPS={res['fps']} ms={res['ms_per_frame']} N={ng}")
if __name__=="__main__":
    p=ArgumentParser(); model=ModelParams(p,sentinel=True); pipe=PipelineParams(p)
    p.add_argument("--iteration",default=-1,type=int); p.add_argument("--n_views",default=3,type=int)
    p.add_argument("--dataset_name",default="LLFF",type=str); p.add_argument("--quiet",action="store_true")
    p.add_argument("--bench_out",required=True,type=str)
    a=get_combined_args(p); safe_state(a.quiet); bench(model.extract(a), a.iteration, pipe.extract(a), a.bench_out)
PYEOF

# ============================================================
# bench_fps.py cho CRSGaussian (CoR-GS)  (render(cam,g,pipe,bg,disable_dropout=True), GaussianModel(args))
# ============================================================
cat > "$CRS_REPO/bench_fps.py" <<PYEOF
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
        for i in range(WARMUP): render(cams[i%n], g, pipeline, bg, disable_dropout=True)
        torch.cuda.synchronize(); t0=time.time()
        for i in range(TIMED): render(cams[i%n], g, pipeline, bg, disable_dropout=True)["render"]
        torch.cuda.synchronize(); dt=time.time()-t0
        res={"fps":round(TIMED/dt,2),"ms_per_frame":round(1000*dt/TIMED,3),"n_gauss":ng,
             "n_test_views":n,"warmup":WARMUP,"timed":TIMED,"model_path":args.model_path}
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(res, open(out_path,"w"), indent=2)
        print(f"{os.path.basename(out_path)}: FPS={res['fps']} ms={res['ms_per_frame']} N={ng}")
if __name__=="__main__":
    p=ArgumentParser(); model=ModelParams(p,sentinel=True); pipe=PipelineParams(p)
    p.add_argument("--iteration",default=-1,type=int); p.add_argument("--quiet",action="store_true")
    p.add_argument("--bench_out",required=True,type=str)
    a=get_combined_args(p); safe_state(a.quiet); bench(a, pipe.extract(a), a.bench_out)
PYEOF

echo ">> protocol: warmup=$WARMUP timed=$TIMED  GPU=$GPU  RES=$RES" | tee -a "$MASTER_LOG"
echo ">> FPS json: $ABLATION_OUT/{binocular3dgs,crsgaussian}/<scene>.json" | tee -a "$MASTER_LOG"
echo ">> Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"

# ---- Run Binocular ----
echo ">> [Binocular3DGS] bench_fps ..." | tee -a "$MASTER_LOG"
for s in "${SCENES[@]}"; do
  M="$BINO_OUT/${s}_${NV}views"
  OUT="$BINO_ABL/${s}.json"
  LOG="$BINO_LOG_DIR/bench_fps_${s}_${TS}.log"
  [ -d "$M/point_cloud" ] || { echo "  skip $s (no model @ $M)" | tee -a "$MASTER_LOG"; continue; }
  ( cd "$BINO_REPO" && CUDA_VISIBLE_DEVICES=$GPU "$BINO_PY" bench_fps.py -m "$M" --bench_out "$OUT" --quiet ) 2>&1 | tee "$LOG" | tee -a "$MASTER_LOG"
done

# ---- Run CRSGaussian ----
echo ">> [CRSGaussian] bench_fps ..." | tee -a "$MASTER_LOG"
for s in "${SCENES[@]}"; do
  scene="$s"; eval "sub=\"$CRS_SCENE_DIR\""
  M="$CRS_OUT/$sub"
  OUT="$CRS_ABL/${s}.json"
  LOG="$CRS_LOG_DIR/bench_fps_${s}_${TS}.log"
  [ -d "$M/point_cloud" ] || { echo "  skip $s (no model @ $M)" | tee -a "$MASTER_LOG"; continue; }
  ( cd "$CRS_REPO" && CUDA_VISIBLE_DEVICES=$GPU "$CRS_PY" bench_fps.py -m "$M" --bench_out "$OUT" --quiet ) 2>&1 | tee "$LOG" | tee -a "$MASTER_LOG"
done

# ---- Aggregate: bảng FPS cạnh nhau ----
BINO_ABL="$BINO_ABL" CRS_ABL="$CRS_ABL" "$BINO_PY" - <<'PY'
import json, os
scenes=["fern","flower","fortress","horns","leaves","orchids","room","trex"]
ba=os.environ["BINO_ABL"]; ca=os.environ["CRS_ABL"]
def load(p):
    return json.load(open(p)) if os.path.exists(p) else None
print(f"\n{'scene':9s}{'Bino_FPS':>10s}{'Bino_ms':>9s}{'CRS_FPS':>9s}{'CRS_ms':>8s}{'Bino_Ng':>9s}{'CRS_Ng':>9s}")
print("-"*63)
bf=[];cf=[]
for s in scenes:
    b=load(f"{ba}/{s}.json")
    c=load(f"{ca}/{s}.json")
    bs=f"{b['fps']:.1f}" if b else "-"; bm=f"{b['ms_per_frame']:.2f}" if b else "-"
    cs=f"{c['fps']:.1f}" if c else "-"; cm=f"{c['ms_per_frame']:.2f}" if c else "-"
    bn=str(b['n_gauss']) if b else "-"; cn=str(c['n_gauss']) if c else "-"
    print(f"{s:9s}{bs:>10s}{bm:>9s}{cs:>9s}{cm:>8s}{bn:>9s}{cn:>9s}")
    if b: bf.append(b['fps'])
    if c: cf.append(c['fps'])
import statistics as st
print("-"*63)
mb=f"{st.mean(bf):.1f}" if bf else "-"; mc=f"{st.mean(cf):.1f}" if cf else "-"
print(f"{'MEAN':9s}{mb:>10s}{'':>9s}{mc:>9s}")
PY
echo ">> DONE bench_fps_compare $(date)" | tee -a "$MASTER_LOG"
echo "   FPS json: $ABLATION_OUT/{binocular3dgs,crsgaussian}/<scene>.json" | tee -a "$MASTER_LOG"
echo "   Logs:     $BINO_LOG_DIR/bench_fps_*.log" | tee -a "$MASTER_LOG"
echo "             $CRS_LOG_DIR/bench_fps_*.log" | tee -a "$MASTER_LOG"
echo "   Master:   $MASTER_LOG" | tee -a "$MASTER_LOG"
