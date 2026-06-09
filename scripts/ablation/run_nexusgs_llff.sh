#!/usr/bin/env bash
# ============================================================
# Ablation baseline 04 — NexusGS (CVPR 2025) trên LLFF 3-view
# Route HuggingFace: init+flow_depth bake sẵn → KHÔNG cần flow/MVS, KHÔNG đụng data Phase 22.
# Protocol: images_8 (≈-r8) + 30k iter + unified eval (corgs).
# Doc: docs/ablation/04_nexusgs.md
#
# Dùng (env nexus):
#   bash run_nexusgs_llff.sh smoke      # 1 scene fern
#   bash run_nexusgs_llff.sh full       # 8 scene, 2-GPU
#   bash run_nexusgs_llff.sh full1       # 8 scene, 1 GPU tuần tự (GPU=0)
#   bash run_nexusgs_llff.sh eval|agg
# ============================================================
set -u

REPO="/home/aidev/workspace/representation-3d/duyen/NexusGS"
ENV_NAME="nexus"
HF_SRC="Yukinoo/NexusGS-llff"          # auto-download per-scene revision
CORGS_REPO="/home/aidev/workspace/representation-3d/duyen/CoR-GS"
CORGS_ENV="corgs"
NV=3; ITER=30000
SPLIT_NUM=4; VALID_DIS=1.0; DROP_RATE=1.0; NEAR_N=2
GPU_A=0; SCENES_A=(fern flower fortress horns)
GPU_B=1; SCENES_B=(leaves orchids room trex)
ALL_SCENES=("${SCENES_A[@]}" "${SCENES_B[@]}")

MODE="${1:-full}"
OUT_DIR="$REPO/output/LLFF_ablation"
LOG_DIR="$CORGS_REPO/logs/ablation/nexusgs"
mkdir -p "$OUT_DIR" "$LOG_DIR"
LOGCSV="$LOG_DIR/timings_llff_${NV}v.csv"
LOGFILE="$LOG_DIR/nexusgs_llff_${NV}v_${MODE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1
echo ">> LOG: $LOGFILE · OUT: $OUT_DIR"

ENVS_DIR="${CONDA_PREFIX:+$(dirname "$CONDA_PREFIX")}"; ENVS_DIR="${ENVS_DIR:-$HOME/miniconda3/envs}"
CORGS_PY="$ENVS_DIR/$CORGS_ENV/bin/python"

python -c "import torch" 2>/dev/null || { echo "!! hãy: conda activate $ENV_NAME"; exit 1; }
echo ">> env active: ${CONDA_DEFAULT_ENV:-?}"; python -c "import torch,diff_gaussian_rasterization,simple_knn; print('>> torch',torch.__version__)" || exit 1

# ============================================================
# Per-scene: train 30k (HF init) -> render. M = OUT/<scene>/3_views
# ============================================================
run_scene() {
  local gpu=$1 scene=$2
  local M="$OUT_DIR/${scene}/${NV}_views"
  if [ -f "$M/point_cloud/iteration_${ITER}/point_cloud.ply" ]; then
    echo ">> [GPU$gpu] $scene — đã có model, SKIP (FORCE=1 train lại)"; [ -z "${FORCE:-}" ] && return 0
  fi
  local slog="$LOG_DIR/${scene}_${NV}v_$(date +%Y%m%d_%H%M%S).log"
  (
    exec > >(tee -a "$slog") 2>&1
    echo ">> [GPU$gpu] === $scene ===  ($(date '+%F %T'))  log -> $slog"
    local t0=$(date +%s)
    ( cd "$REPO" && CUDA_VISIBLE_DEVICES=$gpu python train.py \
          --source_path "$HF_SRC" --model_path "$M" --eval --n_views "$NV" \
          --save_iterations "$ITER" --iterations "$ITER" --densify_until_iter "$ITER" --position_lr_max_steps "$ITER" \
          --dataset_type llff --images images_8 --split_num "$SPLIT_NUM" \
          --valid_dis_threshold "$VALID_DIS" --drop_rate "$DROP_RATE" --near_n "$NEAR_N" \
          --huggingface --revision "$scene" ) || { echo "!! train $scene"; exit 1; }
    echo "$scene,$(( $(date +%s) - t0 ))" >> "$LOGCSV"
    ( cd "$REPO" && CUDA_VISIBLE_DEVICES=$gpu python render.py \
          --source_path "$HF_SRC" --model_path "$M" --iteration "$ITER" --render_depth --huggingface ) || { echo "!! render $scene"; exit 1; }
    echo ">> [GPU$gpu] $scene DONE ($(date '+%T'))"
  )
}

unified_eval() {
  local scenes=("$@"); [ ${#scenes[@]} -eq 0 ] && scenes=("${ALL_SCENES[@]}")
  echo ">> [eval] unified = $CORGS_REPO/metrics.py (env $CORGS_ENV)"
  for scene in "${scenes[@]}"; do
    local M="$OUT_DIR/${scene}/${NV}_views"; [ -d "$M/test" ] || { echo "   skip $scene (chưa render)"; continue; }
    ( cd "$CORGS_REPO" && "$CORGS_PY" metrics.py -s x -m "$M" ) || echo "!! eval $scene FAIL"
  done
}

aggregate() {
  OUT_DIR="$OUT_DIR" TIMINGS="$LOGCSV" ITER="$ITER" NV="$NV" python - <<'PY'
import json, os, glob
scenes=["fern","flower","fortress","horns","leaves","orchids","room","trex"]
out=os.environ["OUT_DIR"]; tim=os.environ["TIMINGS"]; it=os.environ["ITER"]; nv=os.environ["NV"]
times={}
if os.path.exists(tim):
    for ln in open(tim):
        if "," in ln: s,t=ln.strip().split(","); times[s]=int(t)
def ng(s):
    g=glob.glob(f"{out}/{s}/{nv}_views/point_cloud/iteration_*/point_cloud.ply")
    if not g: return None
    try:
        from plyfile import PlyData; return len(PlyData.read(sorted(g)[-1])["vertex"])
    except Exception: return None
hdr=f"{'scene':9s}{'PSNR':>8s}{'SSIM':>8s}{'SSIMsk':>8s}{'LPIPS':>8s}{'AVGE':>8s}{'N_gauss':>9s}{'train_s':>8s}"
print("\n"+hdr); print("-"*len(hdr))
Q={"PSNR":[],"SSIM":[],"SSIM_sk":[],"LPIPS":[],"AVGE":[]}; g_=[];t_=[]
for s in scenes:
    rp=f"{out}/{s}/{nv}_views/results.json"
    if not os.path.exists(rp): print(f"{s:9s}{'MISSING':>8s}"); continue
    m=list(json.load(open(rp)).values())[0]; n=ng(s); t=times.get(s)
    print(f"{s:9s}{m['PSNR']:8.3f}{m['SSIM']:8.4f}{m.get('SSIM_sk',float('nan')):8.4f}{m['LPIPS']:8.4f}{m.get('AVGE',float('nan')):8.4f}"
          f"{(str(n) if n else '-'):>9s}{(str(t) if t else '-'):>8s}")
    for k in Q:
        if k in m: Q[k].append(m[k])
    if n:g_.append(n)
    if t:t_.append(t)
if Q["PSNR"]:
    import statistics as st; A=lambda x:round(st.mean(x),3) if x else float('nan')
    print("-"*len(hdr))
    print(f"{'AVG':9s}{A(Q['PSNR']):8.3f}{A(Q['SSIM']):8.4f}{A(Q['SSIM_sk']):8.4f}{A(Q['LPIPS']):8.4f}{A(Q['AVGE']):8.4f}"
          f"{(str(round(st.mean(g_))) if g_ else '-'):>9s}{(str(round(st.mean(t_))) if t_ else '-'):>8s}")
    print(f"\n(n={len(Q['PSNR'])}/8 · images_8(~-r8) · {it} iter · eval=unified corgs)")
    print("Ref: ours 21.918 · Binocular 21.356 · FSGS 20.407 · CoR-GS 20.110")
PY
}

case "$MODE" in
  smoke) run_scene "$GPU_A" fern; unified_eval fern; aggregate ;;
  full)
    ( for s in "${SCENES_A[@]}"; do run_scene "$GPU_A" "$s"; done ) &
    ( for s in "${SCENES_B[@]}"; do run_scene "$GPU_B" "$s"; done ) &
    wait; unified_eval; aggregate ;;
  full1) for s in "${ALL_SCENES[@]}"; do run_scene "${GPU:-0}" "$s"; done; unified_eval; aggregate ;;
  eval) unified_eval; aggregate ;;
  agg)  aggregate ;;
  *) echo "Mode sai: $MODE (smoke|full|full1|eval|agg)"; exit 1 ;;
esac
echo ">> DONE ($MODE)"
