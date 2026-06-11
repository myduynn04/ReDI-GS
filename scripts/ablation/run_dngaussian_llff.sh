#!/usr/bin/env bash
# ============================================================
# Ablation baseline 05 — DNGaussian (CVPR 2024) trên LLFF 3-view
# Protocol: -r8 + 6k iter + rand_pcd init (sparse) + MiDaS DPT depth. Unified eval (corgs).
# Doc: docs/ablation/05_dngaussian.md
#
# Data tree RIÊNG (copy sparse + symlink images) → KHÔNG đụng data Phase 22
# (DNGaussian ghi depth_maps/ + points3D_random.ply).
#
# Dùng (env dngaussian):
#   bash run_dngaussian_llff.sh prepare   # data tree + DPT depth gen (1 lần, cần GPU)
#   bash run_dngaussian_llff.sh smoke|full|full1|eval|agg
# ============================================================
set -u

REPO="/home/aidev/workspace/representation-3d/duyen/DNGaussian"
DATA="/home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data"
ENV_NAME="dngaussian"
CORGS_REPO="/home/aidev/workspace/representation-3d/duyen/CoR-GS"
CORGS_ENV="corgs"
NV=3; ITER=6000
GPU_A=0; SCENES_A=(fern flower fortress horns)
GPU_B=1; SCENES_B=(leaves orchids room trex)
ALL_SCENES=("${SCENES_A[@]}" "${SCENES_B[@]}")

MODE="${1:-full}"
OUT_DIR="$REPO/output/LLFF_ablation"
FDATA="$REPO/dataset_ablation/nerf_llff_data"     # data tree riêng
LOG_DIR="$CORGS_REPO/logs/ablation/dngaussian"
mkdir -p "$OUT_DIR" "$LOG_DIR"
LOGCSV="$LOG_DIR/timings_llff_${NV}v.csv"
LOGFILE="$LOG_DIR/dngaussian_llff_${NV}v_${MODE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1
echo ">> LOG: $LOGFILE · OUT: $OUT_DIR · FDATA: $FDATA"

ENVS_DIR="${CONDA_PREFIX:+$(dirname "$CONDA_PREFIX")}"; ENVS_DIR="${ENVS_DIR:-$HOME/miniconda3/envs}"
CORGS_PY="$ENVS_DIR/$CORGS_ENV/bin/python"

# ---- prepare: data tree + DPT depth ----
if [ "$MODE" = "prepare" ]; then
  for s in "${ALL_SCENES[@]}"; do
    src="$DATA/$s"; dst="$FDATA/$s"; mkdir -p "$dst"
    cp -r "$src/sparse" "$dst/sparse"                       # COPY (rand_pcd ghi points3D_random.ply vào đây)
    for item in images images_8 images_4 poses_bounds.npy; do
      [ -e "$src/$item" ] && ln -sfn "$src/$item" "$dst/$item"
    done
    echo "  $s tree -> $dst"
  done
  echo ">> DPT depth gen (MiDaS DPT_Hybrid, cached từ FSGS) — cần GPU ..."
  ( cd "$REPO/dpt" && CUDA_VISIBLE_DEVICES="${GPU:-0}" python get_depth_map_for_llff_dtu.py --root_path "$FDATA" --benchmark LLFF )
  echo ">> prepare DONE. depth_maps + sparse copy ở $FDATA (data gốc KHÔNG đụng)."
  exit 0
fi

python -c "import torch" 2>/dev/null || { echo "!! hãy: conda activate $ENV_NAME"; exit 1; }
echo ">> env: ${CONDA_DEFAULT_ENV:-?}"; python -c "import torch,diff_gaussian_rasterization,simple_knn; print('>> torch',torch.__version__)" || exit 1
# NOTE: gridencoder/shencoder (torch-ngp style) chỉ build .so, KHÔNG cài python package →
#   chỉ import được khi cwd ở trong DNGaussian/. train_llff.py chạy với `cd $REPO` nên OK.
#   ĐỪNG thêm gridencoder,shencoder vào guard này (guard chạy từ scripts/ablation → sẽ fail giả).

run_scene() {
  local gpu=$1 scene=$2
  local M="$OUT_DIR/${scene}"
  if [ -f "$M/point_cloud/iteration_${ITER}/point_cloud.ply" ]; then
    echo ">> [GPU$gpu] $scene — đã có model, SKIP (FORCE=1 train lại)"; [ -z "${FORCE:-}" ] && return 0
  fi
  local slog="$LOG_DIR/${scene}_${NV}v_$(date +%Y%m%d_%H%M%S).log"
  (
    exec > >(tee -a "$slog") 2>&1
    echo ">> [GPU$gpu] === $scene ===  ($(date '+%F %T'))  log -> $slog"
    local t0=$(date +%s)
    ( cd "$REPO" && CUDA_VISIBLE_DEVICES=$gpu python train_llff.py -s "$FDATA/$scene" --model_path "$M" \
          -r 8 --eval --n_sparse "$NV" --rand_pcd --iterations "$ITER" --lambda_dssim 0.2 \
          --densify_grad_threshold 0.0013 --prune_threshold 0.01 --densify_until_iter "$ITER" --percent_dense 0.01 \
          --position_lr_init 0.016 --position_lr_final 0.00016 --position_lr_max_steps 5500 --position_lr_start 500 \
          --split_opacity_thresh 0.1 --error_tolerance 0.00025 --scaling_lr 0.003 \
          --shape_pena 0.002 --opa_pena 0.001 --near 10 ) || { echo "!! train $scene"; exit 1; }
    echo "$scene,$(( $(date +%s) - t0 ))" >> "$LOGCSV"
    ( cd "$REPO" && CUDA_VISIBLE_DEVICES=$gpu python render.py -s "$FDATA/$scene" --model_path "$M" -r 8 --near 10 ) || { echo "!! render $scene"; exit 1; }
    echo ">> [GPU$gpu] $scene DONE ($(date '+%T'))"
  )
}

unified_eval() {
  local scenes=("$@"); [ ${#scenes[@]} -eq 0 ] && scenes=("${ALL_SCENES[@]}")
  echo ">> [eval] unified = $CORGS_REPO/metrics.py (env $CORGS_ENV)"
  for scene in "${scenes[@]}"; do
    local M="$OUT_DIR/${scene}"; [ -d "$M/test" ] || { echo "   skip $scene (chưa render)"; continue; }
    ( cd "$CORGS_REPO" && "$CORGS_PY" metrics.py -s x -m "$M" ) || echo "!! eval $scene FAIL"
  done
}

aggregate() {
  OUT_DIR="$OUT_DIR" TIMINGS="$LOGCSV" ITER="$ITER" python - <<'PY'
import json, os, glob
scenes=["fern","flower","fortress","horns","leaves","orchids","room","trex"]
out=os.environ["OUT_DIR"]; tim=os.environ["TIMINGS"]; it=os.environ["ITER"]
times={}
if os.path.exists(tim):
    for ln in open(tim):
        if "," in ln: s,t=ln.strip().split(","); times[s]=int(t)
def ng(s):
    g=glob.glob(f"{out}/{s}/point_cloud/iteration_*/point_cloud.ply")
    if not g: return None
    try:
        from plyfile import PlyData; return len(PlyData.read(sorted(g)[-1])["vertex"])
    except Exception: return None
hdr=f"{'scene':9s}{'PSNR':>8s}{'SSIM':>8s}{'SSIMsk':>8s}{'LPIPS':>8s}{'AVGE':>8s}{'N_gauss':>9s}{'train_s':>8s}"
print("\n"+hdr); print("-"*len(hdr))
Q={"PSNR":[],"SSIM":[],"SSIM_sk":[],"LPIPS":[],"AVGE":[]}; g_=[];t_=[]
for s in scenes:
    rp=f"{out}/{s}/results.json"
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
    print(f"\n(n={len(Q['PSNR'])}/8 · -r8 · {it} iter · rand_pcd · eval=unified corgs)")
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
  *) echo "Mode sai: $MODE (prepare|smoke|full|full1|eval|agg)"; exit 1 ;;
esac
echo ">> DONE ($MODE)"
