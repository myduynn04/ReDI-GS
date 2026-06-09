#!/usr/bin/env bash
# ============================================================
# Ablation baseline 02 — FSGS (ICLR 2024) trên LLFF 3-view
# Protocol ta: images_8 (≈ -r8) + 10k iter + unified eval (corgs).
# Doc: docs/ablation/02_fsgs.md · Chuẩn: docs/ablation/00_index.md
#
# Init = <scene>/3_views/dense/fused.ply (COLMAP-MVS) — PHẢI là MVS gốc (xem mode check/restore).
# Depth = MiDaS DPT_Hybrid (đã cache). KHÔNG có bước triangulate.
#
# Dùng (env fsgs):
#   bash run_fsgs_llff.sh check     # xem fused.ply gốc/swap (in size + backup)
#   bash run_fsgs_llff.sh prepare   # tạo data tree RIÊNG cho FSGS (symlink + MVS) — KHÔNG đụng data gốc
#   bash run_fsgs_llff.sh smoke      # 1 scene fern
#   bash run_fsgs_llff.sh full       # 8 scene, 2-GPU
#   bash run_fsgs_llff.sh eval       # unified eval lại + bảng
#   bash run_fsgs_llff.sh agg        # in bảng
# ============================================================
set -u

REPO="/home/aidev/workspace/representation-3d/duyen/FSGS"
DATA="/home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data"
ENV_NAME="fsgs"
CORGS_REPO="/home/aidev/workspace/representation-3d/duyen/CoR-GS"
CORGS_ENV="corgs"
NV=3; ITER=10000
GPU_A=0; SCENES_A=(fern flower fortress horns)
GPU_B=1; SCENES_B=(leaves orchids room trex)
ALL_SCENES=("${SCENES_A[@]}" "${SCENES_B[@]}")

MODE="${1:-full}"
OUT_DIR="$REPO/output/LLFF_ablation"
FDATA="$REPO/dataset_ablation/nerf_llff_data"   # data tree RIÊNG cho FSGS (symlink + MVS fused.ply); KHÔNG đụng data gốc
LOG_DIR="$CORGS_REPO/logs/ablation/fsgs"
mkdir -p "$OUT_DIR" "$LOG_DIR"
LOGCSV="$LOG_DIR/timings_llff_${NV}v.csv"
LOGFILE="$LOG_DIR/fsgs_llff_${NV}v_${MODE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1
echo ">> LOG: $LOGFILE · OUT: $OUT_DIR · TIMINGS: $LOGCSV"

ENVS_DIR="${CONDA_PREFIX:+$(dirname "$CONDA_PREFIX")}"; ENVS_DIR="${ENVS_DIR:-$HOME/miniconda3/envs}"
CORGS_PY="$ENVS_DIR/$CORGS_ENV/bin/python"

# ---- check / restore fused.ply init ----
if [ "$MODE" = "check" ]; then
  for s in "${ALL_SCENES[@]}"; do
    f="$DATA/$s/3_views/dense/fused.ply"
    printf "%-9s " "$s"
    [ -f "$f" ] && printf "fused=%s " "$(du -h "$f" | cut -f1)" || printf "fused=MISSING "
    [ -f "$f.colmap_mvs_backup" ] && echo "[có MVS backup → fused.ply hiện ĐÃ SWAP]" || echo "[no backup]"
  done
  echo ">> Nếu có backup: chạy 'prepare' để tạo data tree RIÊNG cho FSGS (data gốc KHÔNG đụng)."
  exit 0
fi
if [ "$MODE" = "prepare" ]; then
  # Tạo data tree riêng cho FSGS: symlink images/sparse/poses (chỉ đọc) + copy MVS fused.ply.
  # Data Phase 22 gốc NGUYÊN VẸN (chỉ đọc qua symlink + đọc file .colmap_mvs_backup).
  for s in "${ALL_SCENES[@]}"; do
    src="$DATA/$s"; dst="$FDATA/$s"; mvs="$src/3_views/dense/fused.ply.colmap_mvs_backup"
    [ -f "$mvs" ] || { echo "  $s: KHÔNG có MVS backup → bỏ qua"; continue; }
    mkdir -p "$dst/3_views/dense"
    for item in sparse images images_8 images_4 poses_bounds.npy; do
      [ -e "$src/$item" ] && ln -sfn "$src/$item" "$dst/$item"
    done
    cp "$mvs" "$dst/3_views/dense/fused.ply"
    echo "  $s -> $dst (fused.ply=MVS $(du -h "$dst/3_views/dense/fused.ply"|cut -f1))"
  done
  echo ">> Data FSGS riêng tạo xong tại $FDATA — data gốc Phase 22 KHÔNG đụng."
  exit 0
fi

# ---- env guard ----
python -c "import torch" 2>/dev/null || { echo "!! hãy: conda activate $ENV_NAME"; exit 1; }
echo ">> env active: ${CONDA_DEFAULT_ENV:-?}"; python -c "import torch,diff_gaussian_rasterization,simple_knn; print('>> torch',torch.__version__)" || exit 1

# ============================================================
# Per-scene: train 10k -> render -> (eval riêng). Log riêng từng scene.
# ============================================================
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
    ( cd "$REPO" && CUDA_VISIBLE_DEVICES=$gpu python train.py -s "$FDATA/$scene" -m "$M" \
          --eval --n_views "$NV" --sample_pseudo_interval 1 ) || { echo "!! train $scene"; exit 1; }
    echo "$scene,$(( $(date +%s) - t0 ))" >> "$LOGCSV"
    ( cd "$REPO" && CUDA_VISIBLE_DEVICES=$gpu python render.py -s "$FDATA/$scene" -m "$M" \
          --iteration "$ITER" ) || { echo "!! render $scene"; exit 1; }
    echo ">> [GPU$gpu] $scene DONE ($(date '+%T'))"
  )
}

# ---- unified eval (corgs metrics.py trên renders FSGS) ----
unified_eval() {
  local scenes=("$@"); [ ${#scenes[@]} -eq 0 ] && scenes=("${ALL_SCENES[@]}")
  echo ">> [eval] unified = $CORGS_REPO/metrics.py (env $CORGS_ENV)"
  for scene in "${scenes[@]}"; do
    local M="$OUT_DIR/${scene}"; [ -d "$M/test" ] || { echo "   skip $scene (chưa render)"; continue; }
    ( cd "$CORGS_REPO" && "$CORGS_PY" metrics.py -s "$FDATA/$scene" -m "$M" ) || echo "!! eval $scene FAIL"
  done
}

# ---- aggregate (results.json + N_gauss từ ply + train csv; FPS điền sau bằng bench_fps_compare.sh) ----
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
    print(f"\n(n={len(Q['PSNR'])}/8 · images_8(~-r8) · {it} iter · eval=unified corgs)")
    print("Ref: ours 21.918 · Binocular 21.356 · FSGS paper ~20.31. FPS: chạy bench_fps_compare.sh.")
PY
}

case "$MODE" in
  smoke) run_scene "$GPU_A" fern; unified_eval fern; aggregate ;;
  full)
    ( for s in "${SCENES_A[@]}"; do run_scene "$GPU_A" "$s"; done ) &
    ( for s in "${SCENES_B[@]}"; do run_scene "$GPU_B" "$s"; done ) &
    wait; unified_eval; aggregate ;;
  full1)   # 8 scene TUẦN TỰ trên 1 GPU (mặc định GPU 0; đổi: GPU=1 bash ... full1)
    for s in "${ALL_SCENES[@]}"; do run_scene "${GPU:-0}" "$s"; done
    unified_eval; aggregate ;;
  eval) unified_eval; aggregate ;;
  agg)  aggregate ;;
  *) echo "Mode sai: $MODE (check|prepare|smoke|full|full1|eval|agg)"; exit 1 ;;
esac
echo ">> DONE ($MODE)"
