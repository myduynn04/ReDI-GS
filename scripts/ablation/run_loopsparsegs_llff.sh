#!/usr/bin/env bash
# ============================================================
# Ablation baseline 06 — LoopSparseGS (arXiv 2024) trên LLFF 3-view
# Protocol: images_8 (≈-r8) + multi-loop (4 rounds × 10k) + unified eval (corgs).
# Doc: docs/ablation/06_loopsparsegs.md
#
# KHÔNG đụng CRS/data gốc: data tree riêng + pre_llff bản patched riêng (gốc giữ nguyên).
# Cần COLMAP (ở env corgs) trên PATH cho pre_llff.
#
# Dùng (env loopsparsegs):
#   bash run_loopsparsegs_llff.sh tree                 # data tree riêng (copy sparse + symlink images)
#   LSGS_SCENES=fern bash run_loopsparsegs_llff.sh preproc   # patch pre_llff + COLMAP+depth (test fern)
#   bash run_loopsparsegs_llff.sh preproc               # pre_llff cho cả 8 scene
#   bash run_loopsparsegs_llff.sh smoke|full|full1|eval|agg
# ============================================================
set -u

REPO="/home/aidev/workspace/representation-3d/duyen/LoopSparseGS"
DATA="/home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data"
ENV_NAME="loopsparsegs"
CORGS_REPO="/home/aidev/workspace/representation-3d/duyen/CoR-GS"
CORGS_ENV="corgs"
COLMAP_BIN_DIR="/home/aidev/miniconda3/envs/corgs/bin"   # colmap nằm ở env corgs
NV=3
GPU_A=0; SCENES_A=(fern flower fortress horns)
GPU_B=1; SCENES_B=(leaves orchids room trex)
ALL_SCENES=("${SCENES_A[@]}" "${SCENES_B[@]}")

MODE="${1:-full}"
FDATA="$REPO/dataset_ablation/nerf_llff_data"   # data tree riêng
OUT_PREFIX="ab_"                                 # exp_name namespace → output/ab_<scene>_3
LOG_DIR="$CORGS_REPO/logs/ablation/loopsparsegs"
mkdir -p "$FDATA" "$LOG_DIR"
LOGCSV="$LOG_DIR/timings_llff_${NV}v.csv"
LOGFILE="$LOG_DIR/loopsparsegs_${NV}v_${MODE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1
echo ">> LOG: $LOGFILE · FDATA: $FDATA"

ENVS_DIR="${CONDA_PREFIX:+$(dirname "$CONDA_PREFIX")}"; ENVS_DIR="${ENVS_DIR:-$HOME/miniconda3/envs}"
CORGS_PY="$ENVS_DIR/$CORGS_ENV/bin/python"

# ---- tree: data tree riêng (copy sparse [writable] + symlink images) ----
if [ "$MODE" = "tree" ]; then
  for s in "${ALL_SCENES[@]}"; do
    src="$DATA/$s"; dst="$FDATA/$s"; mkdir -p "$dst"
    rm -rf "$dst/sparse"; cp -r "$src/sparse" "$dst/sparse"     # COPY (pre_llff ghi cameras.txt vào sparse/0)
    for item in images images_8 images_4 poses_bounds.npy; do
      [ -e "$src/$item" ] && ln -sfn "$src/$item" "$dst/$item"
    done
    echo "  $s tree -> $dst"
  done
  echo ">> tree DONE (data gốc KHÔNG đụng)."
  exit 0
fi

# ---- preproc: patch pre_llff (bản riêng) + chạy COLMAP+depth trên tree ----
if [ "$MODE" = "preproc" ]; then
  echo ">> patch pre_llff -> tools/pre_llff_ablation.py (gốc giữ nguyên) ..."
  ( cd "$REPO" && python - <<'PYEOF'
src="tools/pre_llff.py"; dst="tools/pre_llff_ablation.py"
s=open(src).read()
b=s
s=s.replace("COPY + r'..\\images\\\\' + img_name + r'  images\\\\' + img_name",
            "COPY + '../images/' + img_name + '  images/' + img_name")
s=s.replace("COPY + r'..\\sparse\\0\\cameras.txt created\\.'",
            "COPY + '../sparse/0/cameras.txt created/.'")
s=s.replace("open(r'created\\points3D.txt'", "open('created/points3D.txt'")
# COLMAP 3.13: SiftMatching.* -> FeatureMatching.*; guided_matching/max_num_matches removed; GPU1 đủ VRAM
s=s.replace("'colmap exhaustive_matcher --database_path database.db --SiftMatching.guided_matching 1 --SiftMatching.max_num_matches 32768'",
            "'colmap exhaustive_matcher --database_path database.db --FeatureMatching.gpu_index 1'")
s=s.replace("for scene in ['fern', 'flower', 'fortress',  'horns',  'leaves',  'orchids',  'room',  'trex']:",
            "import os as _os\n    for scene in _os.environ.get('LSGS_SCENES','fern flower fortress horns leaves orchids room trex').split():")
open(dst,"w").write(s)
print("  path-fix:", b!=s, "| matcher-fix:", "FeatureMatching.gpu_index" in s,
      "| LSGS_SCENES:", "LSGS_SCENES" in s)
# loop.py cũng re-run COLMAP (round 1-3) -> loop_ablation.py
l=open("tools/loop.py").read()
# (1) matcher: SiftMatching.* -> FeatureMatching.* (COLMAP 3.9+).
#     use_gpu 0 (CPU) vì matcher GPU OOM khi round cao (render nét -> nhiều feature) + contention 2-GPU.
#     CPU dùng RAM, an toàn tuyệt đối; loop colmap không phải bottleneck.
l=l.replace("--SiftMatching.guided_matching 1 --SiftMatching.max_num_matches 32768",
            "--FeatureMatching.use_gpu 0")
# (2) feature_extractor: cap threads tránh OOM khi 8-scene song song
l=l.replace("--SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1'",
            "--SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1 --FeatureExtraction.num_threads 4'")
# (3) CHÍ MẠNG: env loopsparsegs set LD_LIBRARY_PATH -> colmap load nhầm libsqlite3/libstdc++ cũ -> SQLite abort.
#     Ép mọi lệnh colmap dùng lib + binary của corgs.
_CM="LD_LIBRARY_PATH=/home/aidev/miniconda3/envs/corgs/lib /home/aidev/miniconda3/envs/corgs/bin/colmap "
l=l.replace("os.system('colmap ", "os.system('"+_CM)
open("tools/loop_ablation.py","w").write(l)
print("  loop matcher-fix:", "FeatureMatching.gpu_index" in l,
      "| threads-fix:", "FeatureExtraction.num_threads" in l,
      "| colmap-libenv:", l.count(_CM), "calls")
PYEOF
  )
  echo ">> run pre_llff_ablation (COLMAP + depth) trên tree (scenes=${LSGS_SCENES:-all 8}) ..."
  ( cd "$REPO" && PATH="$COLMAP_BIN_DIR:$PATH" python tools/pre_llff_ablation.py --source_path "$FDATA" --train_sub "$NV" )
  echo ">> preproc DONE."
  exit 0
fi

python -c "import torch" 2>/dev/null || { echo "!! hãy: conda activate $ENV_NAME"; exit 1; }
echo ">> env: ${CONDA_DEFAULT_ENV:-?}"; python -c "import torch,diff_gaussian_rasterization_weights,simple_knn; print('>> torch',torch.__version__)" || exit 1

# ---- multi-loop train 1 scene: round0(train+loop) ... round3(train) → output/ab_<scene>_3 ----
run_scene() {
  local gpu=$1 scene=$2
  local FINAL="$REPO/output/${OUT_PREFIX}${scene}_3"
  if [ -d "$FINAL/test" ]; then
    echo ">> [GPU$gpu] $scene — đã có output round3, SKIP (FORCE=1 chạy lại)"; [ -z "${FORCE:-}" ] && return 0
  fi
  local slog="$LOG_DIR/${scene}_${NV}v_$(date +%Y%m%d_%H%M%S).log"
  (
    exec > >(tee -a "$slog") 2>&1
    echo ">> [GPU$gpu] === $scene === multi-loop 4 rounds ($(date '+%F %T'))"
    S="$FDATA/$scene"; X="${OUT_PREFIX}${scene}"; t0=$(date +%s); PORT=$((6109 + gpu))   # cổng GUI riêng/GPU tránh conflict
    cd "$REPO"
    CUDA_VISIBLE_DEVICES=$gpu python train.py -s "$S" --exp_name "$X"   --eval --pseudo_loop_iters 0 --train_sub "$NV" --port "$PORT"        || { echo "!! r0 train $scene"; exit 1; }
    CUDA_VISIBLE_DEVICES=$gpu python tools/loop_ablation.py -s "$S" -m "output/$X"   -p 0                                            || { echo "!! r0 loop $scene"; exit 1; }
    CUDA_VISIBLE_DEVICES=$gpu python train.py -s "$S" --exp_name "${X}_1" --eval --pseudo_loop_iters 1 -sps --train_sub "$NV" --port "$PORT" || { echo "!! r1 train $scene"; exit 1; }
    CUDA_VISIBLE_DEVICES=$gpu python tools/loop_ablation.py -s "$S" -m "output/${X}_1" -p 1                                          || { echo "!! r1 loop $scene"; exit 1; }
    CUDA_VISIBLE_DEVICES=$gpu python train.py -s "$S" --exp_name "${X}_2" --eval --pseudo_loop_iters 2 -sps --train_sub "$NV" --port "$PORT" || { echo "!! r2 train $scene"; exit 1; }
    CUDA_VISIBLE_DEVICES=$gpu python tools/loop_ablation.py -s "$S" -m "output/${X}_2" -p 2                                          || { echo "!! r2 loop $scene"; exit 1; }
    CUDA_VISIBLE_DEVICES=$gpu python train.py -s "$S" --exp_name "${X}_3" --eval --pseudo_loop_iters 3 -sps --train_sub "$NV" --port "$PORT" || { echo "!! r3 train $scene"; exit 1; }
    echo "$scene,$(( $(date +%s) - t0 ))" >> "$LOGCSV"
    echo ">> [GPU$gpu] $scene DONE → output/${X}_3 ($(date '+%T'))"
  )
}

unified_eval() {
  local scenes=("$@"); [ ${#scenes[@]} -eq 0 ] && scenes=("${ALL_SCENES[@]}")
  echo ">> [eval] unified = $CORGS_REPO/metrics.py (env $CORGS_ENV)"
  for scene in "${scenes[@]}"; do
    local M="$REPO/output/${OUT_PREFIX}${scene}_3"; [ -d "$M/test" ] || { echo "   skip $scene (chưa render)"; continue; }
    ( cd "$CORGS_REPO" && "$CORGS_PY" metrics.py -s x -m "$M" ) || echo "!! eval $scene FAIL"
  done
}

aggregate() {
  REPO="$REPO" PREFIX="$OUT_PREFIX" TIMINGS="$LOGCSV" python - <<'PY'
import json, os, glob
scenes=["fern","flower","fortress","horns","leaves","orchids","room","trex"]
repo=os.environ["REPO"]; pre=os.environ["PREFIX"]; tim=os.environ["TIMINGS"]
times={}
if os.path.exists(tim):
    for ln in open(tim):
        if "," in ln: s,t=ln.strip().split(","); times[s]=int(t)
def ng(s):
    g=glob.glob(f"{repo}/output/{pre}{s}_3/point_cloud/iteration_*/point_cloud.ply")
    if not g: return None
    try:
        from plyfile import PlyData; return len(PlyData.read(sorted(g)[-1])["vertex"])
    except Exception: return None
hdr=f"{'scene':9s}{'PSNR':>8s}{'SSIM':>8s}{'SSIMsk':>8s}{'LPIPS':>8s}{'AVGE':>8s}{'N_gauss':>9s}{'train_s':>8s}"
print("\n"+hdr); print("-"*len(hdr))
Q={"PSNR":[],"SSIM":[],"SSIM_sk":[],"LPIPS":[],"AVGE":[]}; g_=[];t_=[]
for s in scenes:
    rp=f"{repo}/output/{pre}{s}_3/results.json"
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
    print("\nRef: ours 21.918 · Binocular 21.356 · FSGS 20.407 · CoR-GS 20.110")
PY
}

case "$MODE" in
  smoke) run_scene "$GPU_A" fern; unified_eval fern; aggregate ;;
  full)
    ( for s in "${SCENES_A[@]}"; do run_scene "$GPU_A" "$s"; done ) &
    ( for s in "${SCENES_B[@]}"; do run_scene "$GPU_B" "$s"; done ) &
    wait; unified_eval; aggregate ;;
  full1)  # RUN_SCENES="flower fortress ..." để chọn scene (vd loại fern đang chạy GPU khác)
    if [ -n "${RUN_SCENES:-}" ]; then SC=($RUN_SCENES); else SC=("${ALL_SCENES[@]}"); fi
    for s in "${SC[@]}"; do run_scene "${GPU:-0}" "$s"; done; unified_eval; aggregate ;;
  eval) unified_eval; aggregate ;;
  agg)  aggregate ;;
  *) echo "Mode sai: $MODE (tree|preproc|smoke|full|full1|eval|agg)"; exit 1 ;;
esac
echo ">> DONE ($MODE)"
