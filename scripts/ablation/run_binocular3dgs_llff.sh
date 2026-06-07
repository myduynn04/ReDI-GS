#!/usr/bin/env bash
# ============================================================
# Ablation baseline 01 — Binocular3DGS (NeurIPS 2024) trên LLFF 3-view
# Chạy ở protocol CRSGaussian (-r 8) + ĐÁNH GIÁ BẰNG EVALUATOR CHUNG.
# Doc: docs/ablation/01_binocular3dgs.md · Chuẩn chung: docs/ablation/00_index.md
#
# CHUẨN CHUNG:
#   - render bằng repo baseline (env binocular3dgs)
#   - eval quality bằng MỘT evaluator duy nhất = CoR-GS/CRSGaussian metrics.py (env corgs)
#     => PSNR / SSIM(GS) / SSIM_sk / LPIPS(vgg) / AVGE / AVGE_sk đồng nhất mọi baseline
#   - efficiency (bonus): train time + FPS(infer) + N_gauss + model size(MB) + peak VRAM
#
# Dùng (trên SERVER):
#   bash run_binocular3dgs_llff.sh build   # build 2 CUDA ext + sinh bench script
#   bash run_binocular3dgs_llff.sh smoke    # 1 scene fern
#   bash run_binocular3dgs_llff.sh full     # 8 scene, 2-GPU
#   bash run_binocular3dgs_llff.sh eval      # chỉ chạy lại unified-eval (corgs)
#   bash run_binocular3dgs_llff.sh agg       # chỉ in bảng tổng
# Lưu log: thêm  2>&1 | tee ~/bino_<mode>.log   (vd: bash ... full 2>&1 | tee ~/bino_full.log)
# Override: RES=2 bash run_binocular3dgs_llff.sh full
# ============================================================
set -u

# ---- Config ----
REPO="/home/aidev/workspace/representation-3d/duyen/Binocular3DGS"
DATA="/home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data"
ENV_NAME="binocular3dgs"                 # env render/train của baseline
CORGS_REPO="/home/aidev/workspace/representation-3d/duyen/CoR-GS"   # = codebase CRSGaussian
CORGS_ENV="corgs"                        # env có evaluator chung (skimage/lpips/avge)
RES="${RES:-8}"                          # -r 8 = chuẩn LLFF sparse-view
N_VIEWS=3
ITER=30000
GPU_A=0; SCENES_A=(fern flower fortress horns)
GPU_B=1; SCENES_B=(leaves orchids room trex)
ALL_SCENES=("${SCENES_A[@]}" "${SCENES_B[@]}")
MODE="${1:-full}"

# Output baseline tách riêng (KHÔNG đụng output/LLFF cũ/bẩn).
OUT_DIR="$REPO/output/LLFF_ablation"
# Log + timings ở CoR-GS/logs/ablation/<model>.
LOG_DIR="$CORGS_REPO/logs/ablation/binocular"
mkdir -p "$OUT_DIR" "$LOG_DIR"
LOGCSV="$LOG_DIR/timings_llff_r${RES}.csv"

# ---- Auto-log: output vừa in màn hình vừa lưu file ----
LOGFILE="$LOG_DIR/binocular_llff_r${RES}_${MODE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1
echo "=================================================================="
echo ">> LOG file : $LOGFILE"
echo ">> KẾT QUẢ  : $OUT_DIR/<scene>_3views/"
echo ">>            ├─ results.json   (PSNR/SSIM/SSIM_sk/LPIPS/AVGE)"
echo ">>            ├─ speed.json     (N_gauss/FPS/VRAM)"
echo ">>            ├─ point_cloud/iteration_30000/point_cloud.ply"
echo ">>            └─ test/ours_30000/{renders,gt}/"
echo ">> TIMINGS  : $LOGCSV"
echo "=================================================================="

# ⚠️ TỰ activate TRƯỚC khi chạy:  conda activate binocular3dgs
# Script KHÔNG tự activate (bash con không load conda từ zsh). Chỉ kiểm tra + dùng env đang active.
python -c "import torch" 2>/dev/null || { echo "!! chưa thấy python/torch — hãy: conda activate $ENV_NAME  rồi chạy lại"; exit 1; }
echo ">> env active: ${CONDA_DEFAULT_ENV:-?}  RES=$RES  REPO=$REPO  eval-env=$CORGS_ENV"
python -c "import torch; print('>> torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available())" || exit 1

# Python của env evaluator chung (corgs) — gọi trực tiếp binary, KHÔNG cần conda activate/run.
ENVS_DIR="${CONDA_PREFIX:+$(dirname "$CONDA_PREFIX")}"; ENVS_DIR="${ENVS_DIR:-$HOME/miniconda3/envs}"
CORGS_PY="$ENVS_DIR/$CORGS_ENV/bin/python"
[ -x "$CORGS_PY" ] || echo "!! cảnh báo: không thấy $CORGS_PY (unified eval sẽ fail — sửa CORGS_ENV/ENVS_DIR)"

# ============================================================
# Stage 0a — build 2 CUDA ext (idempotent)
# ============================================================
ensure_build() {
  if python -c "import diff_gaussian_rasterization, simple_knn" 2>/dev/null; then
    echo ">> [build] đã có rasterizer + simple_knn — skip"; return 0; fi
  echo ">> [build] building (--no-build-isolation: dùng torch của env hiện tại) ..."
  # --no-build-isolation BẮT BUỘC: setup.py import torch; pip PEP517 build-env cô lập KHÔNG có torch.
  # Env này: torch 2.4.1+cu121 → cần nvcc 12.x (CUDA_HOME trỏ cuda-12.x).
  pip install --no-build-isolation "$REPO/submodules/diff-gaussian-rasterization" || { echo "!! build rasterizer FAIL (check nvcc 12.x + CUDA_HOME)"; exit 1; }
  pip install --no-build-isolation "$REPO/submodules/simple-knn"                   || { echo "!! build simple-knn FAIL"; exit 1; }
  python -c "import diff_gaussian_rasterization, simple_knn; print('>> [build] OK')" || exit 1
}

# ============================================================
# Stage 0b — sinh bench_render_speed.py (đo FPS + VRAM) vào $REPO
# ============================================================
ensure_bench() {
  cat > "$REPO/bench_render_speed.py" <<'PY'
# [ablation] đo FPS render + N_gauss + peak VRAM -> speed.json. Mô phỏng render.py của repo.
import torch, os, time, json
from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

def bench(dataset, iteration, pipeline, warmup=20, repeats=5):
    with torch.no_grad():
        g = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, g, load_iteration=iteration, shuffle=False)
        bg = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
        cams = scene.getTestCameras()
        ng = int(g.get_xyz.shape[0])
        for _ in range(warmup): render(cams[0], g, pipeline, bg)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time(); n = 0
        for _ in range(repeats):
            for c in cams: render(c, g, pipeline, bg)["render"]; n += 1
        torch.cuda.synchronize(); dt = time.time() - t0
        vram = round(torch.cuda.max_memory_allocated()/1e6, 1)   # peak VRAM lúc render
        res = {"n_gauss": ng, "fps": round(n/dt,2), "ms_per_frame": round(1000*dt/n,3),
               "vram_mb": vram, "n_test_views": len(cams)}
        json.dump(res, open(os.path.join(dataset.model_path, "speed.json"), "w"), indent=2)
        print(dataset.model_path, res)

if __name__ == "__main__":
    p = ArgumentParser(); model = ModelParams(p, sentinel=True); pipe = PipelineParams(p)
    p.add_argument("--iteration", default=-1, type=int); p.add_argument("--n_views", default=3, type=int)
    p.add_argument("--dataset_name", default="LLFF", type=str); p.add_argument("--quiet", action="store_true")
    a = get_combined_args(p); safe_state(a.quiet); bench(model.extract(a), a.iteration, pipe.extract(a))
PY
  echo ">> [bench] wrote $REPO/bench_render_speed.py"
}

# ============================================================
# Per-scene RENDER PASS (env binocular3dgs): triangulate -> train -> render -> bench
# KHÔNG eval ở đây (eval làm riêng bằng evaluator chung).
# ============================================================
run_scene() {
  local gpu=$1 scene=$2
  local M="$OUT_DIR/${scene}_${N_VIEWS}views"
  local slog="$LOG_DIR/${scene}_r${RES}_$(date +%Y%m%d_%H%M%S).log"
  (  # toàn bộ scene chạy trong 1 subshell, log riêng ra $slog (vẫn hiện màn hình)
    exec > >(tee -a "$slog") 2>&1
    echo ">> [GPU$gpu] === $scene ===  ($(date '+%F %T'))  log -> $slog"
    ( cd "$REPO/submodules/dense_matcher" && CUDA_VISIBLE_DEVICES=$gpu python triangulate.py \
          --data_path "$DATA/$scene" --output_path "$REPO/keypoints_to_3d/LLFF" \
          --resolution "$RES" --dataset_name LLFF --n_views "$N_VIEWS" ) || { echo "!! triangulate $scene"; exit 1; }
    t0=$(date +%s)
    ( cd "$REPO" && CUDA_VISIBLE_DEVICES=$gpu python train.py -s "$DATA/$scene" -m "$M" \
          --n_views "$N_VIEWS" --dataset_name LLFF --resolution "$RES" --eval ) || { echo "!! train $scene"; exit 1; }
    echo "$scene,$(( $(date +%s) - t0 ))" >> "$LOGCSV"
    ( cd "$REPO" && CUDA_VISIBLE_DEVICES=$gpu python render.py -m "$M" \
          --n_views "$N_VIEWS" --skip_train --resolution "$RES" --eval --dataset_name LLFF ) || { echo "!! render $scene"; exit 1; }
    ( cd "$REPO" && CUDA_VISIBLE_DEVICES=$gpu python bench_render_speed.py -m "$M" \
          --n_views "$N_VIEWS" --dataset_name LLFF --quiet ) || echo "!! bench $scene (bỏ qua)"
    echo ">> [GPU$gpu] $scene DONE ($(date '+%T'))"
  )
}

# ============================================================
# UNIFIED EVAL (env corgs): chạy CRSGaussian metrics.py trên renders của baseline.
# => results.json đồng bộ: PSNR/SSIM/SSIM_sk/LPIPS/AVGE/AVGE_sk. Ghi đè metrics riêng của repo.
# ============================================================
unified_eval() {
  local scenes=("$@"); [ ${#scenes[@]} -eq 0 ] && scenes=("${ALL_SCENES[@]}")
  echo ">> [eval] unified evaluator = $CORGS_REPO/metrics.py (env $CORGS_ENV)"
  for scene in "${scenes[@]}"; do
    local M="$OUT_DIR/${scene}_${N_VIEWS}views"
    [ -d "$M/test" ] || { echo "   skip $scene (chưa render)"; continue; }
    ( cd "$CORGS_REPO" && "$CORGS_PY" metrics.py -s "$DATA/$scene" -m "$M" ) \
        || echo "!! unified eval $scene FAIL"
  done
}

# ============================================================
# AGGREGATE — gom results.json (đủ bộ) + speed.json + timings + ply size
# ============================================================
aggregate() {
  OUT_DIR="$OUT_DIR" TIMINGS="$LOGCSV" RES="$RES" N_VIEWS="$N_VIEWS" ITER="$ITER" python - <<'PY'
import json, os, glob
scenes=["fern","flower","fortress","horns","leaves","orchids","room","trex"]
nv=os.environ.get("N_VIEWS","3"); res=os.environ.get("RES","8"); it=os.environ.get("ITER","30000")
out=os.environ["OUT_DIR"]; tim=os.environ["TIMINGS"]
times={}
if os.path.exists(tim):
    for ln in open(tim):
        if "," in ln: s,t=ln.strip().split(","); times[s]=int(t)
def ply_mb(s):
    g=glob.glob(f"{out}/{s}_{nv}views/point_cloud/iteration_*/point_cloud.ply")
    return round(os.path.getsize(sorted(g)[-1])/1e6,1) if g else None
hdr=f"{'scene':9s}{'PSNR':>8s}{'SSIM':>8s}{'SSIMsk':>8s}{'LPIPS':>8s}{'AVGE':>8s}{'N_gauss':>9s}{'FPS':>7s}{'MB':>6s}{'VRAM':>8s}{'train_s':>8s}"
print("\n"+hdr); print("-"*len(hdr))
Q={"PSNR":[],"SSIM":[],"SSIM_sk":[],"LPIPS":[],"AVGE":[]}; ng=[];fp=[];mb=[];vr=[];tt=[]
for s in scenes:
    rp=f"{out}/{s}_{nv}views/results.json"; sp=f"{out}/{s}_{nv}views/speed.json"
    if not os.path.exists(rp): print(f"{s:9s}{'MISSING':>8s}"); continue
    m=list(json.load(open(rp)).values())[0]
    sd=json.load(open(sp)) if os.path.exists(sp) else {}
    g=sd.get("n_gauss"); f=sd.get("fps"); v=sd.get("vram_mb"); size=ply_mb(s); t=times.get(s)
    print(f"{s:9s}{m['PSNR']:8.3f}{m['SSIM']:8.4f}{m.get('SSIM_sk',float('nan')):8.4f}{m['LPIPS']:8.4f}{m.get('AVGE',float('nan')):8.4f}"
          f"{(str(g) if g else '-'):>9s}{(str(f) if f else '-'):>7s}{(str(size) if size else '-'):>6s}{(str(v) if v else '-'):>8s}{(str(t) if t else '-'):>8s}")
    for k in Q:
        if k in m: Q[k].append(m[k])
    if g:ng.append(g)
    if f:fp.append(f)
    if size:mb.append(size)
    if v:vr.append(v)
    if t:tt.append(t)
if Q["PSNR"]:
    import statistics as st
    A=lambda x:round(st.mean(x),3) if x else float('nan')
    print("-"*len(hdr))
    print(f"{'AVG':9s}{A(Q['PSNR']):8.3f}{A(Q['SSIM']):8.4f}{A(Q['SSIM_sk']):8.4f}{A(Q['LPIPS']):8.4f}{A(Q['AVGE']):8.4f}"
          f"{(str(round(st.mean(ng))) if ng else '-'):>9s}{(str(A(fp))):>7s}{(str(A(mb))):>6s}{(str(A(vr))):>8s}{(str(round(st.mean(tt))) if tt else '-'):>8s}")
    print(f"\n(n={len(Q['PSNR'])}/8 · -r{res} · {it} iter · eval=unified corgs/metrics.py · FPS@-r{res})")
    print("CRSGaussian ref (ours, -r8, 10k): PSNR 21.89")
    print("Cột SSIM_sk/AVGE_sk đầy đủ nằm trong từng results.json.")
PY
}

# ---- Dispatch ----
case "$MODE" in
  build) ensure_build; ensure_bench ;;
  smoke) ensure_build; ensure_bench; : > "$LOGCSV"; run_scene "$GPU_A" fern; unified_eval fern; aggregate ;;
  full)
    ensure_build; ensure_bench; : > "$LOGCSV"
    ( for s in "${SCENES_A[@]}"; do run_scene "$GPU_A" "$s"; done ) &
    ( for s in "${SCENES_B[@]}"; do run_scene "$GPU_B" "$s"; done ) &
    wait
    unified_eval
    aggregate ;;
  eval) unified_eval; aggregate ;;
  agg)  aggregate ;;
  *) echo "Mode sai: $MODE (build|smoke|full|eval|agg)"; exit 1 ;;
esac
echo ">> DONE ($MODE)"
