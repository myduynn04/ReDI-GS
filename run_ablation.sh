#!/bin/bash
# ============================================================
# [CRSGaussian] Ablation Runner
# File: CRSGaussian/run_ablation.sh
#
# USAGE:
#   bash run_ablation.sh <CONFIG> <SCENE> [GPU]
#
# EXAMPLES:
#   bash run_ablation.sh C7 fern            # C7 trên fern, GPU 0
#   bash run_ablation.sh W4 flower 1        # W4 trên flower, GPU 1
#   bash run_ablation.sh corgs room 0       # CoR-GS 2-field trên room
#   bash run_ablation.sh ALL fern 0         # Chạy TẤT CẢ configs trên fern
#   bash run_ablation.sh C7 ALL 0           # C7 trên TẤT CẢ scenes
#   bash run_ablation.sh ALL ALL 1          # Mọi thứ, GPU 1
#
# CONFIGS:
#   C0     = Baseline + CRS pruning (không informed init)
#   C7     = Informed init equal weights (0.33/0.33/0.34) + pruning
#   W1     = Depth heavy (0.2/0.6/0.2) + pruning
#   W2     = Reproj heavy (0.6/0.2/0.2) + pruning
#   W3     = View heavy (0.2/0.2/0.6) + pruning
#   W4     = No view, reproj+depth (0.5/0.5/off) + pruning
#   corgs  = CoR-GS 2-field gốc (gaussiansN=2, coreg, coprune)
#   ALL    = Chạy tất cả configs trên
#
# SCENES:
#   fern, flower, fortress, horns, leaves, orchids, room, trex
#   ALL = chạy tất cả 8 scenes
#
# OUTPUT:
#   logs/<scene>_<config>.log
#   output/<scene>_<config>/
#
# THAY DOI THAM SO:
#   Sửa phần "SHARED PARAMS" bên dưới để đổi iterations, n_views, v.v.
# ============================================================

set -e

# ── SHARED PARAMS — sửa ở đây ──
ITERATIONS=10000
N_VIEWS=3
RESOLUTION=8
DENSIFY_UNTIL=10000
DENSIFY_GRAD=0.0005
PSEUDO_INTERVAL=1
PSEUDO_START=500
TEST_ITERS="1000 3000 5000 10000"
DAV2_PATH="../Depth-Anything-V2"
DATA_ROOT="data/nerf_llff_data"

# ── Parse args ──
CONFIG=${1:?"Usage: bash run_ablation.sh <CONFIG> <SCENE> [GPU]"}
SCENE=${2:?"Usage: bash run_ablation.sh <CONFIG> <SCENE> [GPU]"}
GPU=${3:-0}

ALL_CONFIGS="C0 C7 W1 W2 W3 W4 corgs"
ALL_SCENES="fern flower fortress horns leaves orchids room trex"

# ── Resolve ALL ──
if [ "$CONFIG" = "ALL" ]; then
    configs="$ALL_CONFIGS"
else
    configs="$CONFIG"
fi

if [ "$SCENE" = "ALL" ]; then
    scenes="$ALL_SCENES"
else
    scenes="$SCENE"
fi

# ── Tạo logs dir ──
mkdir -p logs

# ── Run function ──
run_one() {
    local cfg=$1
    local scene=$2
    local tag="${scene}_${cfg}"
    local src="${DATA_ROOT}/${scene}"
    local out="output/${tag}"
    local log="logs/${tag}.log"

    echo "========================================"
    echo " Running: ${tag} | GPU=${GPU}"
    echo " Config: ${cfg} | Scene: ${scene}"
    echo " Output: ${out}"
    echo " Log: ${log}"
    echo "========================================"

    # Base args chung cho mọi config
    local base_args="--source_path ${src} -m ${out} --eval -r ${RESOLUTION} --n_views ${N_VIEWS} --random_background --iterations ${ITERATIONS} --densify_until_iter ${DENSIFY_UNTIL} --densify_grad_threshold ${DENSIFY_GRAD} --sample_pseudo_interval ${PSEUDO_INTERVAL} --start_sample_pseudo ${PSEUDO_START} --test_iterations ${TEST_ITERS}"

    local extra_args=""

    case $cfg in
        C0)
            # Baseline + CRS pruning, không informed init
            extra_args="--gaussiansN 1 --use_depth_prior --dav2_path ${DAV2_PATH} --use_crs_pruning"
            ;;
        C7)
            # Informed init equal weights + pruning
            extra_args="--gaussiansN 1 --use_depth_prior --dav2_path ${DAV2_PATH} --informed_crs_init --use_crs_pruning"
            ;;
        W1)
            # Depth heavy
            extra_args="--gaussiansN 1 --use_depth_prior --dav2_path ${DAV2_PATH} --informed_crs_init --use_crs_pruning --crs_init_w_reproj 0.2 --crs_init_w_depth 0.6 --crs_init_w_view 0.2"
            ;;
        W2)
            # Reproj heavy
            extra_args="--gaussiansN 1 --use_depth_prior --dav2_path ${DAV2_PATH} --informed_crs_init --use_crs_pruning --crs_init_w_reproj 0.6 --crs_init_w_depth 0.2 --crs_init_w_view 0.2"
            ;;
        W3)
            # View heavy
            extra_args="--gaussiansN 1 --use_depth_prior --dav2_path ${DAV2_PATH} --informed_crs_init --use_crs_pruning --crs_init_w_reproj 0.2 --crs_init_w_depth 0.2 --crs_init_w_view 0.6"
            ;;
        W4)
            # No view, reproj+depth only
            extra_args="--gaussiansN 1 --use_depth_prior --dav2_path ${DAV2_PATH} --informed_crs_init --use_crs_pruning --crs_init_use_view False --crs_init_w_reproj 0.5 --crs_init_w_depth 0.5 --crs_init_w_view 0.0"
            ;;
        corgs)
            # CoR-GS 2-field gốc — không dùng depth prior / CRS
            extra_args="--gaussiansN 2 --coreg --coprune"
            ;;
        *)
            echo "ERROR: Unknown config '${cfg}'"
            echo "Valid: C0 C7 W1 W2 W3 W4 corgs ALL"
            exit 1
            ;;
    esac

    CUDA_VISIBLE_DEVICES=${GPU} python train.py ${base_args} ${extra_args} 2>&1 | tee ${log}

    echo ""
    echo "[DONE] ${tag} → ${log}"
    echo ""
}

# ── Main loop ──
for scene in $scenes; do
    for cfg in $configs; do
        run_one "$cfg" "$scene"
    done
done

echo "========================================"
echo " ALL RUNS COMPLETE"
echo " Logs: logs/"
echo "========================================"
