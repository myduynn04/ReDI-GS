#!/bin/bash
# ============================================================
# [CRSGaussian DIAG E2B] CRS Distribution Analysis Wrapper
# File: scripts/diagnose_crs_distribution.sh
# Mục đích: Wrap diagnose_crs_distribution.py với best config args
#
# USAGE:
#   bash scripts/diagnose_crs_distribution.sh [SCENE] [MODEL_DIR] [ITER]
#
# EXAMPLES:
#   bash scripts/diagnose_crs_distribution.sh fern
#   bash scripts/diagnose_crs_distribution.sh fern output/ablation_pseudo_photo/B0_fern 10000
#   bash scripts/diagnose_crs_distribution.sh ALL    # tất cả 8 LLFF scenes
# ============================================================

SCENE=${1:-fern}
MODEL_DIR=${2:-""}
ITER=${3:-10000}

DATA_ROOT="data/nerf_llff_data"
DAV2_PATH="../Depth-Anything-V2"
LOGDIR="logs/diag_e2b"
mkdir -p ${LOGDIR}

ALL_SCENES="fern flower fortress horns leaves orchids room trex"

run_one() {
    local scene=$1
    local model_dir=$2
    local iter=$3

    # Auto-detect model_dir nếu chưa truyền
    if [ -z "$model_dir" ]; then
        # Thử các path candidate theo thứ tự ưu tiên
        for cand in \
            "output/ablation_pseudo_photo/B0_${scene}" \
            "output/ablation_weights/S1_WG_${scene}" \
            "output/ablation_weights_2/S1_WG_${scene}"; do
            if [ -f "${cand}/point_cloud/iteration_${iter}/point_cloud.ply" ]; then
                model_dir=$cand
                break
            fi
        done
    fi

    if [ -z "$model_dir" ] || [ ! -f "${model_dir}/point_cloud/iteration_${iter}/point_cloud.ply" ]; then
        echo "[SKIP] ${scene}: no trained model found at iteration ${iter}"
        echo "       Tried: output/ablation_pseudo_photo/B0_${scene}"
        echo "              output/ablation_weights/S1_WG_${scene}"
        echo "              output/ablation_weights_2/S1_WG_${scene}"
        return 1
    fi

    local log="${LOGDIR}/diag_${scene}.log"

    echo ""
    echo "============================================"
    echo " DIAG E2B — scene=${scene}"
    echo " model=${model_dir}"
    echo " iter=${iter}"
    echo "============================================"

    python scripts/diagnose_crs_distribution.py \
        --source_path ${DATA_ROOT}/${scene} \
        -m ${model_dir} \
        --eval -r 8 --n_views 3 \
        --use_depth_prior --dav2_path ${DAV2_PATH} \
        --iteration ${iter} \
        2>&1 | tee ${log}

    echo "[DONE] ${scene} → ${log}"
}

case ${SCENE} in
    ALL|all)
        for s in ${ALL_SCENES}; do
            run_one "$s" "" "$ITER"
        done
        ;;
    *)
        run_one "${SCENE}" "${MODEL_DIR}" "${ITER}"
        ;;
esac

echo ""
echo "============================================"
echo " All diagnostics done. Logs in ${LOGDIR}/"
echo "============================================"
