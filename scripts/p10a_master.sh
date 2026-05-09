#!/bin/bash
# ============================================================
# [CRSGaussian Phase 10A] Dense init via DUSt3R — 16-run ablation.
#
# Backbone = Phase 8 FULL (LOCKED 21.335 dB):
#   D1-O999 + R_visible + D_cycle + S_stability + CRS-mod SH freeze
# + Phase 10A dense init từ cache/dust3r_init/<scene>.npz
#
# 2 modes × 8 scenes = 16 runs:
#   AUGMENT: COLMAP + DUSt3R (KDTree dedupe trong dedupe_radius)
#   REPLACE: chỉ DUSt3R points
#
# Đối chứng: logs/p8/FULL_<scene>.log (KHÔNG re-run Phase 8 FULL).
#
# Hypotheses:
#   H1: AUGMENT > P8_FULL by ≥ +0.30 dB → dense init breaks ceiling
#   H2: REPLACE ≈ AUGMENT → COLMAP redundant với DUSt3R
#   H3: AUGMENT > REPLACE → COLMAP keypoints có precision unique value
#
# Cost: ~60-90 phút wall-clock 2 GPU sequential (8000 iter × 16 runs).
# Pre-req: scripts/precompute_dust3r.py đã tạo cache/dust3r_init/<scene>.npz.
# ============================================================

set -eo pipefail
GPU=${GPU:-0}
OUT_BASE=${OUT_DIR:-logs/p10a}
LOG_DIR=${OUT_BASE}
OUT_DIR_VAL=${OUT_BASE/logs/output}
mkdir -p ${LOG_DIR} ${OUT_DIR_VAL}

# Phase 8 FULL backbone — exact components LOCKED ở 21.335 dB
P8_FULL_BACKBONE=(
    --eval -r 8 --n_views 3 --random_background
    --iterations 10000
    --densify_until_iter 5000 --densify_grad_threshold 0.0005
    --gaussiansN 1
    --use_depth_prior --dav2_path ../Depth-Anything-V2
    --use_crs_pruning --informed_crs_init
    --crs_init_use_view False
    --crs_init_w_reproj 0.4 --crs_init_w_depth 0.6 --crs_init_w_view 0.0
    --crs_ema_decay 0.3 --crs_update_interval 100
    --use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2
    --use_opacity_decay --opacity_decay_factor 0.999
    --sample_pseudo_interval 1 --start_sample_pseudo 500
    --test_iterations 10000
    # Phase 8 components
    --use_r_visible
    --use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100
    --use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.5
    --use_sh_reliability --sh_stability_warmup 1000
    --sh_stability_ema_beta 0.95 --crs_w_s 0.33
    # Phase 10A dense init master switch
    --use_dense_init
    --dust3r_cache_dir cache/dust3r_init
    --dense_init_conf_threshold 1.5
    --dense_init_max_points 50000
    --dense_init_dedupe_radius 0.01
)

run_one() {
    local CFG=$1; local SC=$2; shift 2
    local EXTRA=("$@")
    local OUTDIR=${OUT_DIR_VAL}/${CFG}_${SC}
    local LOG=${LOG_DIR}/${CFG}_${SC}.log

    echo "[$(date +%H:%M:%S)] START ${CFG}_${SC} (GPU${GPU})"
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --source_path data/nerf_llff_data/${SC} \
        -m ${OUTDIR} \
        "${EXTRA[@]}" \
        2>&1 | tee ${LOG}
    echo "[$(date +%H:%M:%S)] DONE  ${CFG}_${SC}"
}

SCENES=(${SCENES_OVERRIDE:-fern flower fortress horns leaves orchids room trex})
CONFIGS=${CONFIGS_OVERRIDE:-"AUGMENT REPLACE"}

for S in "${SCENES[@]}"; do
    for CFG_NAME in $CONFIGS; do
        case $CFG_NAME in
            AUGMENT)
                # COLMAP + DUSt3R, KDTree dedupe trong dedupe_radius * scene_extent.
                run_one AUGMENT $S "${P8_FULL_BACKBONE[@]}" \
                    --dense_init_mode augment
                ;;
            REPLACE)
                # Chỉ DUSt3R, bỏ COLMAP — test xem COLMAP keypoints còn cần không.
                run_one REPLACE $S "${P8_FULL_BACKBONE[@]}" \
                    --dense_init_mode replace
                ;;
            *)
                echo "Unknown config: $CFG_NAME"
                echo "Valid: AUGMENT|REPLACE"
                exit 1
                ;;
        esac
    done
done

echo "[GPU${GPU}] Phase 10A ablation complete (LOG_DIR=${LOG_DIR})."
echo "Run: python scripts/p10a_analyze.py"
