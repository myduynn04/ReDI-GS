#!/bin/bash
# ============================================================
# [CRSGaussian Phase 11 Step 1] Diagnostic — covisibility reweight
# 2 configs × 1 scene (orchids) × 2 GPUs parallel.
#
# Backbone = Phase 8 FULL (LOCKED 21.335 dB). Test xem cov reweight
# có break ceiling không.
#
#   A0: Phase 8 FULL ref (no covisibility) — sanity reproduce
#   A1: + covisibility reweight (combined với CRS_pix, mode=min)
#
# P8 FULL ref (orchids): 16.761 dB
#
# Decision tree (Δ_A1 vs A0):
#   ≥ +0.20 → confirm thêm 2 scenes (leaves + horns), nếu hold → scale 8
#    +0.05..+0.20 → marginal, parallel test Step 2 (perceptual)
#   < +0.05 → reject Step 1, đi Step 2
# ============================================================

set -eo pipefail
LOG_DIR=${LOG_DIR:-logs/p11s1}
OUT_DIR_VAL=${OUT_DIR:-output/p11s1}
mkdir -p ${LOG_DIR} ${OUT_DIR_VAL}

SCENE=${SCENE:-orchids}

# Phase 8 FULL backbone — exact components (LOCKED 21.335)
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
    --use_r_visible
    --use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100
    --use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.5
    --use_sh_reliability --sh_stability_warmup 1000
    --sh_stability_ema_beta 0.95 --crs_w_s 0.33
)

# A0 (GPU 0) — pure Phase 8 FULL (no covisibility)
echo "[$(date +%H:%M:%S)] START A0_${SCENE} on GPU0"
CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --source_path data/nerf_llff_data/${SCENE} \
    -m ${OUT_DIR_VAL}/A0_${SCENE} \
    "${P8_FULL_BACKBONE[@]}" \
    > ${LOG_DIR}/A0_${SCENE}.log 2>&1 &
PID_A0=$!

# A1 (GPU 1) — Phase 11 cov reweight + combine với CRS_pix (min)
echo "[$(date +%H:%M:%S)] START A1_${SCENE} on GPU1"
CUDA_VISIBLE_DEVICES=1 python -u train.py \
    --source_path data/nerf_llff_data/${SCENE} \
    -m ${OUT_DIR_VAL}/A1_${SCENE} \
    "${P8_FULL_BACKBONE[@]}" \
    --use_coreliability_reweight \
    --coreliability_gamma 0.3 \
    --coreliability_combine_crs \
    --coreliability_combine_mode min \
    --coreliability_depth_consistency_thr 0.05 \
    > ${LOG_DIR}/A1_${SCENE}.log 2>&1 &
PID_A1=$!

wait $PID_A0 $PID_A1
echo "[$(date +%H:%M:%S)] BOTH DONE"

# Quick result check
echo ""
echo "=== Diagnostic results ==="
for CFG in A0 A1; do
    LOG="${LOG_DIR}/${CFG}_${SCENE}.log"
    if [ -f "$LOG" ]; then
        PSNR=$(grep "Best test PSNR:" "$LOG" | tail -1 | awk -F: '{print $2}' | tr -d ' ')
        echo "  ${CFG}_${SCENE}: PSNR=${PSNR}"
    else
        echo "  ${CFG}_${SCENE}: LOG MISSING"
    fi
done
echo ""
echo "Reference: P8_FULL orchids = 16.761 dB"
echo ""
echo "Decision rule:"
echo "  Δ_A1 vs A0 ≥ +0.20 → confirm leaves + horns → scale 8"
echo "  Δ_A1 vs A0 ∈ +0.05..+0.20 → marginal, parallel test Step 2"
echo "  Δ_A1 vs A0 < +0.05 → reject Step 1, đi Step 2"
