#!/bin/bash
# ============================================================
# DEPRECATED 2026-05-11: Phase 11 Step 5 (TV depth edge-preserving) REJECTED — Δ=−0.026 N=24
# Flags removed from arguments/__init__.py — script no longer runnable.
# Kept for paper reproducibility evidence. See CLAUDE.md Phase 11 results.
# ============================================================
# ============================================================
# [CRSGaussian Phase 11 Step 5] Multi-seed ablation — TV depth edge-preserving.
# 3 seeds × 8 scenes × 2 configs (A0/A1) = 48 runs.
#
# Mục đích:
#   1. Test anti-overfit regularizer trên depth field (smoothness +
#      edge-preserving) — class khác hoàn toàn Step 1/2/3/4 (loss
#      cross-modal/cross-view).
#   2. Compare Δ_paired (A1−A0 mỗi seed × scene) avg over 24 samples
#      → confirm TV depth signal có robust hay chỉ noise
#   3. Verdict ≥ +0.10 → winner candidate cho stack với Step 1/Step 4
#
# A0 = Phase 8 FULL ref (no TV)
# A1 = Phase 8 FULL + TV depth (edge-preserving, λ=0.005, α=10.0,
#       start=1000)
#
# Cost:
#   ~5-6 phút/run × 48 runs / 2 GPU = ~2 giờ wall-clock
#   (TV chỉ tensor diff ~1-2ms/iter — overhead nhỏ nhất trong Phase 11)
#   GPU 0 = 4 scenes × 2 configs × 3 seeds = 24 runs serial
#   GPU 1 = 4 scenes × 2 configs × 3 seeds = 24 runs serial
#
# Usage (2 GPU parallel):
#   GPU=0 SCENES_OVERRIDE="fern flower fortress horns" \
#       bash scripts/p11s5_multiseed.sh > logs/p11s5_ms/gpu0.log 2>&1 &
#   GPU=1 SCENES_OVERRIDE="leaves orchids room trex" \
#       bash scripts/p11s5_multiseed.sh > logs/p11s5_ms/gpu1.log 2>&1 &
#   wait
#
# Output:
#   logs/p11s5_ms/A0_seed{42,137,9999}_<scene>.log
#   logs/p11s5_ms/A1_seed{42,137,9999}_<scene>.log
# ============================================================

set -eo pipefail
GPU=${GPU:-0}
LOG_DIR=${LOG_DIR:-logs/p11s5_ms}
OUT_DIR_VAL=${OUT_DIR:-output/p11s5_ms}
mkdir -p ${LOG_DIR} ${OUT_DIR_VAL}

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

run_one() {
    local CFG=$1; local SC=$2; local SEED=$3; shift 3
    local EXTRA=("$@")
    local OUTDIR=${OUT_DIR_VAL}/${CFG}_seed${SEED}_${SC}
    local LOG=${LOG_DIR}/${CFG}_seed${SEED}_${SC}.log

    echo "[$(date +%H:%M:%S)] START ${CFG}_seed${SEED}_${SC} (GPU${GPU})"
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --source_path data/nerf_llff_data/${SC} \
        -m ${OUTDIR} \
        --seed ${SEED} \
        "${EXTRA[@]}" \
        2>&1 | tee ${LOG}
    echo "[$(date +%H:%M:%S)] DONE  ${CFG}_seed${SEED}_${SC}"
}

SCENES=(${SCENES_OVERRIDE:-fern flower fortress horns leaves orchids room trex})
SEEDS=(${SEEDS_OVERRIDE:-42 137 9999})
CONFIGS=${CONFIGS_OVERRIDE:-"A0 A1"}

for SEED in "${SEEDS[@]}"; do
    for S in "${SCENES[@]}"; do
        for CFG_NAME in $CONFIGS; do
            case $CFG_NAME in
                A0)
                    # Phase 8 FULL ref — no TV depth.
                    run_one A0 $S $SEED "${P8_FULL_BACKBONE[@]}"
                    ;;
                A1)
                    # Phase 11 Step 5 — edge-preserving TV depth regularizer.
                    run_one A1 $S $SEED "${P8_FULL_BACKBONE[@]}" \
                        --use_tv_depth \
                        --lambda_tv_depth 0.005 \
                        --tv_depth_mode edge_preserving \
                        --tv_depth_alpha 10.0 \
                        --tv_depth_start_iter 1000
                    ;;
                *)
                    echo "Unknown config: $CFG_NAME (valid: A0|A1)"
                    exit 1
                    ;;
            esac
        done
    done
done

echo "[GPU${GPU}] Phase 11 Step 5 multi-seed complete (LOG_DIR=${LOG_DIR})."
