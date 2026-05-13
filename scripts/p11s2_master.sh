#!/bin/bash
# ============================================================
# DEPRECATED 2026-05-11: Phase 11 Step 2 (perceptual DINO same-view) REJECTED — Δ=−0.046 N=24
# Flags removed from arguments/__init__.py — script no longer runnable.
# Kept for paper reproducibility evidence. See CLAUDE.md Phase 11 results.
# ============================================================
# ============================================================
# [CRSGaussian Phase 11 Step 2] Same-view perceptual loss DINOv2 master ablation.
# 2 configs × 8 scenes = 16 runs.
#
# Backbone = Phase 8 FULL (LOCKED 21.335). Test xem perceptual loss
# DINOv2 (self-supervised features) có break ceiling không.
#
#   A0: Phase 8 FULL ref (no perceptual)
#   A1: + perceptual_dino (cosine, λ=0.05, freq=5, start=1500)
#
# Usage (2 GPU parallel):
#   GPU=0 SCENES_OVERRIDE="fern flower fortress horns" bash scripts/p11s2_master.sh
#   GPU=1 SCENES_OVERRIDE="leaves orchids room trex"  bash scripts/p11s2_master.sh
#
# Decision (Δ_A1 vs A0 avg):
#   ≥ +0.20 → 🎉 confirm Step 2, lock recipe
#   +0.05..+0.20 → 🟡 marginal, test combined với Step 1
#   < +0.05 → ❌ reject, đi Step 3
# ============================================================

set -eo pipefail
GPU=${GPU:-0}
LOG_DIR=${LOG_DIR:-logs/p11s2}
OUT_DIR_VAL=${OUT_DIR:-output/p11s2}
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
CONFIGS=${CONFIGS_OVERRIDE:-"A0 A1"}

for S in "${SCENES[@]}"; do
    for CFG_NAME in $CONFIGS; do
        case $CFG_NAME in
            A0)
                # Phase 8 FULL ref (no perceptual).
                run_one A0 $S "${P8_FULL_BACKBONE[@]}"
                ;;
            A1)
                # Phase 11 Step 2 — same-view DINOv2 perceptual.
                run_one A1 $S "${P8_FULL_BACKBONE[@]}" \
                    --use_perceptual_dino \
                    --lambda_perceptual_dino 0.05 \
                    --perceptual_dino_start_iter 1500 \
                    --perceptual_dino_freq 5 \
                    --perceptual_dino_mode cosine
                ;;
            *)
                echo "Unknown config: $CFG_NAME (valid: A0|A1)"
                exit 1
                ;;
        esac
    done
done

echo "[GPU${GPU}] Phase 11 Step 2 complete (LOG_DIR=${LOG_DIR}). Run analyzer."
