#!/bin/bash
# ============================================================
# DEPRECATED 2026-05-12: Phase 12 CRS-pull REJECTED — Δ_A1=−0.027, Δ_A2=−0.033, Δ_A3=−0.096 N=8
# Flags removed from arguments/__init__.py — script no longer runnable.
# Module utils/loss/crs_pull.py deleted. Kept for paper reproducibility evidence.
# See logs/p12_crs_pull/SUMMARY_round1.txt for full numbers.
# ============================================================
# ============================================================
# [CRSGaussian Phase 12] CRS-pull multi-seed ablation — 4 configs.
# Adaptive seeding: Round 1 = seed 42 only (8 scenes × 4 configs = 32 runs).
# Round 2 (if winner) = seeds 137 + 9999 với best config.
#
# Configs:
#   A0 = Phase 8 FULL baseline (no CRS-pull, has CRS pruning)
#   A1 = + CRS-pull (full: pull + scale + opacity) — main candidate
#   A2 = + CRS-pull − Phase 4 CRS pruning (test "CRS-pull replace prune")
#        Dùng --disable_crs_pruning (negation flag, vì --use_crs_pruning False
#        không work với action="store_true" argparse pattern).
#   A3 = + CRS-pull pull-only (lambda_scale=0, lambda_opacity=0) — isolate pull
#
# Cost (Round 1, seed 42):
#   ~6-8 phút/run × 32 runs / 2 GPU = ~3.5 giờ wall-clock
#   GPU 0 = 4 scenes × 4 configs = 16 runs serial
#   GPU 1 = 4 scenes × 4 configs = 16 runs serial
#
# Usage Round 1 (2 GPU parallel):
#   GPU=0 SCENES_OVERRIDE="fern flower fortress horns" \
#       bash scripts/p12_crs_pull_multiseed.sh > logs/p12_crs_pull/gpu0_seed42.log 2>&1 &
#   GPU=1 SCENES_OVERRIDE="leaves orchids room trex" \
#       bash scripts/p12_crs_pull_multiseed.sh > logs/p12_crs_pull/gpu1_seed42.log 2>&1 &
#   wait
#
# Round 2 (if winner found):
#   GPU=0 SEEDS_OVERRIDE="137" CONFIGS_OVERRIDE="<best>" SCENES_OVERRIDE="..." \
#       bash scripts/p12_crs_pull_multiseed.sh > logs/.../gpu0_seed137.log 2>&1 &
#   GPU=1 SEEDS_OVERRIDE="9999" CONFIGS_OVERRIDE="<best>" SCENES_OVERRIDE="..." \
#       bash scripts/p12_crs_pull_multiseed.sh > logs/.../gpu1_seed9999.log 2>&1 &
#   wait
#
# Output:
#   logs/p12_crs_pull/{A0,A1,A2,A3}_seed{42,...}_<scene>.log
# ============================================================

set -eo pipefail
GPU=${GPU:-0}
LOG_DIR=${LOG_DIR:-logs/p12_crs_pull}
OUT_DIR_VAL=${OUT_DIR:-output/p12_crs_pull}
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
SEEDS=(${SEEDS_OVERRIDE:-42})
CONFIGS=${CONFIGS_OVERRIDE:-"A0 A1 A2 A3"}

for SEED in "${SEEDS[@]}"; do
    for S in "${SCENES[@]}"; do
        for CFG_NAME in $CONFIGS; do
            case $CFG_NAME in
                A0)
                    # Phase 8 FULL baseline.
                    run_one A0 $S $SEED "${P8_FULL_BACKBONE[@]}"
                    ;;
                A1)
                    # Full CRS-pull: pull + smart scale + opacity fade.
                    run_one A1 $S $SEED "${P8_FULL_BACKBONE[@]}" \
                        --use_crs_pull \
                        --lambda_crs_pull 0.005 \
                        --lambda_crs_pull_scale 0.001 \
                        --lambda_crs_pull_opacity 0.001
                    ;;
                A2)
                    # CRS-pull AS REPLACEMENT for Phase 4 prune.
                    # --disable_crs_pruning bypass Phase 4 CRS-prune block.
                    run_one A2 $S $SEED "${P8_FULL_BACKBONE[@]}" \
                        --disable_crs_pruning \
                        --use_crs_pull \
                        --lambda_crs_pull 0.005 \
                        --lambda_crs_pull_scale 0.001 \
                        --lambda_crs_pull_opacity 0.001
                    ;;
                A3)
                    # Pull-only ablation (scale & opacity weights = 0).
                    # Isolate pull contribution.
                    run_one A3 $S $SEED "${P8_FULL_BACKBONE[@]}" \
                        --use_crs_pull \
                        --lambda_crs_pull 0.005 \
                        --lambda_crs_pull_scale 0.0 \
                        --lambda_crs_pull_opacity 0.0
                    ;;
                *)
                    echo "Unknown config: $CFG_NAME (valid: A0|A1|A2|A3)"
                    exit 1
                    ;;
            esac
        done
    done
done

echo "[GPU${GPU}] Phase 12 CRS-pull multi-seed complete (LOG_DIR=${LOG_DIR})."
