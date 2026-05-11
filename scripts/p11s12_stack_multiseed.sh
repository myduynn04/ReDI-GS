#!/bin/bash
# ============================================================
# [CRSGaussian Phase 11 Step 1+2 Stack] Multi-seed ablation — 3 configs same batch.
# 3 seeds × 8 scenes × 3 configs (A0/A1/A2) = 72 runs.
#
# Mục đích:
#   Re-run cùng batch để fair compare Step 1 alone vs Step 1+2 stack.
#   Step 1 (Δ=−0.028) và Step 2 (Δ=−0.046) chạy ở 2 batches khác nhau
#   → A0 baseline khác 0.040 dB (cuDNN noise) → khó kết luận synergy.
#   Cùng batch cho 3 đường so sánh paired clean.
#
#   A0 = Phase 8 FULL baseline (NO extras)
#   A1 = + Step 1 covisibility reweight (CRS combine, gamma=0.3)
#   A2 = + Step 1 AND Step 2 (perceptual DINO cosine, λ=0.05)
#
# Hypothesis: stack có thể synergize dù mỗi cái alone ≈ noise (Phase 9
# precedent: D+R+S synergize even though leave-one-out hurts).
#
# Cost:
#   ~5-7 phút/run × 72 runs / 2 GPU = ~3 giờ wall-clock
#   GPU 0 = 4 scenes × 3 configs × 3 seeds = 36 runs serial
#   GPU 1 = 4 scenes × 3 configs × 3 seeds = 36 runs serial
#
# Usage (2 GPU parallel):
#   GPU=0 SCENES_OVERRIDE="fern flower fortress horns" \
#       bash scripts/p11s12_stack_multiseed.sh > logs/p11s12_ms/gpu0.log 2>&1 &
#   GPU=1 SCENES_OVERRIDE="leaves orchids room trex" \
#       bash scripts/p11s12_stack_multiseed.sh > logs/p11s12_ms/gpu1.log 2>&1 &
#   wait
#
# Output:
#   logs/p11s12_ms/{A0,A1,A2}_seed{42,137,9999}_<scene>.log
# ============================================================

set -eo pipefail
GPU=${GPU:-0}
LOG_DIR=${LOG_DIR:-logs/p11s12_ms}
OUT_DIR_VAL=${OUT_DIR:-output/p11s12_ms}
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
CONFIGS=${CONFIGS_OVERRIDE:-"A0 A1 A2"}

for SEED in "${SEEDS[@]}"; do
    for S in "${SCENES[@]}"; do
        for CFG_NAME in $CONFIGS; do
            case $CFG_NAME in
                A0)
                    # Phase 8 FULL baseline — NO extras.
                    run_one A0 $S $SEED "${P8_FULL_BACKBONE[@]}"
                    ;;
                A1)
                    # Step 1 only — covisibility reweight với CRS combine.
                    run_one A1 $S $SEED "${P8_FULL_BACKBONE[@]}" \
                        --use_coreliability_reweight \
                        --coreliability_gamma 0.3 \
                        --coreliability_combine_crs \
                        --coreliability_combine_mode min \
                        --coreliability_depth_consistency_thr 0.05
                    ;;
                A2)
                    # Stack: Step 1 + Step 2 cùng bật.
                    run_one A2 $S $SEED "${P8_FULL_BACKBONE[@]}" \
                        --use_coreliability_reweight \
                        --coreliability_gamma 0.3 \
                        --coreliability_combine_crs \
                        --coreliability_combine_mode min \
                        --coreliability_depth_consistency_thr 0.05 \
                        --use_perceptual_dino \
                        --lambda_perceptual_dino 0.05 \
                        --perceptual_dino_start_iter 1500 \
                        --perceptual_dino_freq 5 \
                        --perceptual_dino_mode cosine
                    ;;
                *)
                    echo "Unknown config: $CFG_NAME (valid: A0|A1|A2)"
                    exit 1
                    ;;
            esac
        done
    done
done

echo "[GPU${GPU}] Phase 11 Step 1+Stack multi-seed complete (LOG_DIR=${LOG_DIR})."
