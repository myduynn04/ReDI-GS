#!/bin/bash
# ============================================================
# [CRSGaussian Phase 13] LFCF + AbsGS multi-seed ablation — 5 configs.
# Adaptive seeding: Round 1 = seed 42 only (8 scenes × 5 configs = 40 runs).
# Round 2 (if winner) = seeds 137 + 9999 với best config.
#
# Configs (per design doc Section 8.1):
#   A0 = Phase 8 FULL baseline (no LFCF, no AbsGS) — reference 21.16
#   A1 = + LFCF (TaT defaults: scaler=1.5, interval=2, diffscale=ON, AbsGS OFF)
#        ⭐ core LFCF test
#   A2 = + LFCF (diffscale=OFF) — isolate diffscale contribution
#   A3 = + LFCF (diffscale=ON, AbsGS=ON) — combo synergy test
#   A4 = AbsGS only (no LFCF) — standalone attribution
#
# Attribution matrix (5 deltas):
#   Δ_A1            = LFCF full effect
#   Δ_A1 − Δ_A2     = Diffscale alone contribution
#   Δ_A3 − Δ_A1     = AbsGS bonus on LFCF
#   Δ_A4            = AbsGS standalone (cheap dormant code activation)
#   Δ_A3 − (Δ_A1 + Δ_A4) = LFCF × AbsGS synergy
#
# Cost (Round 1, seed 42):
#   ~6-8 phút/run × 40 runs / 2 GPU = ~4.5h wall-clock
#   GPU 0 = 4 scenes × 5 configs = 20 runs serial
#   GPU 1 = 4 scenes × 5 configs = 20 runs serial
#
# Usage Round 1 (2 GPU parallel):
#   GPU=0 SCENES_OVERRIDE="fern flower fortress horns" \
#       bash scripts/p13_lfcf_multiseed.sh > logs/p13_lfcf/gpu0_seed42.log 2>&1 &
#   GPU=1 SCENES_OVERRIDE="leaves orchids room trex" \
#       bash scripts/p13_lfcf_multiseed.sh > logs/p13_lfcf/gpu1_seed42.log 2>&1 &
#   wait
#
# Round 2 (if winner found):
#   GPU=0 SEEDS_OVERRIDE="137" CONFIGS_OVERRIDE="<best> A0" \
#       SCENES_OVERRIDE="fern flower fortress horns" \
#       bash scripts/p13_lfcf_multiseed.sh > logs/p13_lfcf/gpu0_seed137.log 2>&1 &
#   GPU=1 SEEDS_OVERRIDE="9999" CONFIGS_OVERRIDE="<best> A0" \
#       SCENES_OVERRIDE="leaves orchids room trex" \
#       bash scripts/p13_lfcf_multiseed.sh > logs/p13_lfcf/gpu1_seed9999.log 2>&1 &
#   wait
#
# Output:
#   logs/p13_lfcf/{A0,A1,A2,A3,A4}_seed{42,137,9999}_<scene>.log
# ============================================================

set -eo pipefail
GPU=${GPU:-0}
LOG_DIR=${LOG_DIR:-logs/p13_lfcf}
OUT_DIR_VAL=${OUT_DIR:-output/p13_lfcf}
mkdir -p ${LOG_DIR} ${OUT_DIR_VAL}

# ⭐ P8_FULL_BACKBONE: COPY EXACT từ scripts/p11s1_multiseed.sh (line 36-55)
# KHÔNG infer từ memory hoặc CLAUDE.md — code regression `0511edd` đã làm drift.
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

# Common LFCF flags (TaT defaults — KHÔNG self-tune, per design doc Section 8.4)
LFCF_COMMON=(
    --use_lfcf
    --lfcf_init_scaling_max 1.5
    --lfcf_init_scaling_min 1.0
    --lfcf_last_scaling_max 1.0
    --lfcf_pow 1.0
    --lfcf_splitting_ub 1.0
    --lfcf_interval_times 2
    --lfcf_tolerance 1e-5
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
CONFIGS=${CONFIGS_OVERRIDE:-"A0 A1 A2 A3 A4"}

for SEED in "${SEEDS[@]}"; do
    for S in "${SCENES[@]}"; do
        for CFG_NAME in $CONFIGS; do
            case $CFG_NAME in
                A0)
                    # Phase 8 FULL baseline — no LFCF, no AbsGS
                    run_one A0 $S $SEED "${P8_FULL_BACKBONE[@]}"
                    ;;
                A1)
                    # LFCF (TaT defaults, diffscale ON, AbsGS OFF) — core test
                    run_one A1 $S $SEED "${P8_FULL_BACKBONE[@]}" \
                        "${LFCF_COMMON[@]}" \
                        --lfcf_diffscale True
                    ;;
                A2)
                    # LFCF (diffscale OFF) — isolate diffscale contribution
                    run_one A2 $S $SEED "${P8_FULL_BACKBONE[@]}" \
                        "${LFCF_COMMON[@]}" \
                        --lfcf_diffscale False
                    ;;
                A3)
                    # LFCF + AbsGS combo — synergy test
                    run_one A3 $S $SEED "${P8_FULL_BACKBONE[@]}" \
                        "${LFCF_COMMON[@]}" \
                        --lfcf_diffscale True \
                        --absdensify
                    ;;
                A4)
                    # AbsGS standalone — no LFCF
                    run_one A4 $S $SEED "${P8_FULL_BACKBONE[@]}" \
                        --absdensify
                    ;;
                *)
                    echo "Unknown config: $CFG_NAME (valid: A0|A1|A2|A3|A4)"
                    exit 1
                    ;;
            esac
        done
    done
done

echo "[GPU${GPU}] Phase 13 LFCF multi-seed complete (LOG_DIR=${LOG_DIR})."
