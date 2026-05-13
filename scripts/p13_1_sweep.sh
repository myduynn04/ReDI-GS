#!/bin/bash
# ============================================================
# [CRSGaussian Phase 13.1] LFCF intensity sweep — find optimal scaler+interval.
# 3 variants × 8 scenes × seed 42 = 24 runs.
#
# Baseline reference: A3 (scaler=1.5, interval=2, AbsGS=ON) — đã có data Round 1+2.
# Sweep test LFCF intensity tăng dần với AbsGS đồng thời bật:
#
#   M = medium   scaler_max=1.8, interval_times=2  (enlarge nhẹ hơn baseline)
#   H = high     scaler_max=2.0, interval_times=1  (enlarge mạnh + LFCF mỗi 100 iter)
#   X = extra    scaler_max=2.5, interval_times=1  (max enlarge + max frequency)
#
# Mục đích: A3 hiện train 34.21 (overfit gap +0.25 vs A0). Sweep test xem
# LFCF brake mạnh hơn có giảm train PSNR mà giữ test PSNR không.
#
# Cost: ~6-8 phút/run × 24 runs / 2 GPU = ~2h wall-clock
#
# Usage (2 GPU parallel):
#   GPU=0 SCENES_OVERRIDE="fern flower fortress horns" \
#       bash scripts/p13_1_sweep.sh > logs/p13_1_sweep/gpu0.log 2>&1 &
#   GPU=1 SCENES_OVERRIDE="leaves orchids room trex" \
#       bash scripts/p13_1_sweep.sh > logs/p13_1_sweep/gpu1.log 2>&1 &
#   wait
#
# Output:
#   logs/p13_1_sweep/{M,H,X}_seed42_<scene>.log
# ============================================================

set -eo pipefail
GPU=${GPU:-0}
LOG_DIR=${LOG_DIR:-logs/p13_1_sweep}
OUT_DIR_VAL=${OUT_DIR:-output/p13_1_sweep}
mkdir -p ${LOG_DIR} ${OUT_DIR_VAL}

# P8_FULL_BACKBONE — copy EXACT từ scripts/p11s1_multiseed.sh (giữ 10k iter)
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

# LFCF flags common to all sweep variants (TaT defaults)
LFCF_COMMON=(
    --use_lfcf
    --lfcf_init_scaling_min 1.0
    --lfcf_last_scaling_max 1.0
    --lfcf_pow 1.0
    --lfcf_splitting_ub 1.0
    --lfcf_tolerance 1e-5
    --lfcf_diffscale True
    --absdensify   # All sweep variants include AbsGS (matches A3 combo)
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
CONFIGS=${CONFIGS_OVERRIDE:-"M H X"}

for SEED in "${SEEDS[@]}"; do
    for S in "${SCENES[@]}"; do
        for CFG_NAME in $CONFIGS; do
            case $CFG_NAME in
                M)
                    # Medium: scaler 1.8, interval 2 — moderate stronger enlarge
                    run_one M $S $SEED "${P8_FULL_BACKBONE[@]}" \
                        "${LFCF_COMMON[@]}" \
                        --lfcf_init_scaling_max 1.8 \
                        --lfcf_interval_times 2
                    ;;
                H)
                    # High: scaler 2.0, interval 1 — strong enlarge + LFCF mỗi 100 iter
                    run_one H $S $SEED "${P8_FULL_BACKBONE[@]}" \
                        "${LFCF_COMMON[@]}" \
                        --lfcf_init_scaling_max 2.0 \
                        --lfcf_interval_times 1
                    ;;
                X)
                    # Extra: scaler 2.5, interval 1 — max aggression
                    run_one X $S $SEED "${P8_FULL_BACKBONE[@]}" \
                        "${LFCF_COMMON[@]}" \
                        --lfcf_init_scaling_max 2.5 \
                        --lfcf_interval_times 1
                    ;;
                *)
                    echo "Unknown config: $CFG_NAME (valid: M|H|X)"
                    exit 1
                    ;;
            esac
        done
    done
done

echo "[GPU${GPU}] Phase 13.1 LFCF intensity sweep complete (LOG_DIR=${LOG_DIR})."
