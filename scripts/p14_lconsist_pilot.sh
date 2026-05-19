#!/bin/bash
# ============================================================
# [CRSGaussian Phase 14] L_consist 2×2 factorial pilot.
#
# 2×2 {D_cycle ON/OFF} × {L_consist OFF/ON}:
#   A = A3 (D_cycle ON, Lc OFF)  → REUSE logs/p13_lfcf/A3_seed42_* (KHÔNG run)
#   B = A3 + Lc                  → +--use_lconsist
#   C = A3 − D_cycle             → BỎ --use_d_cycle (→ CRS dùng D_DAV2; 0 code)
#   D = A3 − D_cycle + Lc        → bỏ --use_d_cycle +--use_lconsist
#
# Đọc: A−C=biên D_cycle/A3 ; B−A=Lc thêm ; D vs B=Lc thay D_cycle? ;
#      B≈A=Lc bão hoà. (D_cycle-off justified: decisions_log:948
#      "D_cycle flips negative on strong backbone", chưa A/B lại trên A3.)
#
# Single-seed 42, full-8 (feedback_full_8scene cho verdict). 3 cell mới
# × 8 = 24 run. 2-GPU split 4+4 (feedback_use_both_gpus).
#
# P8_FULL_BACKBONE + A3_FLAGS = EXACT copy scripts/p13_lfcf_multiseed.sh
# (CFG A3 = P8 + LFCF_COMMON + diffscale True + absdensify). KHÔNG infer.
#
# Usage (2 GPU parallel):
#   GPU=0 SCENES_OVERRIDE="fern flower fortress horns" \
#       bash scripts/p14_lconsist_pilot.sh > logs/p14_lconsist/gpu0.log 2>&1 &
#   GPU=1 SCENES_OVERRIDE="leaves orchids room trex" \
#       bash scripts/p14_lconsist_pilot.sh > logs/p14_lconsist/gpu1.log 2>&1 &
#   wait
#
# Pre-pilot verify (CHẠY TRƯỚC, rẻ):
#   # (a) flag-OFF = A3 byte-identical (code mới, KHÔNG --use_lconsist):
#   GPU=0 SCENES_OVERRIDE="orchids" CELLS_OVERRIDE="VOFF" \
#       bash scripts/p14_lconsist_pilot.sh
#     → so N vs logs/p13_lfcf/A3_seed42_orchids (reldiff <5% = OK)
#   # (b) flag-ON 1-scene smoke (KHÔNG crash, loss finite):
#   GPU=0 SCENES_OVERRIDE="orchids" CELLS_OVERRIDE="B" \
#       bash scripts/p14_lconsist_pilot.sh
#
# Output: logs/p14_lconsist/{B,C,D,VOFF}_seed42_<scene>.log
# ============================================================
set -eo pipefail
GPU=${GPU:-0}
LOG_DIR=${LOG_DIR:-logs/p14_lconsist}
OUT_DIR_VAL=${OUT_DIR:-output/p14_lconsist}
mkdir -p ${LOG_DIR} ${OUT_DIR_VAL}

# ── P8_FULL_BACKBONE: EXACT copy p13_lfcf_multiseed.sh:55-74 ──
# CHIA 2 phần: phần chung + dòng D_cycle TÁCH RIÊNG (cell C/D bỏ dòng này
# → use_d_cycle=False → crs_module:565 fallback D_DAV2, 0 code).
P8_CORE=(
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
    --use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.5
    --use_sh_reliability --sh_stability_warmup 1000
    --sh_stability_ema_beta 0.95 --crs_w_s 0.33
)
DCYCLE_FLAGS=(
    --use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0
    --d_cycle_update_freq 100
)
A3_FLAGS=(
    --use_lfcf
    --lfcf_init_scaling_max 1.5
    --lfcf_init_scaling_min 1.0
    --lfcf_last_scaling_max 1.0
    --lfcf_pow 1.0
    --lfcf_splitting_ub 1.0
    --lfcf_interval_times 2
    --lfcf_tolerance 1e-5
    --lfcf_diffscale True
    --absdensify
)
LCONSIST_FLAGS=(--use_lconsist)   # defaults: start 6700, trans 0.05, λ1, sm.05

run_one() {
    local CELL=$1; local SC=$2; local SEED=$3
    local DC=("${DCYCLE_FLAGS[@]}"); local LC=()
    case $CELL in
        B)    LC=("${LCONSIST_FLAGS[@]}") ;;                       # Dcyc ON  + Lc
        C)    DC=() ;;                                              # Dcyc OFF
        D)    DC=(); LC=("${LCONSIST_FLAGS[@]}") ;;                 # Dcyc OFF + Lc
        VOFF) : ;;                                                  # = A3 (verify)
        *) echo "Unknown CELL $CELL (B|C|D|VOFF)"; exit 1 ;;
    esac
    local OUTDIR=${OUT_DIR_VAL}/${CELL}_seed${SEED}_${SC}
    local LOG=${LOG_DIR}/${CELL}_seed${SEED}_${SC}.log
    echo "[$(date +%H:%M:%S)] START ${CELL}_seed${SEED}_${SC} (GPU${GPU})"
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --source_path data/nerf_llff_data/${SC} \
        -m ${OUTDIR} \
        --seed ${SEED} \
        "${P8_CORE[@]}" "${DC[@]}" "${A3_FLAGS[@]}" "${LC[@]}" \
        2>&1 | tee ${LOG}
    echo "[$(date +%H:%M:%S)] DONE  ${CELL}_seed${SEED}_${SC}"
}

SCENES=(${SCENES_OVERRIDE:-fern flower fortress horns leaves orchids room trex})
SEEDS=(${SEEDS_OVERRIDE:-42})
CELLS=${CELLS_OVERRIDE:-"B C D"}     # A reuse logs/p13_lfcf — KHÔNG run

echo "=== Phase 14 L_consist pilot (GPU${GPU}) CELLS=${CELLS} ==="
echo "  Scenes: ${SCENES[@]}  Seeds: ${SEEDS[@]}"
for SEED in "${SEEDS[@]}"; do
    for S in "${SCENES[@]}"; do
        for CL in $CELLS; do
            run_one $CL $S $SEED
        done
    done
done
echo "[GPU${GPU}] Phase 14 pilot complete."
