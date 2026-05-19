#!/bin/bash
# ============================================================
# [CRSGaussian Phase 15] Anisotropy shape-reg pilot (trên A3-clean).
#
# Cells (A3 baseline = REUSE logs/p13_lfcf/A3_seed42_* — KHÔNG run):
#   Ablunt : --use_shape_reg blunt λ=1e-2  → CONTROL falsify Q4
#            (Q4: 77% high-aniso=legitimate-flat → blunt predicted hại;
#             diagnostic project misfire nhiều → kiểm empirical)
#   Bexc3  : smax_excess λ=1e-3  → form Q4-data CHỈ vào (né 77%-flat)
#   Bexc2  : smax_excess λ=1e-2  → cùng form, λ mạnh hơn (λ no natural
#            scale → sweep 2 mức cho B = ứng viên thật)
#   VOFF   : KHÔNG --use_shape_reg = A3 → verify flag-OFF byte-identical
#
# Full-8 seed42 (feedback_full_8scene cho verdict). 3 cell mới × 8 = 24
# run. 2-GPU split 4+4 (feedback_use_both_gpus). P8_CORE+A3_FLAGS EXACT
# copy p13_lfcf_multiseed.sh (CFG A3) — KHÔNG infer.
#
# Pre-pilot verify (CHẠY TRƯỚC, rẻ):
#   GPU=0 SCENES_OVERRIDE=orchids CELLS_OVERRIDE=VOFF bash scripts/p15_shape_pilot.sh
#     → so N vs logs/p13_lfcf/A3_seed42_orchids (reldiff<5% = code Phase-15
#       OFF KHÔNG phá A3) ; + 1 cell ON smoke không crash + loss finite.
#
# Usage (2 GPU parallel):
#   GPU=0 SCENES_OVERRIDE="fern flower fortress horns" \
#       bash scripts/p15_shape_pilot.sh > logs/p15_shape/gpu0.log 2>&1 &
#   GPU=1 SCENES_OVERRIDE="leaves orchids room trex" \
#       bash scripts/p15_shape_pilot.sh > logs/p15_shape/gpu1.log 2>&1 &
#   wait
# Output: logs/p15_shape/{Ablunt,Bexc3,Bexc2,VOFF}_seed42_<scene>.log
# ============================================================
set -eo pipefail
GPU=${GPU:-0}
LOG_DIR=${LOG_DIR:-logs/p15_shape}
OUT_DIR_VAL=${OUT_DIR:-output/p15_shape}
mkdir -p ${LOG_DIR} ${OUT_DIR_VAL}

# P8_CORE + A3_FLAGS = EXACT copy p13_lfcf_multiseed.sh:55-86 (CFG A3).
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
    --use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100
    --use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.5
    --use_sh_reliability --sh_stability_warmup 1000
    --sh_stability_ema_beta 0.95 --crs_w_s 0.33
)
A3_FLAGS=(
    --use_lfcf
    --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0
    --lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0
    --lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True
    --absdensify
)

run_one() {
    local CELL=$1; local SC=$2; local SEED=$3
    local SR=()
    case $CELL in
        Ablunt) SR=(--use_shape_reg --shape_reg_mode blunt --shape_reg_lambda 1e-2) ;;
        Bexc3)  SR=(--use_shape_reg --shape_reg_mode smax_excess --shape_reg_lambda 1e-3) ;;
        Bexc2)  SR=(--use_shape_reg --shape_reg_mode smax_excess --shape_reg_lambda 1e-2) ;;
        VOFF)   SR=() ;;
        *) echo "Unknown CELL $CELL (Ablunt|Bexc3|Bexc2|VOFF)"; exit 1 ;;
    esac
    local OUTDIR=${OUT_DIR_VAL}/${CELL}_seed${SEED}_${SC}
    local LOG=${LOG_DIR}/${CELL}_seed${SEED}_${SC}.log
    echo "[$(date +%H:%M:%S)] START ${CELL}_seed${SEED}_${SC} (GPU${GPU})"
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --source_path data/nerf_llff_data/${SC} \
        -m ${OUTDIR} --seed ${SEED} \
        "${P8_CORE[@]}" "${A3_FLAGS[@]}" "${SR[@]}" \
        2>&1 | tee ${LOG}
    echo "[$(date +%H:%M:%S)] DONE  ${CELL}_seed${SEED}_${SC}"
}

SCENES=(${SCENES_OVERRIDE:-fern flower fortress horns leaves orchids room trex})
SEEDS=(${SEEDS_OVERRIDE:-42})
CELLS=${CELLS_OVERRIDE:-"Ablunt Bexc3 Bexc2"}   # A3 reuse logs/p13_lfcf

echo "=== Phase 15 shape-reg pilot (GPU${GPU}) CELLS=${CELLS} ==="
for SEED in "${SEEDS[@]}"; do
    for S in "${SCENES[@]}"; do
        for CL in $CELLS; do run_one $CL $S $SEED; done
    done
done
echo "[GPU${GPU}] Phase 15 shape-reg pilot complete."
