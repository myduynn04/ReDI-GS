#!/bin/bash
# ============================================================
# [CRSGaussian Phase 34] CRS framework LOO trên MVS backbone
# File: scripts/p34_crs_mvs_run.sh
#
# Mục đích: đo đóng góp của CRS framework trên MVS init, 3 cells:
#   no_dcycle        full − D_cycle (giữ SH-freeze + SH-rel)
#   no_shcrs         full − SH-freeze − SH-reliability (giữ D_cycle)
#   no_crs_modules   full − D_cycle − SH-freeze − SH-reliability (composite)
#
# Reference: Phase 13 A3 multi-seed N=24 = 21.330 PSNR (trim_full on MVS).
# Match methodology với Phase 27 (RoMa v1) loo_no_crs để cross-backbone.
#
# Usage (MỘT lệnh — auto split 2 GPU):
#   cd duyen/CoR-GS
#   nohup bash scripts/p34_crs_mvs_run.sh > logs/p34_crs_mvs/driver.log 2>&1 & disown
#
# Usage (manual 1 GPU subset, debug):
#   MODE=worker GPU=0 CELLS_OVERRIDE="no_dcycle" SCENES_OVERRIDE="fern" \
#       SEEDS_OVERRIDE=42 bash scripts/p34_crs_mvs_run.sh
#
# Output: output/p34_crs_mvs/<cell>/A3_seed<sd>_<scene>/
# Log:    logs/p34_crs_mvs/<cell>/A3_seed<sd>_<scene>.log
#
# Cost: 3 cells × 8 scenes × 1 seed = 24 runs × ~6min = ~144min single-GPU
#       2-GPU split ~72min wall-clock.
# ============================================================
set -eo pipefail
MODE=${MODE:-launcher}     # launcher | worker
GPU=${GPU:-0}
LOG_ROOT=logs/p34_crs_mvs
OUT_ROOT=output/p34_crs_mvs
mkdir -p ${LOG_ROOT} ${OUT_ROOT}

# ============================================================
# LAUNCHER MODE — split 8 scenes across 2 GPUs, parallel + wait
# Both GPUs run ALL 3 cells trên scene subset của mình
# ============================================================
if [ "${MODE}" = "launcher" ]; then
    GPU0_ID=${GPU0_ID:-0}
    GPU1_ID=${GPU1_ID:-1}
    GPU0_SCENES=${GPU0_SCENES:-"fern flower fortress horns"}
    GPU1_SCENES=${GPU1_SCENES:-"leaves orchids room trex"}
    CELLS_OVERRIDE=${CELLS_OVERRIDE:-"no_dcycle no_shcrs no_crs_modules"}
    SEEDS_OVERRIDE=${SEEDS_OVERRIDE:-42}

    echo "=== Phase 34 launcher — 2-GPU split × 3 cells ==="
    echo "  CELLS:  ${CELLS_OVERRIDE}"
    echo "  GPU${GPU0_ID}: ${GPU0_SCENES}"
    echo "  GPU${GPU1_ID}: ${GPU1_SCENES}"
    echo "  SEEDS:  ${SEEDS_OVERRIDE}"
    echo ""

    MODE=worker GPU=${GPU0_ID} CELLS_OVERRIDE="${CELLS_OVERRIDE}" \
        SCENES_OVERRIDE="${GPU0_SCENES}" SEEDS_OVERRIDE="${SEEDS_OVERRIDE}" \
        bash "$0" > ${LOG_ROOT}/_worker_gpu${GPU0_ID}.log 2>&1 &
    PID0=$!

    MODE=worker GPU=${GPU1_ID} CELLS_OVERRIDE="${CELLS_OVERRIDE}" \
        SCENES_OVERRIDE="${GPU1_SCENES}" SEEDS_OVERRIDE="${SEEDS_OVERRIDE}" \
        bash "$0" > ${LOG_ROOT}/_worker_gpu${GPU1_ID}.log 2>&1 &
    PID1=$!

    echo "  GPU${GPU0_ID} worker PID=${PID0}"
    echo "  GPU${GPU1_ID} worker PID=${PID1}"
    echo ""
    echo "Waiting for both workers to finish..."
    wait ${PID0}; RC0=$?
    wait ${PID1}; RC1=$?
    echo ""
    echo "=== Phase 34 launcher DONE (GPU${GPU0_ID} rc=${RC0}, GPU${GPU1_ID} rc=${RC1}) ==="
    exit $((RC0 + RC1))
fi

# ── PROTOCOL — cố định MỌI cell ──
PROTOCOL="--eval -r 8 --n_views 3 --random_background --iterations 10000 \
--densify_until_iter 5000 --densify_grad_threshold 0.0005 --gaussiansN 1 \
--sample_pseudo_interval 1 --start_sample_pseudo 500 --test_iterations 10000"

# ── Module flag-strings ──
# Note: informed_crs_init / use_crs_pruning / use_r_visible đã remove khỏi arguments
#       (Phase 20 N=24 verified-dead, Phase 24 anti-synergy DROP).
M_DEPTHCFG="--use_depth_prior --dav2_path ../Depth-Anything-V2 --crs_ema_decay 0.3 --crs_update_interval 100"
M_DCYCLE="--use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100"
M_SHFREEZE="--use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.5"
M_SHREL="--use_sh_reliability --sh_stability_warmup 1000 --sh_stability_ema_beta 0.95 --crs_w_s 0.33"
M_DROP="--use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2"
M_OPACITY="--use_opacity_decay --opacity_decay_factor 0.999"
M_EFA="--use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0 \
--lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0 \
--lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True --absdensify"

INDEP_FULL="${M_DROP} ${M_OPACITY} ${M_EFA}"

# Build flags per cell
build_flags() {
    case "$1" in
        # trim − D_cycle (giữ SH-freeze + SH-rel)
        no_dcycle)        echo "${M_DEPTHCFG} ${M_SHFREEZE} ${M_SHREL} ${INDEP_FULL}" ;;
        # trim − SH-freeze − SH-rel (giữ D_cycle)
        no_shcrs)         echo "${M_DEPTHCFG} ${M_DCYCLE} ${INDEP_FULL}" ;;
        # trim − D_cycle − SH-freeze − SH-rel (composite, all CRS off)
        no_crs_modules)   echo "${M_DEPTHCFG} ${INDEP_FULL}" ;;
        *) echo "ERROR_UNKNOWN_CELL"; return 1 ;;
    esac
}

run_one() {
    local CELL=$1; local SC=$2; local SEED=$3
    local FLAGS=$(build_flags "$CELL")
    local OUTDIR=${OUT_ROOT}/${CELL}/A3_seed${SEED}_${SC}
    local LOG=${LOG_ROOT}/${CELL}/A3_seed${SEED}_${SC}.log
    mkdir -p $(dirname ${OUTDIR}) $(dirname ${LOG})

    if [ -f "${LOG}" ] && grep -q "10000 | test" "${LOG}" 2>/dev/null; then
        echo "[SKIP] ${CELL} seed${SEED}_${SC} (log exists)"
        return
    fi
    echo "[$(date +%H:%M:%S)] START [${CELL}] seed${SEED}_${SC} (GPU${GPU})"
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --source_path data/nerf_llff_data/${SC} \
        -m ${OUTDIR} --seed ${SEED} \
        ${PROTOCOL} ${FLAGS} \
        2>&1 | tee ${LOG}
    echo "[$(date +%H:%M:%S)] DONE  [${CELL}] seed${SEED}_${SC}"
}

CELLS=(${CELLS_OVERRIDE:-no_dcycle no_shcrs no_crs_modules})
SCENES=(${SCENES_OVERRIDE:-fern flower fortress horns leaves orchids room trex})
SEEDS=(${SEEDS_OVERRIDE:-42})

echo "=== Phase 34 worker — GPU${GPU} ==="
echo "    CELLS:  ${CELLS[*]}"
echo "    SCENES: ${SCENES[*]}"
echo "    SEEDS:  ${SEEDS[*]}"

for CELL in "${CELLS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for S in "${SCENES[@]}"; do
            run_one $CELL $S $SEED
        done
    done
done

echo "[GPU${GPU}] Phase 34 worker complete."
