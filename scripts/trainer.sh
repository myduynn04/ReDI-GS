#!/bin/bash
# ============================================================
# trainer.sh — Inner training runner.
#
# Trains a single ReDI-GS configuration on the dense initialization.
# This script is called by run.sh; users normally do not invoke it
# directly. Each call trains one CONFIG over a set of scenes and seeds.
#
# Configs (selected by the CONFIG environment variable):
#   trim_full         Full ReDI-GS recipe (production):
#                     CRS + DropAnSH + opacity decay + LFCF + AbsGS.
#   trim_no_efa       trim_full minus LFCF and AbsGS.
#   trim_no_drop      trim_full minus DropAnSH.
#   trim_no_opacity   trim_full minus opacity decay.
#   trim_no_dcycle    trim_full minus the D-cycle CRS component.
#   trim_no_shcrs     trim_full minus SH-modulated freeze and SH reliability.
#   trim_no_depthcrs  trim_full minus depth supervision and the whole CRS.
#   base              Vanilla 3DGS (no ReDI-GS additions).
#
# Prerequisite: place_init.py must have been run so that fused.ply
# contains the dense initialization (RoMa v1 or another matcher).
# ============================================================
set -eo pipefail
GPU=${GPU:-0}
CONFIG=${CONFIG:?must set CONFIG (base/trim_full/trim_no_efa/trim_no_drop/trim_no_opacity/trim_no_dcycle/trim_no_shcrs/trim_no_depthcrs)}
LOG_DIR=${LOG_DIR:-logs/run/${CONFIG}}
OUT_DIR_VAL=${OUT_DIR:-output/run/${CONFIG}}
mkdir -p ${LOG_DIR} ${OUT_DIR_VAL}

# Fixed protocol for every config (do not change to keep ablations comparable).
PROTOCOL="--eval -r 8 --n_views 3 --random_background --iterations 10000 \
--densify_until_iter 5000 --densify_grad_threshold 0.0005 --gaussiansN 1 \
--sample_pseudo_interval 1 --start_sample_pseudo 500 --test_iterations 10000"

# Module flag strings.
M_DEPTHCFG="--use_depth_prior --dav2_path ../Depth-Anything-V2 --crs_ema_decay 0.3 --crs_update_interval 100"
M_DCYCLE="--use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100"
M_SHFREEZE="--use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.65"
M_SHREL="--use_sh_reliability --sh_stability_warmup 1000 --sh_stability_ema_beta 0.95 --crs_w_s 0.33"
M_DROP="--use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2"
M_OPACITY="--use_opacity_decay --opacity_decay_factor 0.999"
M_EFA="--use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0 \
--lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0 \
--lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True --absdensify"

# CRS framework: depth + D_cycle + SH-freeze + SH-stability.
CRS_TRIM="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL}"
# Module groups orthogonal to CRS.
INDEP_FULL="${M_DROP} ${M_OPACITY} ${M_EFA}"

case "${CONFIG}" in
  base)             FLAGS="" ;;
  trim_full)        FLAGS="${CRS_TRIM} ${INDEP_FULL}" ;;
  trim_no_efa)      FLAGS="${CRS_TRIM} ${M_DROP} ${M_OPACITY}" ;;
  trim_no_drop)     FLAGS="${CRS_TRIM} ${M_OPACITY} ${M_EFA}" ;;
  trim_no_opacity)  FLAGS="${CRS_TRIM} ${M_DROP} ${M_EFA}" ;;
  trim_no_dcycle)   FLAGS="${M_DEPTHCFG} ${M_SHFREEZE} ${M_SHREL} ${INDEP_FULL}" ;;
  trim_no_shcrs)    FLAGS="${M_DEPTHCFG} ${M_DCYCLE} ${INDEP_FULL}" ;;
  trim_no_depthcrs) FLAGS="${INDEP_FULL}" ;;
  *) echo "ERROR: invalid CONFIG: ${CONFIG}"; exit 1 ;;
esac

run_one() {
    local SC=$1; local SEED=$2
    local OUTDIR=${OUT_DIR_VAL}/A3_seed${SEED}_${SC}
    local LOG=${LOG_DIR}/A3_seed${SEED}_${SC}.log
    echo "[$(date +%H:%M:%S)] START [${CONFIG}] seed${SEED}_${SC} (GPU${GPU})"
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --source_path data/nerf_llff_data/${SC} \
        -m ${OUTDIR} --seed ${SEED} \
        ${PROTOCOL} ${FLAGS} \
        2>&1 | tee ${LOG}
    echo "[$(date +%H:%M:%S)] DONE  [${CONFIG}] seed${SEED}_${SC}"
}

SCENES=(${SCENES_OVERRIDE:-fern flower fortress horns leaves orchids room trex})
SEEDS=(${SEEDS_OVERRIDE:-42 137 9999})

echo "=== trainer.sh CONFIG=${CONFIG} GPU=${GPU} ==="
echo "    Prerequisite: fused.ply has been replaced by the chosen dense init."
echo "    FLAGS: ${FLAGS:-<none — BASE>}"
echo "    SCENES=${SCENES[*]} SEEDS=${SEEDS[*]}"
for SEED in "${SEEDS[@]}"; do
    for S in "${SCENES[@]}"; do
        run_one $S $SEED
    done
done
echo "[GPU${GPU}] Phase 20 DENSE ablation CONFIG=${CONFIG} complete."
