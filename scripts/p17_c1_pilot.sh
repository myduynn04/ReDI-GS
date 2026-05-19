#!/bin/bash
# ============================================================
# [CRSGaussian Phase 17 — C1] DSINE normal-prior pilot (trên A3-clean).
#
# PREREQUISITE: chạy preprocess TRƯỚC (ghi DSINE normal/train-view):
#   SCENES="fern flower fortress horns leaves orchids room trex" \
#       python scripts/p17_c1_preprocess_dsine.py
#
# Cells (A3 baseline = REUSE logs/p13_lfcf/A3_seed{seed}_* — KHÔNG run):
#   C1L10 : --use_c1_normal --c1_normal_lambda 0.10  → λ CHÍNH
#           PRE-REGISTERED, DUY NHẤT quyết GO/NO. 8scene × 3seed = N=24
#           (chuẩn Phase-13 / project_3dgs_variance_floor).
#   C1L05 : λ=0.05  → SENSITIVITY-ONLY (8×seed42). KHÔNG lật verdict.
#   C1L20 : λ=0.20  → SENSITIVITY-ONLY (8×seed42). KHÔNG lật verdict.
#   VOFF  : KHÔNG --use_c1_normal = A3 → verify flag-OFF byte-identical
#           (N-criterion vs logs/p13_lfcf/A3_seed42_<scene>).
#
# PRE-REGISTERED (khóa, decisions_log Phase-17, chống cherry-pick /
#   multiple-comparison — đúng audit #3 đã baked vào p16):
#   GO ⟺ C1L10: Δ_mean(vs A3) ≥ +0.10 N=24  VÀ  95%CI loại 0  VÀ
#        ≥7/8 scene non-neg + horns ≥ −0.05  VÀ  report train-time+N.
#   Khác = NO → đóng C1 (đã trả pilot). λ=0.05/0.20 CHỈ sensitivity.
#   PREDICTION (logged trước chạy): ∇(depth→normal finite-diff) nhiễu
#   (dn-splatter detach đúng chỗ này) → dự đoán degrade/no-gain, tập
#   trung horns/foliage. Pilot ra vậy = XÁC NHẬN, KHÔNG re-engineer.
#
# P8_CORE + A3_FLAGS = EXACT copy p15_shape_pilot.sh:38-64 (= CFG A3
#   p13_lfcf_multiseed.sh). KHÔNG infer. feedback_use_both_gpus 2-GPU.
#
# Pre-pilot verify (CHẠY TRƯỚC, rẻ — flag-OFF KHÔNG phá A3):
#   GPU=0 SCENES_OVERRIDE=orchids CELLS_OVERRIDE=VOFF SEEDS_OVERRIDE=42 \
#       bash scripts/p17_c1_pilot.sh
#     → so N vs logs/p13_lfcf/A3_seed42_orchids (reldiff<5%) ; + 1 cell
#       C1L10 orchids smoke không crash + loss finite.
#
# Usage (PRIMARY N=24, 2 GPU):
#   GPU=0 CELLS_OVERRIDE=C1L10 SEEDS_OVERRIDE="42 137 9999" \
#     SCENES_OVERRIDE="fern flower fortress horns" \
#     bash scripts/p17_c1_pilot.sh > logs/p17_c1/g0.log 2>&1 &
#   GPU=1 CELLS_OVERRIDE=C1L10 SEEDS_OVERRIDE="42 137 9999" \
#     SCENES_OVERRIDE="leaves orchids room trex" \
#     bash scripts/p17_c1_pilot.sh > logs/p17_c1/g1.log 2>&1 &
#   wait
#   # then sensitivity (seed42 only, non-verdict):
#   GPU=0 CELLS_OVERRIDE="C1L05 C1L20" SEEDS_OVERRIDE=42 \
#     SCENES_OVERRIDE="fern flower fortress horns" bash scripts/p17_c1_pilot.sh &
#   GPU=1 CELLS_OVERRIDE="C1L05 C1L20" SEEDS_OVERRIDE=42 \
#     SCENES_OVERRIDE="leaves orchids room trex" bash scripts/p17_c1_pilot.sh &
#   wait
#   python scripts/p17_c1_analyze.py
# ============================================================
set -eo pipefail
GPU=${GPU:-0}
LOG_DIR=${LOG_DIR:-logs/p17_c1}
OUT_DIR_VAL=${OUT_DIR:-output/p17_c1}
mkdir -p ${LOG_DIR} ${OUT_DIR_VAL}

# P8_CORE + A3_FLAGS = EXACT copy p15_shape_pilot.sh:38-64 (CFG A3).
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
    local C1=()
    case $CELL in
        C1L10) C1=(--use_c1_normal --c1_normal_lambda 0.10) ;;
        C1L05) C1=(--use_c1_normal --c1_normal_lambda 0.05) ;;
        C1L20) C1=(--use_c1_normal --c1_normal_lambda 0.20) ;;
        VOFF)  C1=() ;;
        *) echo "Unknown CELL $CELL (C1L10|C1L05|C1L20|VOFF)"; exit 1 ;;
    esac
    local OUTDIR=${OUT_DIR_VAL}/${CELL}_seed${SEED}_${SC}
    local LOG=${LOG_DIR}/${CELL}_seed${SEED}_${SC}.log
    echo "[$(date +%H:%M:%S)] START ${CELL}_seed${SEED}_${SC} (GPU${GPU})"
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --source_path data/nerf_llff_data/${SC} \
        -m ${OUTDIR} --seed ${SEED} \
        "${P8_CORE[@]}" "${A3_FLAGS[@]}" "${C1[@]}" \
        2>&1 | tee ${LOG}
    echo "[$(date +%H:%M:%S)] DONE  ${CELL}_seed${SEED}_${SC}"
}

SCENES=(${SCENES_OVERRIDE:-fern flower fortress horns leaves orchids room trex})
SEEDS=(${SEEDS_OVERRIDE:-42})
CELLS=${CELLS_OVERRIDE:-"C1L10"}   # A3 baseline reuse logs/p13_lfcf

echo "=== Phase 17 C1 pilot (GPU${GPU}) CELLS=${CELLS} SEEDS=${SEEDS[*]} ==="
for SEED in "${SEEDS[@]}"; do
    for S in "${SCENES[@]}"; do
        for CL in $CELLS; do run_one $CL $S $SEED; done
    done
done
echo "[GPU${GPU}] Phase 17 C1 pilot complete."
