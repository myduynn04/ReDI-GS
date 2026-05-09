#!/bin/bash
# [CRSGaussian Phase 8] 5-config ablation runner — formula redesign + S signal + SH freeze
# Backbone: D1-O999 (DropAnSH + DECAY 0.999) — strong backbone từ Phase 7 Stage 1.
#
# 5 configs cho clean attribution diagnostic:
#   OLD         : D_DAV2 + R_old + gate prune              (= TIER1_DAV2_GATE)
#   FIX_R_DAV2  : D_DAV2 + R_visible + gate prune          (R fix alone)
#   FIX_R_DC    : D_cycle + R_visible + gate prune         (+ D_cycle)
#   FIX_RS      : D_cycle + R_visible + S + gate prune     (+ S signal)
#   FULL        : D_cycle + R_visible + S + CRS-modulated SH freeze (+ mechanism)
#
# Attribution deltas:
#   Δ_R = FIX_R_DAV2 - OLD          (R_visible alone)
#   Δ_D = FIX_R_DC   - FIX_R_DAV2   (D_cycle on clean R)
#   Δ_S = FIX_RS     - FIX_R_DC     (S signal addition)
#   Δ_M = FULL       - FIX_RS       (SH freeze mechanism)
#   Δ_FULL = FULL    - OLD          (combined Phase 8)
#
# Verdict tree:
#   FULL > 21.51 (>+0.30 vs no-CRS 21.21)  → 🟢 BREAKTHROUGH
#   FULL +0.15 ~ +0.30 vs no-CRS            → 🟡 SOLID
#   FULL < +0.15 vs no-CRS                  → 🔴 STOP, CRS axis exhausted
#
# OLD reuse: nếu logs/t2min_final/TIER1_DAV2_GATE_*.log tồn tại,
# có thể symlink reuse (cùng config). Default chạy fresh.
#
# Cost estimate (D1-O999 backbone ~6 phút/scene):
#   Phase 8 features (R_visible + S + SH freeze): ~+10-15% slowdown.
#   FULL: ~7 phút/scene × 8 = 56 phút trên 1 GPU.
#   40 runs / 2 GPU sequential ~ 100-120 phút wall-clock.

set -eo pipefail
GPU=${GPU:-0}
OUT_BASE=${OUT_DIR:-logs/p8}
LOG_DIR=${OUT_BASE}
OUT_DIR_VAL=${OUT_BASE/logs/output}
mkdir -p ${LOG_DIR} ${OUT_DIR_VAL}

# ── Backbone D1-O999 (DropAnSH + DECAY 0.999) ──
BACKBONE_ARGS=(
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
        "${BACKBONE_ARGS[@]}" "${EXTRA[@]}" \
        2>&1 | tee ${LOG}
    echo "[$(date +%H:%M:%S)] DONE  ${CFG}_${SC}"
}

SCENES=(${SCENES_OVERRIDE:-fern flower fortress horns leaves orchids room trex})
CONFIGS=${CONFIGS_OVERRIDE:-"OLD FIX_R_DAV2 FIX_R_DC FIX_RS FULL"}

for S in "${SCENES[@]}"; do
    for CFG_NAME in $CONFIGS; do
        case $CFG_NAME in
            OLD)
                # Reference: D_DAV2 + R_old, gate prune
                run_one OLD $S
                ;;
            FIX_R_DAV2)
                # R_visible alone (D vẫn DAV2)
                run_one FIX_R_DAV2 $S \
                    --use_r_visible
                ;;
            FIX_R_DC)
                # D_cycle + R_visible (Phase 7 + 8a combined)
                run_one FIX_R_DC $S \
                    --use_r_visible \
                    --use_d_cycle \
                    --d_cycle_warmup 1000 \
                    --d_cycle_sigma 5.0 \
                    --d_cycle_update_freq 100
                ;;
            FIX_RS)
                # + S signal (3-component CRS formula)
                run_one FIX_RS $S \
                    --use_r_visible \
                    --use_d_cycle \
                    --d_cycle_warmup 1000 \
                    --d_cycle_sigma 5.0 \
                    --d_cycle_update_freq 100 \
                    --use_sh_reliability \
                    --sh_stability_warmup 1000 \
                    --sh_stability_ema_beta 0.95 \
                    --crs_w_s 0.33
                ;;
            FULL)
                # + CRS-modulated SH freeze (mechanism path)
                run_one FULL $S \
                    --use_r_visible \
                    --use_d_cycle \
                    --d_cycle_warmup 1000 \
                    --d_cycle_sigma 5.0 \
                    --d_cycle_update_freq 100 \
                    --use_sh_reliability \
                    --sh_stability_warmup 1000 \
                    --sh_stability_ema_beta 0.95 \
                    --crs_w_s 0.33 \
                    --use_crs_modulated_sh_freeze \
                    --crs_freeze_start 1000 \
                    --crs_freeze_tau 0.5
                ;;
            *)
                echo "Unknown config: $CFG_NAME (expected OLD|FIX_R_DAV2|FIX_R_DC|FIX_RS|FULL)"
                exit 1
                ;;
        esac
    done
done

echo "[GPU${GPU}] Phase 8 ablation complete (LOG_DIR=${LOG_DIR}). Run scripts/p8_analyze.py."
