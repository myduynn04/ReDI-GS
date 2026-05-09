#!/bin/bash
# [CRSGaussian Phase 9] 5-config ablation runner — formula simplification + cross-backbone
#
# Test 1 — D1-O999 backbone simplification (3 NEW configs, FULL reuse từ Phase 8):
#   FULL          (reuse logs/p8/FULL_*.log)              — D_cycle + R_visible + S + CRS-mod freeze
#   FULL_NoS      D_cycle + R_visible + CRS-mod freeze    — Test H2: drop S
#   D_ONLY_FREEZE D_cycle + CRS-mod freeze (no R, no S)   — Test H1: drop R
#   D_ONLY_GATE   D_cycle + gate prune (no R, no S, no SH freeze)  — Isolate D alone
#
# Test 2 — A1+B1β cross-backbone (2 NEW configs):
#   A1B1_BASELINE Pure A1 + B1β (no CRS additions)        — Reproduce Track A+B ref ~20.96
#   A1B1_BEST     A1+B1β NO global SH freeze + R_visible + D_cycle + CRS-mod freeze
#                 — Test H3: SH freeze universal mechanism cross-backbone
#
# Hypotheses:
#   H1 (drop R): D_ONLY_FREEZE > FULL_NoS by +0.05-0.20 → R contributes negatively
#   H2 (drop S): FULL_NoS ≈ FULL (within ±0.05)         → S adds nothing
#   H3 (cross): A1B1_BEST > A1B1_BASELINE by +0.10-0.30  → SH freeze universal
#
# Cost: 5 configs × 8 scenes = 40 runs. Reuse FULL từ Phase 8 → 32 NEW.
# ~100 phút wall-clock 2 GPU sequential.

set -eo pipefail
GPU=${GPU:-0}
OUT_BASE=${OUT_DIR:-logs/p9}
LOG_DIR=${OUT_BASE}
OUT_DIR_VAL=${OUT_BASE/logs/output}
mkdir -p ${LOG_DIR} ${OUT_DIR_VAL}

# ── D1-O999 backbone (Test 1) ──
D1_O999_BACKBONE=(
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

# ── A1+B1β backbone (Test 2) — KHÔNG có DropAnSH, KHÔNG có DECAY ──
# Mặc định có --freeze_sh_after 1000 (A1 global freeze).
# A1B1_BEST sẽ truyền --disable_global_sh_freeze để bypass.
A1B1_BACKBONE=(
    --eval -r 8 --n_views 3 --random_background
    --iterations 10000
    --densify_until_iter 5000 --densify_grad_threshold 0.0005
    --gaussiansN 1
    --use_depth_prior --dav2_path ../Depth-Anything-V2
    --use_crs_pruning --informed_crs_init
    --crs_init_use_view False
    --crs_init_w_reproj 0.4 --crs_init_w_depth 0.6 --crs_init_w_view 0.0
    --crs_ema_decay 0.3 --crs_update_interval 100
    --sh_degree 1
    --freeze_sh_after 1000
    --use_dropout --dropout_mode uniform --dropout_base 0.2 --dropout_start_iter 1000
    --sample_pseudo_interval 1 --start_sample_pseudo 500
    --test_iterations 10000
)

# Common D_cycle + CRS-mod freeze args (reuse cho nhiều configs)
DC_FREEZE_ARGS=(
    --use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100
    --use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.5
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
        "${EXTRA[@]}" \
        2>&1 | tee ${LOG}
    echo "[$(date +%H:%M:%S)] DONE  ${CFG}_${SC}"
}

SCENES=(${SCENES_OVERRIDE:-fern flower fortress horns leaves orchids room trex})
CONFIGS=${CONFIGS_OVERRIDE:-"FULL_NoS D_ONLY_FREEZE D_ONLY_GATE A1B1_BASELINE A1B1_BEST"}

for S in "${SCENES[@]}"; do
    for CFG_NAME in $CONFIGS; do
        case $CFG_NAME in
            # ── Test 1 — D1-O999 backbone simplification ──
            FULL)
                # Phase 8 FULL — reuse logs nếu có. Chỉ chạy nếu user explicit.
                run_one FULL $S "${D1_O999_BACKBONE[@]}" \
                    --use_r_visible \
                    "${DC_FREEZE_ARGS[@]}" \
                    --use_sh_reliability --sh_stability_warmup 1000 \
                    --sh_stability_ema_beta 0.95 --crs_w_s 0.33
                ;;
            FULL_NoS)
                # H2 test: D_cycle + R_visible + CRS-mod freeze, NO S signal.
                run_one FULL_NoS $S "${D1_O999_BACKBONE[@]}" \
                    --use_r_visible \
                    "${DC_FREEZE_ARGS[@]}"
                ;;
            D_ONLY_FREEZE)
                # H1 test: D_cycle + CRS-mod freeze, NO R, NO S (D-only formula).
                run_one D_ONLY_FREEZE $S "${D1_O999_BACKBONE[@]}" \
                    --disable_r_signal \
                    "${DC_FREEZE_ARGS[@]}"
                ;;
            D_ONLY_GATE)
                # Isolate D signal: D_cycle + gate prune (CRS prune from BACKBONE),
                # NO R, NO S, NO SH freeze mechanism. Reference cho Δ_freeze attribution.
                run_one D_ONLY_GATE $S "${D1_O999_BACKBONE[@]}" \
                    --disable_r_signal \
                    --use_d_cycle --d_cycle_warmup 1000 \
                    --d_cycle_sigma 5.0 --d_cycle_update_freq 100
                ;;

            # ── Test 2 — A1+B1β cross-backbone ──
            A1B1_BASELINE)
                # Reproduce Track A+B reference (~20.96 dB).
                run_one A1B1_BASELINE $S "${A1B1_BACKBONE[@]}"
                ;;
            A1B1_BEST)
                # H3 test: replace A1 global freeze với CRS-mod per-Gaussian freeze.
                # + R_visible + D_cycle (Phase 8 components carry).
                run_one A1B1_BEST $S "${A1B1_BACKBONE[@]}" \
                    --disable_global_sh_freeze \
                    --use_r_visible \
                    "${DC_FREEZE_ARGS[@]}"
                ;;
            *)
                echo "Unknown config: $CFG_NAME"
                echo "Valid: FULL|FULL_NoS|D_ONLY_FREEZE|D_ONLY_GATE|A1B1_BASELINE|A1B1_BEST"
                exit 1
                ;;
        esac
    done
done

echo "[GPU${GPU}] Phase 9 ablation complete (LOG_DIR=${LOG_DIR}). Run scripts/p9_analyze.py."
