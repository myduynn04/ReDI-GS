#!/bin/bash
# [CRSGaussian Tier 2-min + Phase 7-FINAL] Master ablation script
#
# Phase 7 (original — Tier 2-min trên Phase 5 weak backbone):
#   B0 / DCYCLE / LWEIGHT / TIER2MIN (4 configs)
#
# [CRSGaussian Phase 7-FINAL] Confirmatory test trên strong backbones:
#   Stage 1 (D1-O999 family — DropAnSH + DECAY 0.999):
#     TIER1_DAV2_GATE  : reference (CRS prune ON, D_DAV2)         ~21.13 dB
#     TIER1_DC_GATE    : signal upgrade alone (D_cycle, gate)
#     TIER1_DC_LW      : mechanism upgrade (D_cycle + loss reweighter, no gate)
#   Stage 2 (A1+B1β family — sh1 + freeze_sh + dropout, no DECAY):
#     TIER2_DAV2_GATE  : reference                                 ~20.96 dB
#     TIER2_DC_GATE    : signal upgrade cross-backbone confirmation
#
# OUT_DIR env var (optional): override output + log dir (default logs/t2min, output/t2min).
#   OUT_DIR=logs/t2min_final → logs vào logs/t2min_final/, output vào output/t2min_final/

set -eo pipefail
GPU=${GPU:-0}

# [CRSGaussian Phase 7-FINAL] Configurable output dir cho staged ablation
OUT_BASE=${OUT_DIR:-logs/t2min}
LOG_DIR=${OUT_BASE}
OUT_DIR_VAL=${OUT_BASE/logs/output}
mkdir -p ${LOG_DIR} ${OUT_DIR_VAL}

# ── BACKBONE_ARGS — Phase 7 original (Phase 5 weak backbone) ──
BACKBONE_ARGS=(
    --eval -r 8 --n_views 3 --random_background
    --iterations 10000
    --densify_until_iter 5000 --densify_grad_threshold 0.0005
    --gaussiansN 1
    --use_depth_prior --dav2_path ../Depth-Anything-V2
    --use_crs_pruning
    --informed_crs_init
    --crs_init_use_view False
    --crs_init_w_reproj 0.4 --crs_init_w_depth 0.6 --crs_init_w_view 0.0
    --crs_ema_decay 0.3 --crs_update_interval 100
    --sample_pseudo_interval 1 --start_sample_pseudo 500
    --test_iterations 10000
)

# ── [CRSGaussian Phase 7-FINAL] TIER1_BACKBONE_ARGS — D1-O999 family ──
# Adds DropAnSH + DECAY 0.999 to Phase 7 BACKBONE.
# This is the strong backbone where current best 21.21 dB lives.
TIER1_BACKBONE_ARGS=(
    --eval -r 8 --n_views 3 --random_background
    --iterations 10000
    --densify_until_iter 5000 --densify_grad_threshold 0.0005
    --gaussiansN 1
    --use_depth_prior --dav2_path ../Depth-Anything-V2
    --informed_crs_init
    --crs_init_use_view False
    --crs_init_w_reproj 0.4 --crs_init_w_depth 0.6 --crs_init_w_view 0.0
    --crs_ema_decay 0.3 --crs_update_interval 100
    # DropAnSH (verified flag names trong arguments/__init__.py:181-184)
    --use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2
    # Opacity decay 0.999 (verified arguments/__init__.py:136-137; rule project_budget_scaling)
    --use_opacity_decay --opacity_decay_factor 0.999
    --sample_pseudo_interval 1 --start_sample_pseudo 500
    --test_iterations 10000
)

# ── [CRSGaussian Phase 7-FINAL] TIER2_BACKBONE_ARGS — A1+B1β family ──
# Different regularization type: low SH degree + freeze SH + uniform dropout.
# No DECAY, no DropAnSH. Cross-backbone confirmation backbone.
TIER2_BACKBONE_ARGS=(
    --eval -r 8 --n_views 3 --random_background
    --iterations 10000
    --densify_until_iter 5000 --densify_grad_threshold 0.0005
    --gaussiansN 1
    --use_depth_prior --dav2_path ../Depth-Anything-V2
    --informed_crs_init
    --crs_init_use_view False
    --crs_init_w_reproj 0.4 --crs_init_w_depth 0.6 --crs_init_w_view 0.0
    --crs_ema_decay 0.3 --crs_update_interval 100
    # A1: low SH degree + freeze SH after iter 1000 (arguments/__init__.py:72,264)
    --sh_degree 1
    --freeze_sh_after 1000
    # B1β: uniform dropout từ iter 1000 (arguments/__init__.py:168-174)
    --use_dropout --dropout_mode uniform
    --dropout_base 0.2 --dropout_start_iter 1000
    --sample_pseudo_interval 1 --start_sample_pseudo 500
    --test_iterations 10000
)

# ── run_one helper — accept BACKBONE flag từ caller, log/output theo OUT_DIR ──
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

# Default Phase 7 Tier 2-min (3 NEW configs). Phase 7-FINAL configs require explicit override.
CONFIGS=${CONFIGS_OVERRIDE:-"DCYCLE LWEIGHT TIER2MIN"}

for S in "${SCENES[@]}"; do
    for CFG_NAME in $CONFIGS; do
        case $CFG_NAME in
            # ── Phase 7 original (Phase 5 weak backbone) ──
            B0)
                run_one B0 $S "${BACKBONE_ARGS[@]}"
                ;;
            DCYCLE)
                run_one DCYCLE $S "${BACKBONE_ARGS[@]}" \
                    --use_d_cycle \
                    --d_cycle_warmup 1000 \
                    --d_cycle_sigma 5.0 \
                    --d_cycle_update_freq 100
                ;;
            LWEIGHT)
                run_one LWEIGHT $S "${BACKBONE_ARGS[@]}" \
                    --use_loss_reweight \
                    --lossw_gamma 0.5 \
                    --lossw_render_freq 100
                ;;
            TIER2MIN)
                run_one TIER2MIN $S "${BACKBONE_ARGS[@]}" \
                    --use_d_cycle \
                    --d_cycle_warmup 1000 \
                    --d_cycle_sigma 5.0 \
                    --d_cycle_update_freq 100 \
                    --use_loss_reweight \
                    --lossw_gamma 0.5 \
                    --lossw_render_freq 100
                ;;

            # ── [CRSGaussian Phase 7-FINAL] Stage 1: D1-O999 family ──
            # Reference TIER1_DAV2_GATE expected ~21.13 dB AVG (matches D1-O999 from Phase 6).
            # TIER1_DC_GATE: thay D_DAV2 bằng D_cycle, GIỮ gate prune mechanism.
            # TIER1_DC_LW: cả 2 upgrade — D_cycle + loss reweighter, KHÔNG dùng gate prune
            #              (lý do: loss reweighter là alternative mechanism cho gate, không stack).
            TIER1_DAV2_GATE)
                run_one TIER1_DAV2_GATE $S "${TIER1_BACKBONE_ARGS[@]}" \
                    --use_crs_pruning
                ;;
            TIER1_DC_GATE)
                run_one TIER1_DC_GATE $S "${TIER1_BACKBONE_ARGS[@]}" \
                    --use_crs_pruning \
                    --use_d_cycle \
                    --d_cycle_warmup 1000 \
                    --d_cycle_sigma 5.0 \
                    --d_cycle_update_freq 100
                ;;
            TIER1_DC_LW)
                # Lưu ý: KHÔNG --use_crs_pruning. LW thay thế gate prune mechanism.
                run_one TIER1_DC_LW $S "${TIER1_BACKBONE_ARGS[@]}" \
                    --use_d_cycle \
                    --d_cycle_warmup 1000 \
                    --d_cycle_sigma 5.0 \
                    --d_cycle_update_freq 100 \
                    --use_loss_reweight \
                    --lossw_gamma 0.5 \
                    --lossw_render_freq 100
                ;;

            # ── [CRSGaussian Phase 7-FINAL] Stage 2: A1+B1β family ──
            # Cross-backbone confirmation. Run CHỈ KHI Stage 1 NOT 🔴 (best > 21.16).
            # Reference TIER2_DAV2_GATE expected ~20.96 dB.
            TIER2_DAV2_GATE)
                run_one TIER2_DAV2_GATE $S "${TIER2_BACKBONE_ARGS[@]}" \
                    --use_crs_pruning
                ;;
            TIER2_DC_GATE)
                run_one TIER2_DC_GATE $S "${TIER2_BACKBONE_ARGS[@]}" \
                    --use_crs_pruning \
                    --use_d_cycle \
                    --d_cycle_warmup 1000 \
                    --d_cycle_sigma 5.0 \
                    --d_cycle_update_freq 100
                ;;
            *)
                echo "Unknown config: $CFG_NAME"
                echo "Valid: B0|DCYCLE|LWEIGHT|TIER2MIN|TIER1_DAV2_GATE|TIER1_DC_GATE|TIER1_DC_LW|TIER2_DAV2_GATE|TIER2_DC_GATE"
                exit 1
                ;;
        esac
    done
done

echo "[GPU${GPU}] Ablation complete (LOG_DIR=${LOG_DIR}). Run scripts/t2min_final_analyze.py to read verdict."
