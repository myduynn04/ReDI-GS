#!/bin/bash
# ============================================================
# [CRSGaussian Phase 10A] DIAGNOSTIC — Option B: 6 runs (2 scenes × 3 configs)
#
# Phase 10A AUGMENT/REPLACE FAILED hard (Δ_AUG=−0.898, Δ_REP=−3.529).
# Trước khi pivot Phase 10B (Feature MPC), isolate WHICH lever (filter
# vs densify scaling) cần thiết để cứu Phase 10A — hoặc confirm DEAD.
#
# 3 configs — mỗi cái isolate 1 hypothesis:
#
# FILTER (aggressive DUSt3R filter, KEEP Phase 8 densify defaults):
#   conf_threshold 1.5 → 3.0   (gấp 2× — chỉ giữ DUSt3R points reliable)
#   max_points     50K → 10K   (giảm 5× — tránh init Gaussian dày quá)
#   dedupe_radius  0.01 → 0.05 (rộng 5× — loại duplicate aggressive)
#   densify_grad_threshold 0.0005, dropansh_pa 0.02 (PHASE 8 DEFAULT)
#   → Test H_filter: lọc cứng DUSt3R noise có cứu được không?
#
# DENSIFY (scale densify thresholds, KEEP Phase 10A filter defaults):
#   conf_threshold 1.5, max_points 50K, dedupe_radius 0.01 (PHASE 10A DEFAULT)
#   densify_grad_threshold 0.0005 → 0.002  (cao 4× — scale theo init density)
#   dropansh_pa            0.02   → 0.05   (drop nhiều hơn — population lớn)
#   → Test H_densify: tune densify thresholds có cứu được không?
#
# BOTH (FILTER + DENSIFY combined):
#   Tất cả thay đổi gộp → test có synergy không hay redundant.
#
# Scenes — cặp easy+hard:
#   orchids — Δ_AUG = −0.274 (smallest hit, easy scene để detect signal)
#   leaves  — Δ_AUG = −3.727 (worst hit, textureless foliage stress test)
#   → Nếu cả 2 flip positive → strong evidence, scale 8 scenes
#   → Nếu chỉ orchids flip → easy scene only, fundamental issue
#   → Nếu cả 2 vẫn âm → Phase 10A confirmed DEAD, pivot 10B
#
# Reuse cache cũ (cache/dust3r_init/*.npz) — KHÔNG cần re-precompute.
# Cost: 6 runs × ~6 phút = ~36 phút wall-clock.
#
# Decision tree (Δ_PSNR vs P8_FULL, threshold +0.05 = noise floor):
#   ≥ +0.05  → Genuine improvement, scale lên 8 scenes
#   0..+0.05 → No-signal (within noise) → likely DEAD
#   < 0      → Confirmed FAIL, pivot Phase 10B
# ============================================================

set -eo pipefail
GPU=${GPU:-0}
OUT_BASE=${OUT_DIR:-logs/p10a}
LOG_DIR=${OUT_BASE}
OUT_DIR_VAL=${OUT_BASE/logs/output}
mkdir -p ${LOG_DIR} ${OUT_DIR_VAL}

# Phase 8 FULL backbone — exact components LOCKED ở 21.335 dB
P8_FULL_BACKBONE=(
    --eval -r 8 --n_views 3 --random_background
    --iterations 10000
    --densify_until_iter 5000
    --gaussiansN 1
    --use_depth_prior --dav2_path ../Depth-Anything-V2
    --use_crs_pruning --informed_crs_init
    --crs_init_use_view False
    --crs_init_w_reproj 0.4 --crs_init_w_depth 0.6 --crs_init_w_view 0.0
    --crs_ema_decay 0.3 --crs_update_interval 100
    --use_opacity_decay --opacity_decay_factor 0.999
    --sample_pseudo_interval 1 --start_sample_pseudo 500
    --test_iterations 10000
    # Phase 8 components
    --use_r_visible
    --use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100
    --use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.5
    --use_sh_reliability --sh_stability_warmup 1000
    --sh_stability_ema_beta 0.95 --crs_w_s 0.33
    # Phase 10A dense init master switch — AUGMENT mode (giữ COLMAP)
    --use_dense_init
    --dust3r_cache_dir cache/dust3r_init
    --dense_init_mode augment
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

SCENES=(${SCENES_OVERRIDE:-orchids leaves})
CONFIGS=${CONFIGS_OVERRIDE:-"FILTER DENSIFY BOTH"}

for S in "${SCENES[@]}"; do
    for CFG_NAME in $CONFIGS; do
        case $CFG_NAME in
            FILTER)
                # Aggressive DUSt3R filter, Phase 8 densify defaults.
                # Isolate H_filter: lọc cứng DUSt3R noise có cứu được không?
                run_one FILTER $S "${P8_FULL_BACKBONE[@]}" \
                    --densify_grad_threshold 0.0005 \
                    --use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2 \
                    --dense_init_conf_threshold 3.0 \
                    --dense_init_max_points 10000 \
                    --dense_init_dedupe_radius 0.05
                ;;
            DENSIFY)
                # Phase 10A filter defaults, scaled densify thresholds.
                # Isolate H_densify: tune densify có cứu được không?
                run_one DENSIFY $S "${P8_FULL_BACKBONE[@]}" \
                    --densify_grad_threshold 0.002 \
                    --use_dropansh --dropansh_pa 0.05 --dropansh_psh 0.2 \
                    --dense_init_conf_threshold 1.5 \
                    --dense_init_max_points 50000 \
                    --dense_init_dedupe_radius 0.01
                ;;
            BOTH)
                # FILTER + DENSIFY combined — test synergy/redundancy.
                run_one BOTH $S "${P8_FULL_BACKBONE[@]}" \
                    --densify_grad_threshold 0.002 \
                    --use_dropansh --dropansh_pa 0.05 --dropansh_psh 0.2 \
                    --dense_init_conf_threshold 3.0 \
                    --dense_init_max_points 10000 \
                    --dense_init_dedupe_radius 0.05
                ;;
            *)
                echo "Unknown config: $CFG_NAME"
                echo "Valid: FILTER|DENSIFY|BOTH"
                exit 1
                ;;
        esac
    done
done

echo ""
echo "[GPU${GPU}] Phase 10A DIAGNOSTIC complete (LOG_DIR=${LOG_DIR})."
echo ""
echo "=== Quick result check ==="
echo ""
echo "P8_FULL reference per-scene (from Phase 8/9):"
echo "  orchids: 16.761"
echo "  leaves:  18.520"
echo ""
printf "%-10s %-10s %-10s %-10s %-10s\n" "scene" "FILTER" "DENSIFY" "BOTH" "Δ vs P8"
echo "----------------------------------------------------"
for SC in "${SCENES[@]}"; do
    REF_PSNR=""
    case $SC in
        orchids) REF_PSNR=16.761 ;;
        leaves)  REF_PSNR=18.520 ;;
        horns)   REF_PSNR=20.332 ;;
        fern)    REF_PSNR=23.127 ;;
        flower)  REF_PSNR=21.036 ;;
        fortress) REF_PSNR=24.526 ;;
        room)    REF_PSNR=23.071 ;;
        trex)    REF_PSNR=23.306 ;;
    esac
    row="${SC}"
    for CFG in FILTER DENSIFY BOTH; do
        LOG="${LOG_DIR}/${CFG}_${SC}.log"
        if [ -f "$LOG" ]; then
            PSNR=$(grep "Best test PSNR:" "$LOG" | tail -1 | awk -F: '{print $2}' | tr -d ' ')
            row="${row} ${PSNR}"
        else
            row="${row} -"
        fi
    done
    printf "%-10s %-10s %-10s %-10s (P8=%s)\n" $row $REF_PSNR
done

echo ""
echo "Decision threshold (Δ_PSNR vs P8_FULL, noise floor +0.05):"
echo "  ≥ +0.05    → Genuine improvement → scale lên 8 scenes"
echo "  0..+0.05   → No-signal (within noise) → likely DEAD"
echo "  < 0        → Confirmed FAIL → pivot Phase 10B"
