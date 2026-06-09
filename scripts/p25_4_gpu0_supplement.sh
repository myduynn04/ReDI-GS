#!/bin/bash
# ============================================================
# [CRSGaussian Phase 25_4] GPU 0 supplement — parallel với GPU 1
# File: scripts/p25_4_gpu0_supplement.sh
#
# Mục đích: GPU 1 đang chậm sequential 24 cells (~6-7h). Cho GPU 0 chạy
# parallel từ CUỐI queue GPU 1 lên đầu → gặp nhau giữa đường.
# Skip-if-exists protect race (cell GPU 1 vừa xong → GPU 0 đến → skip).
#
# GPU 1 queue: seed42 → seed137 → seed9999, scenes fern→flower→...→trex
# GPU 0 supplement: seed9999 → seed137 → seed42 (reverse),
#                    scenes trex→room→orchids→leaves→horns→fortress→flower→fern (reverse)
#
# Pre-requisite:
#   conda activate corgs
#   cd ~/workspace/representation-3d/duyen/CoR-GS
#   chmod +x scripts/p25_4_gpu0_supplement.sh
#   nvidia-smi → confirm GPU 0 có ≥ 12 GB free
#
# Usage:
#   nohup bash scripts/p25_4_gpu0_supplement.sh > /tmp/p25_4_gpu0_sup.log 2>&1 &
#   disown
# ============================================================

set -e

# ── Verify init = RoMa v1 9_views ──
FERN_PLY="data/nerf_llff_data/fern/9_views/dense/fused.ply"
FERN_V1="data/nerf_llff_data/fern/9_views/dense/fused.ply.romav1"
if ! cmp -s "$FERN_PLY" "$FERN_V1"; then
    echo "❌ 9_views/fused.ply ≠ fused.ply.romav1"
    exit 1
fi
echo "✓ verified 9_views/fused.ply = RoMa v1"

# ── Phase 22 protocol + n_views=9 (identical p25_4_9view_run.sh) ──
PROTOCOL="--eval -r 8 --n_views 9 --random_background --iterations 10000 \
--densify_until_iter 5000 --densify_grad_threshold 0.0005 --gaussiansN 1 \
--sample_pseudo_interval 1 --start_sample_pseudo 500 --test_iterations 10000"

M_DEPTHCFG="--use_depth_prior --dav2_path ../Depth-Anything-V2 --crs_ema_decay 0.3 --crs_update_interval 100"
M_DCYCLE="--use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100"
M_SHFREEZE="--use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.5"
M_SHREL="--use_sh_reliability --sh_stability_warmup 1000 --sh_stability_ema_beta 0.95 --crs_w_s 0.33"
M_DROP="--use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2"
M_OPACITY="--use_opacity_decay --opacity_decay_factor 0.999"
M_EFA="--use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0 \
--lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0 \
--lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True --absdensify"

A3_TRIM="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL} ${M_DROP} ${M_OPACITY} ${M_EFA}"

# ── REVERSE order: GPU 0 đi từ cuối queue GPU 1 lên ──
SEEDS_REV=(9999 137 42)
SCENES_REV=(trex room orchids leaves horns fortress flower fern)

LOG_ROOT="logs/p25_4_9view"
OUT_ROOT="output/p25_4_9view"
MASTER_LOG="${LOG_ROOT}/_supplement_gpu0.log"
mkdir -p "$LOG_ROOT" "$OUT_ROOT"

run_one() {
    local SEED=$1; local SC=$2
    local OUTDIR=${OUT_ROOT}/A3_seed${SEED}_${SC}
    local LOG=${LOG_ROOT}/A3_seed${SEED}_${SC}.log

    # Skip nếu đã done (GPU 1 hoặc lần trước)
    if [ -f "$LOG" ] && grep -q "Best test PSNR" "$LOG" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] SKIP  [P25_4sup seed${SEED}/${SC}] (done)" | tee -a "$MASTER_LOG"
        return 0
    fi

    echo "[$(date +%H:%M:%S)] START [P25_4sup seed${SEED}/${SC}] GPU0" | tee -a "$MASTER_LOG"
    CUDA_VISIBLE_DEVICES=0 python -u train.py \
        --source_path data/nerf_llff_data/${SC} \
        -m ${OUTDIR} --seed ${SEED} \
        ${PROTOCOL} ${A3_TRIM} \
        2>&1 | tee ${LOG} > /dev/null
    echo "[$(date +%H:%M:%S)] DONE  [P25_4sup seed${SEED}/${SC}]" | tee -a "$MASTER_LOG"
}

echo "[P25_4 GPU0 supplement] start $(date '+%Y-%m-%d %H:%M:%S')" | tee "$MASTER_LOG"
echo "  reverse order: seeds ${SEEDS_REV[*]} × scenes ${SCENES_REV[*]}" | tee -a "$MASTER_LOG"
echo "  total candidates: ${#SEEDS_REV[@]} × ${#SCENES_REV[@]} = $((${#SEEDS_REV[@]} * ${#SCENES_REV[@]})) runs (skip-if-exists)" | tee -a "$MASTER_LOG"

for SEED in "${SEEDS_REV[@]}"; do
    for SC in "${SCENES_REV[@]}"; do
        run_one "$SEED" "$SC"
    done
done

echo "[P25_4 GPU0 supplement] done $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
