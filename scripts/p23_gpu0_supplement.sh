#!/bin/bash
# ============================================================
# [CRSGaussian Phase 23] GPU 0 supplement — parallel với GPU 1 đang chạy
# File: scripts/p23_gpu0_supplement.sh  (KEEP LOCAL — server-only)
#
# Mục đích: tận dụng GPU 0 rảnh (loop chính đã end) để:
#   1. Chạy single_depthcrs (8 scenes, MỚI, GPU 1 sẽ chạm sau cùng → safe)
#   2. Re-run partial logs trong các configs GPU 0 đã end (base 7/8 missing 1)
#   3. Quick-verify configs GPU 1 đã pass (trim_no_depthcrs, single_efa)
#
# AN TOÀN race condition: KHÔNG đụng configs GPU 1 đang/sẽ chạy
#   - GPU 1 hiện trên: single_drop (4/8)
#   - GPU 1 sẽ làm: single_opacity → single_dcycle → single_shcrs → single_depthcrs
#   - Supplement chỉ làm single_depthcrs (CUỐI queue GPU 1) → GPU 0 xong trước GPU 1 reach
#   - + recovery partial trong GPU 0 group và trim_no_depthcrs/single_efa (GPU 1 đã pass)
#
# Pre-requisite:
#   conda activate corgs
#   cd ~/workspace/representation-3d/duyen/CoR-GS
#   chmod +x scripts/p23_gpu0_supplement.sh
#
# Usage (nohup + disown để không die khi đóng terminal):
#   nohup ./scripts/p23_gpu0_supplement.sh > /tmp/p23_gpu0_sup.log 2>&1 &
#   disown
#
# Cost: 8 single_depthcrs + ~1-3 partial recovery = 9-11 runs × 7min = ~70-80min
# ============================================================

set -e

# ── Verify init = RoMa v1 ──
FERN_PLY="data/nerf_llff_data/fern/3_views/dense/fused.ply"
FERN_V1="data/nerf_llff_data/fern/3_views/dense/fused.ply.romav1"
if ! cmp -s "$FERN_PLY" "$FERN_V1"; then
    echo "❌ fused.ply ≠ fused.ply.romav1"
    exit 1
fi
echo "✓ verified fused.ply = RoMa v1"

PROTOCOL="--eval -r 8 --n_views 3 --random_background --iterations 10000 \
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

CRS_TRIM="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL}"
INDEP_FULL="${M_DROP} ${M_OPACITY} ${M_EFA}"

build_flags() {
    case "$1" in
        base)              echo "" ;;
        trim_no_efa)       echo "${CRS_TRIM} ${M_DROP} ${M_OPACITY}" ;;
        trim_no_drop)      echo "${CRS_TRIM} ${M_OPACITY} ${M_EFA}" ;;
        trim_no_opacity)   echo "${CRS_TRIM} ${M_DROP} ${M_EFA}" ;;
        trim_no_dcycle)    echo "${M_DEPTHCFG} ${M_SHFREEZE} ${M_SHREL} ${INDEP_FULL}" ;;
        trim_no_shcrs)     echo "${M_DEPTHCFG} ${M_DCYCLE} ${INDEP_FULL}" ;;
        trim_no_depthcrs)  echo "${INDEP_FULL}" ;;
        single_efa)        echo "${M_EFA}" ;;
        single_depthcrs)   echo "${CRS_TRIM}" ;;
    esac
}

SCENES=(fern flower fortress horns leaves orchids room trex)
SEED=42

# Order quan trọng:
#   1. single_depthcrs đầu (priority — 8 scenes mới, lấy CUỐI queue GPU 1)
#   2. Recovery partial trong các configs đã end (lướt nhanh nhờ skip-if-exists)
SAFE_CONFIGS=(
    single_depthcrs
    single_shcrs   # v2 — last LOO trong queue GPU 1 trước single_depthcrs
    base
    trim_no_efa
    trim_no_drop
    trim_no_opacity
    trim_no_dcycle
    trim_no_shcrs
    trim_no_depthcrs
    single_efa
)

LOG_ROOT="logs/p23_v1_ablation"
OUT_ROOT="output/p23_v1_ablation"
MASTER_LOG="${LOG_ROOT}/_supplement.log"
mkdir -p "$LOG_ROOT"

run_one() {
    local CFG=$1; local SC=$2
    local FLAGS=$(build_flags "$CFG")
    local LOG_DIR="${LOG_ROOT}/${CFG}"
    local OUT_DIR="${OUT_ROOT}/${CFG}"
    mkdir -p "$LOG_DIR" "$OUT_DIR"

    local OUTDIR=${OUT_DIR}/A3_seed${SEED}_${SC}
    local LOG=${LOG_DIR}/A3_seed${SEED}_${SC}.log

    if [ -f "$LOG" ] && grep -q "Best test PSNR" "$LOG" 2>/dev/null; then
        return 0  # skip silently
    fi

    echo "[$(date +%H:%M:%S)] START [$CFG/$SC] GPU0" | tee -a "$MASTER_LOG"
    CUDA_VISIBLE_DEVICES=0 python -u train.py \
        --source_path data/nerf_llff_data/${SC} \
        -m ${OUTDIR} --seed ${SEED} \
        ${PROTOCOL} ${FLAGS} \
        2>&1 | tee ${LOG} > /dev/null
    echo "[$(date +%H:%M:%S)] DONE  [$CFG/$SC]" | tee -a "$MASTER_LOG"
}

echo "[supplement] start $(date '+%H:%M:%S')" | tee "$MASTER_LOG"
echo "  GPU 0, configs: ${SAFE_CONFIGS[*]}" | tee -a "$MASTER_LOG"
echo "  skip-if-exists ENABLED (silent skip for already-done)" | tee -a "$MASTER_LOG"

n_runs=0
for CFG in "${SAFE_CONFIGS[@]}"; do
    for SC in "${SCENES[@]}"; do
        run_one "$CFG" "$SC"
        n_runs=$((n_runs + 1))
    done
done

echo "[supplement] done $(date '+%H:%M:%S')  ($n_runs iterations checked)" | tee -a "$MASTER_LOG"
echo ""
echo "next: kiểm tra với p23_v1_ablation_analyze.py sau khi GPU 1 cũng xong"
