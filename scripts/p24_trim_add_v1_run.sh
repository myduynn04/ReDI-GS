#!/bin/bash
# ============================================================
# [CRSGaussian Phase 24] Re-test 3 Phase 20 TRIM flags on RoMa v1 backbone
# File: scripts/p24_trim_add_v1_run.sh
#
# Mục đích: Phase 20 TRIM verify trên MVS — bỏ informed_crs_init,
# use_crs_pruning, use_r_visible. Phase 22 RoMa v1 backbone CHƯA verify
# riêng 3 flag này (chỉ test full A3-TRIM 8-module = 21.918).
# Test này: thêm từng flag (và stack all 3) vào A3-TRIM trên v1 backbone,
# N=24 paired để xác định Phase 20 TRIM còn valid trên v1 hay không.
#
# 5 cases × 3 seeds × 8 scenes = 120 runs ≈ 7h trên 2 GPU
#
# Cases:
#   c0_base       — A3-TRIM 8-module (re-run base anchor cùng GPU/code state)
#   c1_informed   — base + informed_crs_init (Phase 5 best WG: w_reproj=0.4, w_depth=0.6, w_view=0)
#   c2_crsprune   — base + use_crs_pruning (Phase 4 defaults: T_warmup=1000, tau_crs=0.35)
#   c3_rvisible   — base + use_r_visible (Phase 8a defaults: tolerance=1.05, min_views=2)
#   c4_stack3     — base + cả 3 (test synergy)
#
# Pre-requisite:
#   conda activate corgs
#   cd ~/workspace/representation-3d/duyen/CoR-GS
#   chmod +x scripts/p24_trim_add_v1_run.sh
#
# Usage (tmux + nohup + disown):
#   tmux new -s p24
#   nohup ./scripts/p24_trim_add_v1_run.sh > /tmp/p24_run.log 2>&1 &
#   disown
#   Ctrl-B D  # detach tmux
#
# Monitor:
#   tmux attach -t p24
#   tail -f /tmp/p24_run.log
#   ls logs/p24_trim_add_v1/*/ | wc -l   # đếm số log đã có
#
# Analyze sau khi xong:
#   python scripts/p24_trim_add_v1_analyze.py
# ============================================================

set -e

# ── Verify init = RoMa v1 (chống nhầm với MVS hoặc v2) ──
FERN_PLY="data/nerf_llff_data/fern/3_views/dense/fused.ply"
FERN_V1="data/nerf_llff_data/fern/3_views/dense/fused.ply.romav1"
if ! cmp -s "$FERN_PLY" "$FERN_V1"; then
    echo "❌ fused.ply ≠ fused.ply.romav1 — chạy scripts/p22_place_romav1_init.py trước!"
    exit 1
fi
echo "✓ verified fused.ply = RoMa v1"

# ── Phase 22 protocol (KHÔNG đổi) ──
PROTOCOL="--eval -r 8 --n_views 3 --random_background --iterations 10000 \
--densify_until_iter 5000 --densify_grad_threshold 0.0005 --gaussiansN 1 \
--sample_pseudo_interval 1 --start_sample_pseudo 500 --test_iterations 10000"

# ── A3-TRIM 8-module base recipe (Phase 22 PROJECT BEST = 21.918) ──
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

# ── 3 Phase 20 TRIM modules (đã trim ở Phase 20, test re-add trên v1) ──
# Phase 5 best WG validated: w_reproj=0.4, w_depth=0.6, w_view=0 (no q_view)
M_INFORMED="--informed_crs_init \
--crs_init_use_reproj True --crs_init_use_depth True --crs_init_use_view False \
--crs_init_w_reproj 0.4 --crs_init_w_depth 0.6 --crs_init_w_view 0.0 \
--crs_init_tau_r 2.5 --crs_init_gamma 5.0 --crs_init_eta 0.7 \
--crs_densify_inherit"

# Phase 4 defaults (validated từ T2.7 + T4.2)
M_CRS_PRUNE="--use_crs_pruning --T_warmup 1000 --tau_crs 0.35 --tau_isolated 0.1"

# Phase 8a defaults (validated)
M_R_VISIBLE="--use_r_visible --r_visible_occlusion_tolerance 1.05 --r_visible_min_views 2"

build_flags() {
    case "$1" in
        c0_base)       echo "${A3_TRIM}" ;;
        c1_informed)   echo "${A3_TRIM} ${M_INFORMED}" ;;
        c2_crsprune)   echo "${A3_TRIM} ${M_CRS_PRUNE}" ;;
        c3_rvisible)   echo "${A3_TRIM} ${M_R_VISIBLE}" ;;
        c4_stack3)     echo "${A3_TRIM} ${M_INFORMED} ${M_CRS_PRUNE} ${M_R_VISIBLE}" ;;
        *)             echo "UNKNOWN_CONFIG_$1" ;;
    esac
}

CONFIGS=(c0_base c1_informed c2_crsprune c3_rvisible c4_stack3)
SEEDS=(42 137 9999)

# ── GPU split (8 scenes → 4 + 4) ──
GPU0_SCENES=(fern flower fortress horns)
GPU1_SCENES=(leaves orchids room trex)

LOG_ROOT="logs/p24_trim_add_v1"
OUT_ROOT="output/p24_trim_add_v1"
MASTER_LOG="${LOG_ROOT}/_master.log"
mkdir -p "$LOG_ROOT"

run_one() {
    local GPU=$1; local CFG=$2; local SEED=$3; local SC=$4
    local FLAGS=$(build_flags "$CFG")
    local LOG_DIR="${LOG_ROOT}/${CFG}"
    local OUT_DIR="${OUT_ROOT}/${CFG}"
    mkdir -p "$LOG_DIR" "$OUT_DIR"

    local OUTDIR=${OUT_DIR}/A3_seed${SEED}_${SC}
    local LOG=${LOG_DIR}/A3_seed${SEED}_${SC}.log

    # Skip if already done (resume-safe)
    if [ -f "$LOG" ] && grep -q "Best test PSNR" "$LOG" 2>/dev/null; then
        return 0
    fi

    echo "[$(date +%H:%M:%S)] START [P24 ${CFG} seed${SEED}/${SC}] GPU${GPU}" | tee -a "$MASTER_LOG"
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --source_path data/nerf_llff_data/${SC} \
        -m ${OUTDIR} --seed ${SEED} \
        ${PROTOCOL} ${FLAGS} \
        2>&1 | tee ${LOG} > /dev/null
    echo "[$(date +%H:%M:%S)] DONE  [P24 ${CFG} seed${SEED}/${SC}]" | tee -a "$MASTER_LOG"
}

run_gpu_loop() {
    local GPU=$1; shift
    local SCENES=("$@")
    for CFG in "${CONFIGS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            for SC in "${SCENES[@]}"; do
                run_one "$GPU" "$CFG" "$SEED" "$SC"
            done
        done
    done
}

echo "[P24] start $(date '+%Y-%m-%d %H:%M:%S')" | tee "$MASTER_LOG"
echo "  configs: ${CONFIGS[*]}" | tee -a "$MASTER_LOG"
echo "  seeds: ${SEEDS[*]}" | tee -a "$MASTER_LOG"
echo "  GPU0 scenes: ${GPU0_SCENES[*]}" | tee -a "$MASTER_LOG"
echo "  GPU1 scenes: ${GPU1_SCENES[*]}" | tee -a "$MASTER_LOG"
echo "  total: 5 cfg × 3 seeds × 8 scenes = 120 runs" | tee -a "$MASTER_LOG"
echo "  skip-if-exists ENABLED (silent skip for already-done)" | tee -a "$MASTER_LOG"

# Parallel 2 GPUs
run_gpu_loop 0 "${GPU0_SCENES[@]}" &
PID0=$!
run_gpu_loop 1 "${GPU1_SCENES[@]}" &
PID1=$!

wait $PID0 $PID1

echo "[P24] done $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "next: python scripts/p24_trim_add_v1_analyze.py" | tee -a "$MASTER_LOG"
