#!/bin/bash
# ============================================================
# [Phase 27] Comprehensive pillar ablation on RoMa v1 backbone
#
# Mục đích: Tạo data cho 2 framing trong thesis:
#   FRAMING A — 5 pillars riêng: L_depth | CRS | DropAnSH | Opacity | EFA-GS
#   FRAMING B — 4 pillars (gộp L+CRS): "Depth-guided reg" | DropAnSH | Opacity | EFA-GS
#
# Output cho 3 bảng thesis:
#   Bảng 1 — Cumulative build-up (baseline → +L → +CRS → +Drop → +Opa → +EFA)
#   Bảng 2 — Leave-one-out (full − each pillar)
#   Bảng 3 — EFA-GS internal split (LFCF vs AbsGS)
#
# CELLS (12 distinct):
#
# Cumulative (5 cells, ngoài trim_full):
#   cum_0_base            base 3DGS + RoMa v1 (no modules)
#   cum_1_ldepth          + L_depth supervision
#   cum_2_crs             + L_depth + CRS framework
#   cum_3_drop            + L_depth + CRS + DropAnSH
#   cum_4_opa             + L_depth + CRS + DropAnSH + Opacity (= LOO no_efa)
#   (trim_full = cum_5    + L_depth + CRS + DropAnSH + Opacity + EFA = FULL)
#
# LOO (5 cells, ngoài cum_4_opa và trim_full):
#   trim_full             FULL recipe = Phase 22
#   loo_no_ldepth         FULL − L_depth (via --lambda_depth 0)
#   loo_no_crs            FULL − CRS framework (keep L_depth)
#   loo_no_ldepth_crs     FULL − L_depth − CRS combined (for 4-pillar framing)
#   loo_no_drop           FULL − DropAnSH
#   loo_no_opa            FULL − Opacity decay
#   (loo_no_efa = cum_4_opa)
#
# EFA-GS internal split (2 cells):
#   loo_no_lfcf           FULL − LFCF (keep AbsGS)
#   loo_no_absgs          FULL − AbsGS (keep LFCF)
#
# TOTAL: 12 distinct cells (cum_0..4 + trim_full + 5 LOO + 2 EFA split)
#
# COST (DEFAULT FULL N=24):
#   13 cells × 8 scenes × 3 seeds = 312 runs
#   ~5.4 min/run × ~156 runs per GPU = ~14h parallel 2 GPU
#
# IMPORTANT: Phase 26 cached results KHÔNG reuse — all 13 cells run in same batch
# để paired comparison reliable (same hardware + same code state + same time period).
#
# Usage (DEFAULT — full N=24 1 lệnh ~14h overnight):
#   conda activate corgs
#   cd ~/workspace/representation-3d/duyen/CoR-GS
#   chmod +x scripts/p27_pillar_ablation.sh
#   mkdir -p logs/p27_pillar
#   tmux new -s p27
#   nohup bash scripts/p27_pillar_ablation.sh > logs/p27_pillar/_driver.log 2>&1 &
#   disown
#   Ctrl-B D
#
# Usage (chỉ 1 seed nếu muốn test trước):
#   SEEDS_OVERRIDE="42" nohup bash scripts/p27_pillar_ablation.sh \
#       > logs/p27_pillar/_driver_seed42.log 2>&1 &
#
# Monitor:
#   watch -n 30 'ls logs/p27_pillar/*/A3_seed*.log 2>/dev/null | wc -l'
#   tail -f logs/p27_pillar/_driver.log
# ============================================================
# NOTE: Không dùng set -e vì script này có nhiều || true patterns
# và rmdir nonexistent dir return 1 → set -e sẽ kill silent.
# Thay vào đó, từng critical step có error check riêng.
set +e

# ── Verify RoMa v1 init present ──
FERN_PLY="data/nerf_llff_data/fern/3_views/dense/fused.ply"
FERN_V1="data/nerf_llff_data/fern/3_views/dense/fused.ply.romav1"
if [ ! -f "$FERN_V1" ]; then
    echo "❌ $FERN_V1 chưa được sinh — chạy p22_run_all_scenes.sh trước"
    exit 1
fi
if ! cmp -s "$FERN_PLY" "$FERN_V1"; then
    echo "❌ fused.ply ≠ fused.ply.romav1 — chạy p22_place_romav1_init.py trước"
    exit 1
fi
echo "✓ verified RoMa v1 init"

# ── Phase 22 protocol (identical Phase 22 pilot) ──
PROTOCOL="--eval -r 8 --n_views 3 --random_background --iterations 10000 \
--densify_until_iter 5000 --densify_grad_threshold 0.0005 --gaussiansN 1 \
--sample_pseudo_interval 1 --start_sample_pseudo 500 --test_iterations 10000"

# ── Mechanism flag groups ──
M_DEPTHCFG="--use_depth_prior --dav2_path ../Depth-Anything-V2 --crs_ema_decay 0.3 --crs_update_interval 100"
M_DEPTHCFG_LAMBDA0="--use_depth_prior --dav2_path ../Depth-Anything-V2 --crs_ema_decay 0.3 --crs_update_interval 100 --lambda_depth 0"
M_DCYCLE="--use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100"
M_SHFREEZE="--use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.5"
M_SHREL="--use_sh_reliability --sh_stability_warmup 1000 --sh_stability_ema_beta 0.95 --crs_w_s 0.33"
M_DROP="--use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2"
M_OPACITY="--use_opacity_decay --opacity_decay_factor 0.999"
M_EFA="--use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0 \
--lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0 \
--lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True --absdensify"
M_EFA_ABSONLY="--absdensify"
M_EFA_LFCFONLY="--use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0 \
--lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0 \
--lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True"

# ── 12 ablation cells ──
declare -A CELLS=(
    # Cumulative build-up
    [cum_0_base]=""
    [cum_1_ldepth]="${M_DEPTHCFG}"
    [cum_2_crs]="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL}"
    [cum_3_drop]="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL} ${M_DROP}"
    [cum_4_opa]="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL} ${M_DROP} ${M_OPACITY}"

    # Full recipe (Phase 22)
    [trim_full]="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL} ${M_DROP} ${M_OPACITY} ${M_EFA}"

    # LOO (5-pillar framing)
    [loo_no_ldepth]="${M_DEPTHCFG_LAMBDA0} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL} ${M_DROP} ${M_OPACITY} ${M_EFA}"
    [loo_no_crs]="${M_DEPTHCFG} ${M_DROP} ${M_OPACITY} ${M_EFA}"
    [loo_no_drop]="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL} ${M_OPACITY} ${M_EFA}"
    [loo_no_opa]="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL} ${M_DROP} ${M_EFA}"

    # LOO (4-pillar framing — L+CRS combined)
    [loo_no_ldepth_crs]="${M_DROP} ${M_OPACITY} ${M_EFA}"

    # EFA-GS internal split
    [loo_no_lfcf]="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL} ${M_DROP} ${M_OPACITY} ${M_EFA_ABSONLY}"
    [loo_no_absgs]="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL} ${M_DROP} ${M_OPACITY} ${M_EFA_LFCFONLY}"
)

# Run order: cumulative first (build story), then LOO, then EFA split
CONFIG_ORDER=(
    trim_full
    cum_0_base
    cum_1_ldepth
    cum_2_crs
    cum_3_drop
    cum_4_opa
    loo_no_ldepth
    loo_no_crs
    loo_no_drop
    loo_no_opa
    loo_no_ldepth_crs
    loo_no_lfcf
    loo_no_absgs
)

# ── Seeds — DEFAULT FULL N=24 (3 seeds × 8 scenes), override qua env nếu cần ──
if [ -n "${SEEDS_OVERRIDE}" ]; then
    SEEDS=(${SEEDS_OVERRIDE})
else
    SEEDS=(42 137 9999)  # DEFAULT — FULL N=24 in single batch for paired reliability
fi
echo "✓ Seeds: ${SEEDS[@]}"
echo "✓ Cells (${#CONFIG_ORDER[@]}): ${CONFIG_ORDER[@]}"

# All 8 scenes — dynamic job pool sẽ phân phối tự động giữa 2 GPU
ALL_SCENES=(fern flower fortress horns leaves orchids room trex)

LOG_ROOT="logs/p27_pillar"
OUT_ROOT="output/p27_pillar"
mkdir -p "$LOG_ROOT"

# ── Skip logic helper ──
is_cached() {
    local LOG=$1
    local OUTDIR=$2
    [ -f "$LOG" ] || return 1
    grep -q "Best test PSNR" "$LOG" 2>/dev/null || return 1
    [ -d "${OUTDIR}/point_cloud" ] || return 1
    return 0
}

# Run single experiment (skip if cached + valid)
run_one() {
    local GPU=$1; local CONFIG=$2; local SEED=$3; local SC=$4
    local FLAGS=$5

    local OUTDIR=${OUT_ROOT}/${CONFIG}/A3_seed${SEED}_${SC}
    local LOGDIR=${LOG_ROOT}/${CONFIG}
    local LOG=${LOGDIR}/A3_seed${SEED}_${SC}.log
    mkdir -p "$LOGDIR"

    if is_cached "$LOG" "$OUTDIR"; then
        local PSNR=$(grep "Best test PSNR" "$LOG" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        echo "[$(date +%H:%M:%S)] ⏩ SKIP cached ${CONFIG}/seed${SEED}_${SC} (PSNR=${PSNR})"
        return 0
    fi

    if [ -f "$LOG" ]; then
        echo "[$(date +%H:%M:%S)] ↻ RETRY [${CONFIG}] seed${SEED}_${SC} (log không complete)"
        rm -f "$LOG"
    fi

    echo "[$(date +%H:%M:%S)] ▶ START [${CONFIG}] seed${SEED}_${SC} GPU${GPU}"
    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
        -s data/nerf_llff_data/${SC} -m ${OUTDIR} \
        ${PROTOCOL} ${FLAGS} --seed ${SEED} \
        > "$LOG" 2>&1 || echo "  ⚠ FAIL [${CONFIG}] seed${SEED}_${SC} — check $LOG"

    if grep -q "Best test PSNR" "$LOG" 2>/dev/null; then
        local PSNR=$(grep "Best test PSNR" "$LOG" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        echo "[$(date +%H:%M:%S)] ✓ DONE  [${CONFIG}] seed${SEED}_${SC} (PSNR=${PSNR})"
    else
        echo "[$(date +%H:%M:%S)] ❌ FAIL [${CONFIG}] seed${SEED}_${SC} — check $LOG"
    fi
}

# Pre-check cache
pre_check_cache() {
    local TOTAL=0 CACHED=0 NEED_RUN=0
    for CONFIG in "${CONFIG_ORDER[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            for SC in "${ALL_SCENES[@]}"; do
                TOTAL=$((TOTAL + 1))
                local LOG="${LOG_ROOT}/${CONFIG}/A3_seed${SEED}_${SC}.log"
                local OUTDIR="${OUT_ROOT}/${CONFIG}/A3_seed${SEED}_${SC}"
                if is_cached "$LOG" "$OUTDIR"; then
                    CACHED=$((CACHED + 1))
                else
                    NEED_RUN=$((NEED_RUN + 1))
                fi
            done
        done
    done
    echo ""
    echo "  Cache status: ${CACHED} / ${TOTAL} done, ${NEED_RUN} need to run"
    local SAVED_MIN=$((CACHED * 5))
    echo "  Skipping saves ~${SAVED_MIN} min compute"
}

# ============================================================
# Dynamic job pool — atomic claim via mkdir lock
#   - GPU 0 picks from FRONT of job list
#   - GPU 1 picks from BACK of job list
#   - When front > back, both meet in middle → batch done
#   - Eliminates idle wait between configs (Phase 26 problem)
# ============================================================

# Atomic claim: returns job index or -1 if no more jobs
claim_job() {
    local FROM=$1       # "front" or "back"
    local SEED=$2
    local LOCK_DIR=/tmp/p27_lock_${SEED}_$$
    local IDX="-1"

    # Spin-wait on mkdir lock (POSIX atomic)
    while ! mkdir "$LOCK_DIR" 2>/dev/null; do
        sleep 0.02
    done

    local FRONT=$(cat /tmp/p27_front_${SEED}_$$)
    local BACK=$(cat /tmp/p27_back_${SEED}_$$)

    if [ "$FRONT" -gt "$BACK" ]; then
        IDX="-1"
    elif [ "$FROM" = "front" ]; then
        IDX=$FRONT
        echo $((FRONT + 1)) > /tmp/p27_front_${SEED}_$$
    else
        IDX=$BACK
        echo $((BACK - 1)) > /tmp/p27_back_${SEED}_$$
    fi

    rmdir "$LOCK_DIR"
    echo $IDX
}

# Worker — claim jobs until pool empty
gpu_pool_worker() {
    local GPU=$1
    local FROM=$2       # "front" or "back"
    local SEED=$3
    local JOB_FILE=$4

    while true; do
        local IDX
        IDX=$(claim_job "$FROM" "$SEED")
        if [ -z "$IDX" ] || [ "$IDX" = "-1" ]; then
            echo "[$(date +%H:%M:%S)] [GPU${GPU}] No more jobs in seed${SEED} batch, exiting"
            return 0
        fi

        # Read job at line IDX+1
        local LINE=$(sed -n "$((IDX+1))p" "$JOB_FILE")
        local CONFIG=$(echo "$LINE" | cut -d'|' -f1)
        local SC=$(echo "$LINE" | cut -d'|' -f2)
        local FLAGS="${CELLS[$CONFIG]}"

        run_one "$GPU" "$CONFIG" "$SEED" "$SC" "$FLAGS"
    done
}

# ── MAIN: process seeds sequentially (seed 42 first, then 137, then 9999) ──
echo ""
echo "================================================================"
echo " Phase 27 PILLAR ablation — START $(date '+%Y-%m-%d %H:%M:%S')"
echo " Seeds: ${SEEDS[@]}"
echo " Configs (${#CONFIG_ORDER[@]}): ${CONFIG_ORDER[@]}"
echo " Expected runs: $((${#CONFIG_ORDER[@]} * ${#SEEDS[@]} * 8))"
echo " Strategy: SEED-PRIORITY + dynamic job pool (no idle wait)"
pre_check_cache
echo "================================================================"

# Cleanup any stale locks from previous run
rm -rf /tmp/p27_lock_* /tmp/p27_front_* /tmp/p27_back_* /tmp/p27_jobs_* 2>/dev/null

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "================================================================"
    echo " [$(date '+%H:%M:%S')] === SEED ${SEED} BATCH START ==="
    echo "================================================================"

    # Build job list for this seed: 13 configs × 8 scenes = 104 jobs
    JOB_FILE=/tmp/p27_jobs_seed${SEED}_$$.txt
    > "$JOB_FILE"
    for CONFIG in "${CONFIG_ORDER[@]}"; do
        for SC in "${ALL_SCENES[@]}"; do
            echo "${CONFIG}|${SC}" >> "$JOB_FILE"
        done
    done
    TOTAL_JOBS=$(wc -l < "$JOB_FILE")
    echo "  Job pool: ${TOTAL_JOBS} jobs (${#CONFIG_ORDER[@]} configs × 8 scenes)"

    # Init counters
    echo 0 > /tmp/p27_front_${SEED}_$$
    echo $((TOTAL_JOBS - 1)) > /tmp/p27_back_${SEED}_$$
    # Cleanup stale lock (rm -rf safe with nonexistent)
    rm -rf /tmp/p27_lock_${SEED}_$$ 2>/dev/null

    # Launch 2 GPU workers — claim from opposite ends
    gpu_pool_worker 0 front "$SEED" "$JOB_FILE" &
    PID0=$!
    gpu_pool_worker 1 back "$SEED" "$JOB_FILE" &
    PID1=$!

    echo "  → PID GPU0=$PID0 (claim front)  PID GPU1=$PID1 (claim back)"
    wait $PID0 $PID1

    # Cleanup seed-specific files
    rm -f "$JOB_FILE" /tmp/p27_front_${SEED}_$$ /tmp/p27_back_${SEED}_$$

    # Count completion for this seed
    N_DONE_SEED=$(ls ${LOG_ROOT}/*/A3_seed${SEED}_*.log 2>/dev/null | wc -l)
    EXPECT_SEED=$((${#CONFIG_ORDER[@]} * 8))
    echo "  [SEED ${SEED}] produced ${N_DONE_SEED} / ${EXPECT_SEED} logs"
    echo " [$(date '+%H:%M:%S')] === SEED ${SEED} BATCH DONE ==="
done

echo ""
echo "================================================================"
echo " Phase 27 ALL DONE $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

TOTAL=$(ls ${LOG_ROOT}/*/A3_seed*.log 2>/dev/null | wc -l)
EXPECT_TOTAL=$((${#CONFIG_ORDER[@]} * ${#SEEDS[@]} * 8))
echo " Total logs: ${TOTAL} / ${EXPECT_TOTAL}"
echo ""
echo " Output: ${OUT_ROOT}/<config>/A3_seed<seed>_<scene>/"
echo " Logs:   ${LOG_ROOT}/<config>/A3_seed<seed>_<scene>.log"
echo ""
echo " Quick check PSNR mean per config:"
echo "   python3 -c \""
echo "import re; from pathlib import Path"
echo "R=Path('logs/p27_pillar')"
echo "for c in ${CONFIG_ORDER[@]}:"
echo "    p=[float(re.search(r'PSNR:\\s*([0-9.]+)', l.read_text()).group(1)) for l in sorted((R/c).glob('A3_seed*.log')) if re.search(r'PSNR:\\s*([0-9.]+)', l.read_text())]"
echo "    print(f'{c:<22} {sum(p)/len(p):.4f} (N={len(p)})' if p else f'{c:<22} no data')"
echo "   \""
