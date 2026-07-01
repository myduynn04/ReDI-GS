#!/bin/bash
# ============================================================
# [Phase 26] CRS + EFA contribution ablation on RoMa v1 backbone
#
# Mục đích: Đo Δ thuần của CRS framework và EFA-GS
#           trên Phase 22 dense init (RoMa v1) recipe.
#
# Cells (4 cells = 1 paired baseline + 3 ablation):
#   trim_full            = Phase 22 RECIPE — paired baseline (re-run trong same batch)
#   trim_no_crs          = Full - CRS (bỏ D_cycle + SH freeze + SH reliability)
#                          → Giữ: depth supervision + DropAnSH + opacity + EFA + dense init
#   trim_no_efa          = Full - EFA (bỏ LFCF + AbsGS)
#                          → Giữ: depth + CRS + DropAnSH + opacity + dense init
#   trim_no_crs_no_efa   = Full - CRS - EFA
#                          → Giữ: depth supervision + DropAnSH + opacity + dense init
#
# SEED priority: seed 42 chạy TRƯỚC. 2 seed còn lại (137, 9999) chạy sau nếu có thời gian.
#
# Usage (seed 42 only — DEFAULT priority):
#   conda activate corgs
#   cd ~/workspace/representation-3d/duyen/CoR-GS
#   chmod +x scripts/p26_crs_efa_ablation.sh
#   tmux new -s p26
#   nohup bash scripts/p26_crs_efa_ablation.sh > logs/p26_crs_efa/_driver.log 2>&1 &
#   disown
#   Ctrl-B D
#
# Usage (add seeds 137 + 9999 sau):
#   SEEDS_OVERRIDE="137 9999" nohup bash scripts/p26_crs_efa_ablation.sh \
#       > logs/p26_crs_efa/_driver_extra_seeds.log 2>&1 &
#
# Usage (full N=24 1 lệnh):
#   SEEDS_OVERRIDE="42 137 9999" nohup bash scripts/p26_crs_efa_ablation.sh \
#       > logs/p26_crs_efa/_driver_full.log 2>&1 &
#
# Cost:
#   seed 42 only:  4 cells × 1 seed × 8 scenes = 32 runs
#                  5.4 min/run × 4 runs per GPU per config × 4 configs
#                  = ~86 min = ~1.5h parallel 2 GPU
#   full N=24:     4 cells × 3 seeds × 8 scenes = 96 runs
#                  = ~4.3h parallel 2 GPU
#
# Monitor:
#   watch -n 30 'ls logs/p26_crs_efa/*/A3_seed*.log 2>/dev/null | wc -l'
#   tail -f logs/p26_crs_efa/_driver.log
# ============================================================
set -e

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

# ── Phase 22 protocol (identical Phase 22 pilot, KHÔNG đổi gì) ──
PROTOCOL="--eval -r 8 --n_views 3 --random_background --iterations 10000 \
--densify_until_iter 5000 --densify_grad_threshold 0.0005 --gaussiansN 1 \
--sample_pseudo_interval 1 --start_sample_pseudo 500 --test_iterations 10000"

# ── Mechanism flag groups (identical p20 ablation script) ──
M_DEPTHCFG="--use_depth_prior --dav2_path ../Depth-Anything-V2 --crs_ema_decay 0.3 --crs_update_interval 100"
M_DCYCLE="--use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100"
M_SHFREEZE="--use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.5"
M_SHREL="--use_sh_reliability --sh_stability_warmup 1000 --sh_stability_ema_beta 0.95 --crs_w_s 0.33"
M_DROP="--use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2"
M_OPACITY="--use_opacity_decay --opacity_decay_factor 0.999"
M_EFA="--use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0 \
--lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0 \
--lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True --absdensify"

# ── 4 ablation cells ──
declare -A CELLS=(
    [trim_full]="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL} ${M_DROP} ${M_OPACITY} ${M_EFA}"
    [trim_no_crs]="${M_DEPTHCFG} ${M_DROP} ${M_OPACITY} ${M_EFA}"
    [trim_no_efa]="${M_DEPTHCFG} ${M_DCYCLE} ${M_SHFREEZE} ${M_SHREL} ${M_DROP} ${M_OPACITY}"
    [trim_no_crs_no_efa]="${M_DEPTHCFG} ${M_DROP} ${M_OPACITY}"
)

# Config order — chạy baseline trước để có reference sớm
CONFIG_ORDER=(trim_full trim_no_crs trim_no_efa trim_no_crs_no_efa)

# ── Seeds (default seed 42 priority, override via env) ──
if [ -n "${SEEDS_OVERRIDE}" ]; then
    SEEDS=(${SEEDS_OVERRIDE})
else
    SEEDS=(42)  # DEFAULT — priority seed 42 only
fi
echo "✓ Seeds: ${SEEDS[@]}"

# Scene split GPU 0 / GPU 1
GPU0_SCENES=(fern flower fortress horns)
GPU1_SCENES=(leaves orchids room trex)

LOG_ROOT="logs/p26_crs_efa"
OUT_ROOT="output/p26_crs_efa"
mkdir -p "$LOG_ROOT"

# ── Skip logic helper — check if a run completed successfully ──
# Returns 0 if cached + valid (skip), 1 if needs to run
is_cached() {
    local LOG=$1
    local OUTDIR=$2

    # Condition 1: log file phải tồn tại
    [ -f "$LOG" ] || return 1

    # Condition 2: log phải có marker "Best test PSNR" (train xong + eval xong)
    grep -q "Best test PSNR" "$LOG" 2>/dev/null || return 1

    # Condition 3: output dir phải có point_cloud (model checkpoint exists)
    [ -d "${OUTDIR}/point_cloud" ] || return 1

    return 0   # ✓ tất cả pass → đã cached, bỏ qua
}

# Run single experiment (skip if cached + valid)
run_one() {
    local GPU=$1; local CONFIG=$2; local SEED=$3; local SC=$4
    local FLAGS=$5

    local OUTDIR=${OUT_ROOT}/${CONFIG}/A3_seed${SEED}_${SC}
    local LOGDIR=${LOG_ROOT}/${CONFIG}
    local LOG=${LOGDIR}/A3_seed${SEED}_${SC}.log
    mkdir -p "$LOGDIR"

    # Skip if cached + valid
    if is_cached "$LOG" "$OUTDIR"; then
        local PSNR=$(grep "Best test PSNR" "$LOG" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        echo "[$(date +%H:%M:%S)] ⏩ SKIP cached ${CONFIG}/seed${SEED}_${SC} (PSNR=${PSNR})"
        return 0
    fi

    # Nếu log cũ exists nhưng KHÔNG complete (crash/incomplete) → xóa và re-run
    if [ -f "$LOG" ]; then
        echo "[$(date +%H:%M:%S)] ↻ RETRY [${CONFIG}] seed${SEED}_${SC} (log cũ không complete)"
        rm -f "$LOG"
    fi

    echo "[$(date +%H:%M:%S)] ▶ START [${CONFIG}] seed${SEED}_${SC} GPU${GPU}"
    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
        -s data/nerf_llff_data/${SC} -m ${OUTDIR} \
        ${PROTOCOL} ${FLAGS} --seed ${SEED} \
        > "$LOG" 2>&1 || echo "  ⚠ FAIL [${CONFIG}] seed${SEED}_${SC} — check $LOG"

    # Verify completion
    if grep -q "Best test PSNR" "$LOG" 2>/dev/null; then
        local PSNR=$(grep "Best test PSNR" "$LOG" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        echo "[$(date +%H:%M:%S)] ✓ DONE  [${CONFIG}] seed${SEED}_${SC} (PSNR=${PSNR})"
    else
        echo "[$(date +%H:%M:%S)] ❌ FAIL [${CONFIG}] seed${SEED}_${SC} — check $LOG"
    fi
}

# Pre-check — đếm số runs đã cached vs cần chạy
pre_check_cache() {
    local TOTAL=0
    local CACHED=0
    local NEED_RUN=0

    for CONFIG in "${CONFIG_ORDER[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            for SC in "${GPU0_SCENES[@]}" "${GPU1_SCENES[@]}"; do
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
    echo "  Cache status: ${CACHED} / ${TOTAL} already done, ${NEED_RUN} need to run"
    local SAVED_MIN=$((CACHED * 5))   # ~5 min per cached
    echo "  Skipping saves ~${SAVED_MIN} min of compute"
}

# Worker for 1 GPU — loop scenes assigned to it
run_gpu_worker() {
    local GPU=$1; shift
    local CONFIG=$1; shift
    local FLAGS=$1; shift
    local SCENES_LIST=("$@")

    for SEED in "${SEEDS[@]}"; do
        for SC in "${SCENES_LIST[@]}"; do
            run_one "$GPU" "$CONFIG" "$SEED" "$SC" "$FLAGS"
        done
    done
}

# ── MAIN: loop 4 configs, parallel 2 GPU within each ──
echo ""
echo "================================================================"
echo " Phase 26 CRS+EFA ablation — START $(date '+%Y-%m-%d %H:%M:%S')"
echo " Seeds: ${SEEDS[@]}"
echo " Configs: ${CONFIG_ORDER[@]}"
echo " Expected runs: $((${#CONFIG_ORDER[@]} * ${#SEEDS[@]} * 8))"
pre_check_cache
echo "================================================================"

for CONFIG in "${CONFIG_ORDER[@]}"; do
    FLAGS="${CELLS[$CONFIG]}"
    echo ""
    echo "================================================================"
    echo " [$(date '+%H:%M:%S')] CONFIG = ${CONFIG}"
    echo "================================================================"

    # Parallel 2 GPU
    run_gpu_worker 0 "$CONFIG" "$FLAGS" "${GPU0_SCENES[@]}" &
    PID0=$!
    run_gpu_worker 1 "$CONFIG" "$FLAGS" "${GPU1_SCENES[@]}" &
    PID1=$!

    echo "  → PID GPU0=$PID0  PID GPU1=$PID1"
    wait $PID0 $PID1

    N_DONE=$(ls ${LOG_ROOT}/${CONFIG}/A3_seed*.log 2>/dev/null | wc -l)
    EXPECT=$((${#SEEDS[@]} * 8))
    echo "  [${CONFIG}] produced ${N_DONE} / ${EXPECT} logs"
done

echo ""
echo "================================================================"
echo " Phase 26 ALL DONE $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

TOTAL=$(ls ${LOG_ROOT}/*/A3_seed*.log 2>/dev/null | wc -l)
EXPECT_TOTAL=$((${#CONFIG_ORDER[@]} * ${#SEEDS[@]} * 8))
echo " Total logs: ${TOTAL} / ${EXPECT_TOTAL}"
echo ""
echo " Output: ${OUT_ROOT}/<config>/A3_seed<seed>_<scene>/"
echo " Logs:   ${LOG_ROOT}/<config>/A3_seed<seed>_<scene>.log"
echo ""
echo " Quick check PSNR mean per config:"
echo "   for c in trim_full trim_no_crs trim_no_efa trim_no_crs_no_efa; do"
echo "     echo \"=== \$c ===\"; "
echo "     grep 'Best test PSNR' logs/p26_crs_efa/\$c/A3_seed*.log | "
echo "       awk -F': |at' '{sum+=\$2; n++} END {if(n>0) printf \"  mean=%.4f (N=%d)\\n\", sum/n, n}'"
echo "   done"
