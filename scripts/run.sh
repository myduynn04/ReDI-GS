#!/usr/bin/env bash
# ============================================================
# run.sh -- main training driver.
#
# Trains ReDI-GS with the full production recipe (CONFIG=trim_full)
# over 8 LLFF scenes x 3 seeds = 24 paired runs, split across two GPUs.
# Total wall-clock ~5 hours on two recent GPUs.
#
# Prerequisites:
#   1. conda activate redigs
#   2. fused.ply per scene must contain the dense init produced by
#      preprocess_all.sh and placed by place_init.py.
#   3. chmod +x scripts/run.sh
#
# Usage:    ./scripts/run.sh
# Outputs:  logs/run/A3_seed{42,137,9999}_{scene}.log   (24 files)
#           output/run/A3_seed{42,137,9999}_{scene}/    (trained models)
#
# After completion, run ``python scripts/analyze.py`` to aggregate the
# per-run logs into a single PSNR / SSIM / LPIPS table.
# ============================================================

set -e

mkdir -p logs/run output/run
LOG0=logs/run/_pilot_gpu0.log
LOG1=logs/run/_pilot_gpu1.log
> "$LOG0"; > "$LOG1"

# Safety check: verify fused.ply matches the RoMa v1 dense init.
FERN_PLY="data/nerf_llff_data/fern/3_views/dense/fused.ply"
FERN_V1="data/nerf_llff_data/fern/3_views/dense/fused.ply.romav1"
if [ ! -f "$FERN_V1" ]; then
    echo "ERROR: fused.ply.romav1 not found -- run preprocess_all.sh first."
    exit 1
fi
if ! cmp -s "$FERN_PLY" "$FERN_V1"; then
    echo "ERROR: fused.ply differs from fused.ply.romav1 -- run place_init.py first."
    exit 1
fi
echo "verified: fused.ply == fused.ply.romav1"

GPU0_ID=${GPU0_ID:-0}
GPU1_ID=${GPU1_ID:-1}

echo ""
echo "[run] start $(date '+%H:%M:%S')"
echo "  GPU $GPU0_ID: fern flower fortress horns  × 3 seeds → $LOG0"
echo "  GPU $GPU1_ID: leaves orchids room trex    × 3 seeds → $LOG1"

nohup bash -c "
  GPU=$GPU0_ID CONFIG=trim_full \
  SCENES_OVERRIDE='fern flower fortress horns' \
  SEEDS_OVERRIDE='42 137 9999' \
  LOG_DIR=logs/run OUT_DIR=output/run \
  bash scripts/trainer.sh
" > "$LOG0" 2>&1 &
PID0=$!

nohup bash -c "
  GPU=$GPU1_ID CONFIG=trim_full \
  SCENES_OVERRIDE='leaves orchids room trex' \
  SEEDS_OVERRIDE='42 137 9999' \
  LOG_DIR=logs/run OUT_DIR=output/run \
  bash scripts/trainer.sh
" > "$LOG1" 2>&1 &
PID1=$!

echo "  PID GPU$GPU0_ID=$PID0  PID GPU$GPU1_ID=$PID1"
echo "  monitor: tail -f $LOG0 $LOG1"
echo "  count:   watch -n 5 'ls logs/run/A3_seed*.log | wc -l'  # target 24"

wait $PID0 $PID1
RC0=$?; RC1=$?

echo ""
echo "[run] done $(date '+%H:%M:%S')  RC GPU$GPU0_ID=$RC0  GPU$GPU1_ID=$RC1"

N_LOGS=$(ls logs/run/A3_seed*.log 2>/dev/null | wc -l)
echo "  produced $N_LOGS / 24 log files"

if [ "$N_LOGS" -ne 24 ]; then
    echo "  WARNING: missing runs -- check $LOG0 $LOG1"
fi

echo ""
echo "next: python scripts/analyze.py"

[ $RC0 -ne 0 ] || [ $RC1 -ne 0 ] && exit 1 || exit 0
