#!/usr/bin/env bash
# ============================================================
# [CRSGaussian Phase 22 — Bước 4] Pilot N=24 runner
# File: scripts/p22_pilot_run.sh  (KEEP LOCAL — server-only)
#
# Pre-requisite:
#   1. conda activate corgs
#   2. cd ~/workspace/representation-3d/duyen/CoR-GS
#   3. fused.ply per scene PHẢI = RoMa v1 (chạy p22_place_romav1_init.py trước)
#   4. chmod +x scripts/p22_pilot_run.sh
#
# Usage:
#   ./scripts/p22_pilot_run.sh
#
# Output:
#   logs/p22_pilot/A3_seed{42,137,9999}_<scene>.log  (24 files)
#   output/p22_pilot/A3_seed{42,137,9999}_<scene>/
#
# Design:
#   - A3-TRIM-FULL recipe (reuse p20_ablation_dense_run.sh CONFIG=trim_full)
#   - 8 scenes × 3 seeds = 24 runs paired
#   - 2-GPU split (sửa env GPU_LIST nếu chỉ có 1 GPU available)
# ============================================================

set -e

mkdir -p logs/p22_pilot output/p22_pilot
LOG0=logs/p22_pilot/_pilot_gpu0.log
LOG1=logs/p22_pilot/_pilot_gpu1.log
> "$LOG0"; > "$LOG1"

# ── Safety: verify fused.ply = RoMa v1 ──
FERN_PLY="data/nerf_llff_data/fern/3_views/dense/fused.ply"
FERN_V1="data/nerf_llff_data/fern/3_views/dense/fused.ply.romav1"
if [ ! -f "$FERN_V1" ]; then
    echo "❌ fused.ply.romav1 chưa được sinh — chạy p22_run_all_scenes.sh trước"
    exit 1
fi
if ! cmp -s "$FERN_PLY" "$FERN_V1"; then
    echo "❌ fused.ply ≠ fused.ply.romav1 — chạy p22_place_romav1_init.py trước"
    echo "   (hiện fused.ply có thể là MVS / v2 / hoặc khác)"
    exit 1
fi
echo "✓ verified fused.ply = RoMa v1"

GPU0_ID=${GPU0_ID:-0}
GPU1_ID=${GPU1_ID:-1}

echo ""
echo "[p22 pilot] start $(date '+%H:%M:%S')"
echo "  GPU $GPU0_ID: fern flower fortress horns  × 3 seeds → $LOG0"
echo "  GPU $GPU1_ID: leaves orchids room trex    × 3 seeds → $LOG1"

nohup bash -c "
  GPU=$GPU0_ID CONFIG=trim_full \
  SCENES_OVERRIDE='fern flower fortress horns' \
  SEEDS_OVERRIDE='42 137 9999' \
  LOG_DIR=logs/p22_pilot OUT_DIR=output/p22_pilot \
  bash scripts/p20_ablation_dense_run.sh
" > "$LOG0" 2>&1 &
PID0=$!

nohup bash -c "
  GPU=$GPU1_ID CONFIG=trim_full \
  SCENES_OVERRIDE='leaves orchids room trex' \
  SEEDS_OVERRIDE='42 137 9999' \
  LOG_DIR=logs/p22_pilot OUT_DIR=output/p22_pilot \
  bash scripts/p20_ablation_dense_run.sh
" > "$LOG1" 2>&1 &
PID1=$!

echo "  PID GPU$GPU0_ID=$PID0  PID GPU$GPU1_ID=$PID1"
echo "  monitor: tail -f $LOG0 $LOG1"
echo "  count:   watch -n 5 'ls logs/p22_pilot/A3_seed*.log | wc -l'  # target 24"

wait $PID0 $PID1
RC0=$?; RC1=$?

echo ""
echo "[p22 pilot] done $(date '+%H:%M:%S')  RC GPU$GPU0_ID=$RC0  GPU$GPU1_ID=$RC1"

N_LOGS=$(ls logs/p22_pilot/A3_seed*.log 2>/dev/null | wc -l)
echo "  produced $N_LOGS / 24 log files"

if [ "$N_LOGS" -ne 24 ]; then
    echo "  ⚠ thiếu run — check $LOG0 $LOG1"
fi

echo ""
echo "next: python scripts/p22_pilot_analyze.py"

[ $RC0 -ne 0 ] || [ $RC1 -ne 0 ] && exit 1 || exit 0
