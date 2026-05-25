#!/usr/bin/env bash
# ============================================================
# [CRSGaussian Phase 21 — Bước 4] Pilot N=24 runner
# File: scripts/p21_pilot_run.sh  (KEEP LOCAL — server-only)
#
# Pre-requisite (server):
#   1. conda activate corgs
#   2. cd ~/workspace/representation-3d/duyen/CoR-GS
#   3. fused.ply per scene PHẢI = RoMa (chạy p21_place_roma_init.py trước)
#   4. chmod +x scripts/p21_pilot_run.sh
#
# Usage:
#   ./scripts/p21_pilot_run.sh
#
# Output:
#   logs/p21_pilot/A3_seed{42,137,9999}_<scene>.log  (24 files)
#   output/p21_pilot/A3_seed{42,137,9999}_<scene>/   (24 dirs, có checkpoint)
#
# Design:
#   - A3-TRIM-FULL recipe (reuse p20_ablation_dense_run.sh CONFIG=trim_full)
#   - 8 scenes × 3 seeds = 24 runs paired
#   - 2-GPU split: GPU0 = {fern,flower,fortress,horns} × 3 seeds = 12 runs
#                  GPU1 = {leaves,orchids,room,trex}    × 3 seeds = 12 runs
#   - Wall-time estimate: 24 × 5min / 2 GPU = ~60 min
#
# Sau xong:
#   python scripts/p21_pilot_analyze.py
# ============================================================

set -e

mkdir -p logs/p21_pilot output/p21_pilot
LOG0=logs/p21_pilot/_pilot_gpu0.log
LOG1=logs/p21_pilot/_pilot_gpu1.log
> "$LOG0"
> "$LOG1"

# ── Safety: verify fused.ply là RoMa ─────────────────────────
# RoMa ply size ~400-700 KB; COLMAP-MVS thường khác. Check fern bbox.
FERN_PLY="data/nerf_llff_data/fern/3_views/dense/fused.ply"
FERN_ROMA="data/nerf_llff_data/fern/3_views/dense/fused.ply.roma"
if [ ! -f "$FERN_ROMA" ]; then
    echo "❌ fused.ply.roma chưa được sinh — chạy p21_run_all_scenes.sh trước"
    exit 1
fi
if ! cmp -s "$FERN_PLY" "$FERN_ROMA"; then
    echo "❌ fused.ply ≠ fused.ply.roma — chạy p21_place_roma_init.py trước"
    echo "   (hiện fused.ply có thể vẫn là COLMAP-MVS hoặc PDCNet+ cũ)"
    exit 1
fi
echo "✓ verified fused.ply = RoMa"

echo ""
echo "[p21 pilot] start $(date '+%H:%M:%S')"
echo "  GPU 0: fern flower fortress horns  × 3 seeds → $LOG0"
echo "  GPU 1: leaves orchids room trex    × 3 seeds → $LOG1"

# GPU 0 — 4 scenes × 3 seeds
nohup bash -c "
  GPU=0 CONFIG=trim_full \
  SCENES_OVERRIDE='fern flower fortress horns' \
  SEEDS_OVERRIDE='42 137 9999' \
  LOG_DIR=logs/p21_pilot OUT_DIR=output/p21_pilot \
  bash scripts/p20_ablation_dense_run.sh
" > "$LOG0" 2>&1 &
PID0=$!

# GPU 1 — 4 scenes × 3 seeds
nohup bash -c "
  GPU=1 CONFIG=trim_full \
  SCENES_OVERRIDE='leaves orchids room trex' \
  SEEDS_OVERRIDE='42 137 9999' \
  LOG_DIR=logs/p21_pilot OUT_DIR=output/p21_pilot \
  bash scripts/p20_ablation_dense_run.sh
" > "$LOG1" 2>&1 &
PID1=$!

echo "  PID GPU0=$PID0  PID GPU1=$PID1"
echo "  monitor: tail -f $LOG0 $LOG1"
echo "  hoặc:    watch -n 5 'ls logs/p21_pilot/A3_*.log | wc -l'  # đếm runs xong (target=24)"

wait $PID0 $PID1
RC0=$?
RC1=$?

echo ""
echo "[p21 pilot] done $(date '+%H:%M:%S')  RC GPU0=$RC0  GPU1=$RC1"

# Quick count
N_LOGS=$(ls logs/p21_pilot/A3_seed*.log 2>/dev/null | wc -l)
echo "  produced $N_LOGS / 24 log files"

if [ "$N_LOGS" -ne 24 ]; then
    echo "  ⚠ thiếu run — check $LOG0 $LOG1"
fi

echo ""
echo "next: python scripts/p21_pilot_analyze.py"

[ $RC0 -ne 0 ] || [ $RC1 -ne 0 ] && exit 1 || exit 0
