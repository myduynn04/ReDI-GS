#!/usr/bin/env bash
# ============================================================
# [CRSGaussian Phase 21 — Bước 1 runner] RoMa v2 preprocess 8 scenes
# File: scripts/p21_run_all_scenes.sh  (KEEP LOCAL — server-only)
#
# Pre-requisite (server):
#   conda activate romav2
#   cd ~/workspace/representation-3d/duyen/CoR-GS
#   chmod +x scripts/p21_run_all_scenes.sh
#
# Usage:
#   ./scripts/p21_run_all_scenes.sh
#
# Output:
#   logs/p21_gpu0.log, logs/p21_gpu1.log
#   data/nerf_llff_data/<scene>/3_views/dense/fused.ply.roma (8 files)
#
# Split 4 scenes / GPU song song. Mỗi scene ~10s → total ~40-45s wall-time.
# ============================================================

set -e

mkdir -p logs
LOG0=logs/p21_gpu0.log
LOG1=logs/p21_gpu1.log
> "$LOG0"  # truncate
> "$LOG1"

SCENES_GPU0="fern flower fortress horns"
SCENES_GPU1="leaves orchids room trex"

echo "[p21 runner] start $(date '+%H:%M:%S')"
echo "  GPU 0: $SCENES_GPU0  → $LOG0"
echo "  GPU 1: $SCENES_GPU1  → $LOG1"

nohup bash -c "for s in $SCENES_GPU0; do CUDA_VISIBLE_DEVICES=0 SCENE=\$s python scripts/p21_roma_preprocess.py; done" > "$LOG0" 2>&1 &
PID0=$!
nohup bash -c "for s in $SCENES_GPU1; do CUDA_VISIBLE_DEVICES=1 SCENE=\$s python scripts/p21_roma_preprocess.py; done" > "$LOG1" 2>&1 &
PID1=$!

echo "  PID GPU0=$PID0  PID GPU1=$PID1"
echo "  monitor: tail -f $LOG0 $LOG1"

wait $PID0 $PID1
RC0=$?
RC1=$?

echo "[p21 runner] done $(date '+%H:%M:%S')  RC GPU0=$RC0  GPU1=$RC1"

# Quick summary — file size + N points per scene
# (Parse log block-by-block: "scene=<s>" ... "total points: N")
echo ""
echo "─── Summary ───"
for s in fern flower fortress horns leaves orchids room trex; do
  ply="data/nerf_llff_data/$s/3_views/dense/fused.ply.roma"
  if [ -f "$ply" ]; then
    sz_kb=$(awk "BEGIN{printf \"%.1f\", $(stat -c%s "$ply")/1024}")
    n_pts=$(awk -v sc="scene=$s " '/scene=/{cur=index($0,sc)>0} cur && /total points:/{gsub(/.*total points: /,""); gsub(/ .*/,""); print; exit}' "$LOG0" "$LOG1")
    printf "  %-10s ✅ %8s KB  N=%s\n" "$s" "$sz_kb" "${n_pts:-?}"
  else
    printf "  %-10s ❌ MISSING\n" "$s"
  fi
done

[ $RC0 -ne 0 ] || [ $RC1 -ne 0 ] && exit 1 || exit 0
