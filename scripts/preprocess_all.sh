#!/usr/bin/env bash
# ============================================================
# preprocess_all.sh -- wrapper around preprocess.py for all 8 LLFF scenes.
#
# Splits the 8 scenes across 2 GPUs (4 each) and runs the RoMa v1 dense
# initialization in parallel. Wall-clock ~45-60 seconds on two H100-class
# GPUs.
#
# Prerequisite:
#   conda activate roma_v1
#   chmod +x scripts/preprocess_all.sh
#
# Usage:    ./scripts/preprocess_all.sh
# Outputs:  logs/preprocess_gpu0.log
#           logs/preprocess_gpu1.log
#           data/nerf_llff_data/<scene>/3_views/dense/fused.ply.romav1
# ============================================================

set -e

mkdir -p logs
LOG0=logs/preprocess_gpu0.log
LOG1=logs/preprocess_gpu1.log
> "$LOG0"; > "$LOG1"

SCENES_GPU0="fern flower fortress horns"
SCENES_GPU1="leaves orchids room trex"

echo "[preprocess_all] start $(date '+%H:%M:%S')"
echo "  GPU 0: $SCENES_GPU0  → $LOG0"
echo "  GPU 1: $SCENES_GPU1  → $LOG1"

nohup bash -c "for s in $SCENES_GPU0; do CUDA_VISIBLE_DEVICES=0 SCENE=\$s python scripts/preprocess.py; done" > "$LOG0" 2>&1 &
PID0=$!
nohup bash -c "for s in $SCENES_GPU1; do CUDA_VISIBLE_DEVICES=1 SCENE=\$s python scripts/preprocess.py; done" > "$LOG1" 2>&1 &
PID1=$!

echo "  PID GPU0=$PID0  PID GPU1=$PID1"
echo "  monitor: tail -f $LOG0 $LOG1"

wait $PID0 $PID1
RC0=$?; RC1=$?
echo "[preprocess_all] done $(date '+%H:%M:%S')  RC GPU0=$RC0  GPU1=$RC1"

echo ""
echo "─── Summary ───"
for s in fern flower fortress horns leaves orchids room trex; do
  ply="data/nerf_llff_data/$s/3_views/dense/fused.ply.romav1"
  if [ -f "$ply" ]; then
    sz_kb=$(awk "BEGIN{printf \"%.1f\", $(stat -c%s "$ply")/1024}")
    n_pts=$(awk -v sc="scene=$s " '/scene=/{cur=index($0,sc)>0} cur && /total points:/{gsub(/.*total points: /,""); gsub(/ .*/,""); print; exit}' "$LOG0" "$LOG1")
    printf "  %-10s ✅ %8s KB  N=%s\n" "$s" "$sz_kb" "${n_pts:-?}"
  else
    printf "  %-10s ❌ MISSING\n" "$s"
  fi
done

[ $RC0 -ne 0 ] || [ $RC1 -ne 0 ] && exit 1 || exit 0
