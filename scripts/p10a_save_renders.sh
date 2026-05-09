#!/bin/bash
# ============================================================
# [CRSGaussian Phase 10A] Save test renders for paper figures.
#
# Render test views (skip_train) cho 8 scenes × 3 configs:
#   - P8 FULL (current best CRS, từ Phase 8)
#   - Phase 10A AUGMENT
#   - Phase 10A REPLACE
# = 24 sets × ~5 test views = ~120 PNG, ~700 MB.
#
# Run AFTER scripts/p10a_master.sh hoàn thành 16 runs.
# Output: output/<exp>/<config>_<scene>/test/ours_10000/{renders,gt}/
# ============================================================

set -eo pipefail

SCENES=${SCENES_OVERRIDE:-"fern flower fortress horns leaves orchids room trex"}

for SCENE in $SCENES; do
    # Phase 8 FULL (current best CRS)
    if [ -d "output/p8/FULL_${SCENE}" ]; then
        python render.py -m output/p8/FULL_${SCENE} --skip_train
    else
        echo "[skip] output/p8/FULL_${SCENE} không tồn tại"
    fi

    # Phase 10A AUGMENT
    if [ -d "output/p10a/AUGMENT_${SCENE}" ]; then
        python render.py -m output/p10a/AUGMENT_${SCENE} --skip_train
    fi

    # Phase 10A REPLACE
    if [ -d "output/p10a/REPLACE_${SCENE}" ]; then
        python render.py -m output/p10a/REPLACE_${SCENE} --skip_train
    fi
done

echo ""
echo "Renders saved tại: output/<exp>/<config>_<scene>/test/ours_10000/{renders,gt}/"
echo ""
echo "Disk usage:"
du -sh output/p8 output/p10a 2>/dev/null || true
