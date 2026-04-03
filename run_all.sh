#!/bin/bash

export PATH=/home/aidev/miniconda3/envs/corgs/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

mkdir -p logs/colmap
mkdir -p logs/train

echo "=========================================="
echo "STEP 1: Running COLMAP for all scenes"
echo "=========================================="

python -c "
content = open('tools/colmap_llff.py').read()
content = content.replace(
    \"for scene in ['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex']:\",
    \"for scene in ['flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex']:\"
)
open('tools/colmap_llff_remain.py', 'w').write(content)
"

python tools/colmap_llff_remain.py 2>&1 | tee logs/colmap/colmap_all.txt

echo "=========================================="
echo "STEP 2: Checking COLMAP results"
echo "=========================================="

for scene in flower fortress horns leaves orchids room trex; do
    path="data/nerf_llff_data/$scene/3_views/triangulated"
    if [ -f "$path/points3D.bin" ]; then
        echo "OK $scene"
    else
        echo "MISSING $scene"
    fi
done

echo "=========================================="
echo "STEP 3: Training all scenes"
echo "=========================================="

scenes=(flower fortress horns leaves orchids room trex)
for scene in $scenes; do
    echo "--- Training scene: $scene ---"
    rm -rf output/llff/$scene
    bash scripts/run_llff.sh 0 data/nerf_llff_data/$scene output/llff/$scene 2>&1 | tee logs/train/${scene}_log.txt
    echo "--- Done: $scene ---"
done

echo "=========================================="
echo "ALL DONE!"
echo "=========================================="
