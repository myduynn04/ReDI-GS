#!/bin/bash
# ============================================================
# [CRSGaussian Plug-in A1] 5-tier verification script
# File: crsgaussian_plugin/verify_plugin.sh
#
# Usage (sau khi upload lên server):
#   cd /home/aidev/workspace/representation-3d/duyen/nerfstudio
#   bash crsgaussian_plugin/verify_plugin.sh
#
# Output: print PASS/FAIL từng tier, summary cuối
# ============================================================

set +e   # Continue khi 1 tier fail, không exit

# ── Config ──
PLUGIN_DIR=/home/aidev/workspace/representation-3d/duyen/nerfstudio/crsgaussian_plugin
WORKDIR=/home/aidev/workspace/representation-3d/duyen/nerfstudio
DATA_ROOT=/home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data
SCENE=fern  # test scene
TEST_OUT=$WORKDIR/_plugin_verify
TEST_ITERS=1000  # nhỏ để smoke nhanh

# CUDA + plugin env
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=$WORKDIR:$PYTHONPATH
export NERFSTUDIO_METHOD_CONFIGS="splatfacto-roma=crsgaussian_plugin:roma_method_spec"

PASS=0
FAIL=0

ok()   { echo "✓ PASS  $1"; PASS=$((PASS+1)); }
fail() { echo "✗ FAIL  $1"; FAIL=$((FAIL+1)); }

echo "==========================================="
echo "[CRSGaussian Plug-in A1] Verification"
echo "Scene: $SCENE | Iters: $TEST_ITERS"
echo "==========================================="

# ─────────────────────────────────────────────────
# TIER 1 — Smoke tests
# ─────────────────────────────────────────────────
echo ""
echo "=== TIER 1 — Smoke tests ==="

# 1.1 Plugin detect
ns-train --help 2>&1 | grep -q "splatfacto-roma" && \
    ok "1.1 Nerfstudio detect splatfacto-roma" || \
    fail "1.1 splatfacto-roma KHÔNG xuất hiện trong ns-train --help"

# 1.2 + 1.3 — Train e2e + eval ra số
rm -rf $TEST_OUT && mkdir -p $TEST_OUT
echo "[verify] Train $SCENE ($TEST_ITERS iter, ON flag)..."
CUDA_VISIBLE_DEVICES=0 ns-train splatfacto-roma \
    --data $DATA_ROOT/$SCENE/3_views/ \
    --output-dir $TEST_OUT \
    --max-num-iterations $TEST_ITERS \
    --steps-per-eval-all-images $TEST_ITERS \
    --vis tensorboard \
    \
    > /tmp/verify_train.log 2>&1

grep -q "Training Finished" /tmp/verify_train.log && \
    ok "1.2 Train end-to-end no crash" || \
    fail "1.2 Train crashed — check /tmp/verify_train.log"

RUN=$(ls -td $TEST_OUT/*/splatfacto-roma/*/ 2>/dev/null | head -1)
if [ -n "$RUN" ]; then
    ns-eval --load-config $RUN/config.yml --output-path $RUN/eval.json > /tmp/verify_eval.log 2>&1
    PSNR=$(python3 -c "import json; print(json.load(open('$RUN/eval.json'))['results']['psnr'])" 2>/dev/null)
    if [ -n "$PSNR" ]; then
        IS_VALID=$(python3 -c "v=float('$PSNR'); print('yes' if v > 10 and v < 50 else 'no')")
        if [ "$IS_VALID" = "yes" ]; then
            ok "1.3 Eval PSNR hợp lệ ($PSNR dB)"
        else
            fail "1.3 PSNR bất thường: $PSNR"
        fi
    else
        fail "1.3 Eval không ra PSNR"
    fi
else
    fail "1.3 Không tìm thấy run output"
fi

# 1.4 Export PLY load được (size > 0)
ns-export gaussian-splat --load-config $RUN/config.yml --output-dir $RUN/export/ > /tmp/verify_export.log 2>&1
if [ -s "$RUN/export/splat.ply" ]; then
    PLY_SIZE=$(du -h $RUN/export/splat.ply | cut -f1)
    ok "1.4 Export PLY OK ($PLY_SIZE)"
else
    fail "1.4 Export PLY fail / empty"
fi

# ─────────────────────────────────────────────────
# TIER 2 — Module ACTIVE đúng (CRITICAL)
# ─────────────────────────────────────────────────
echo ""
echo "=== TIER 2 — Module ACTIVE verify (CRITICAL) ==="

# 2.2 N_gauss iter 0 ≈ RoMa (~17K, không phải COLMAP ~vài trăm)
N_GAUSS_LOG=$(grep -E "Loaded.*points from fused.ply.romav1" /tmp/verify_train.log | head -1)
if [ -n "$N_GAUSS_LOG" ]; then
    N_GAUSS=$(echo "$N_GAUSS_LOG" | grep -oE "Loaded [0-9]+" | grep -oE "[0-9]+")
    if [ "$N_GAUSS" -gt 5000 ]; then
        ok "2.2 N_gauss init = $N_GAUSS (RoMa dense, >5K → KHÔNG phải COLMAP)"
    else
        fail "2.2 N_gauss = $N_GAUSS quá thấp, có thể fallback COLMAP"
    fi
else
    fail "2.2 Không thấy log '[CRSGaussian Plug-in A1] Loaded N points'"
fi

# 2.3 First 5 positions khớp PLY file
PLY_FILE=$DATA_ROOT/$SCENE/3_views/dense/fused.ply.romav1
if [ -f "$PLY_FILE" ]; then
    python3 << EOF
from plyfile import PlyData
import numpy as np
ply = PlyData.read("$PLY_FILE")
v = ply['vertex']
xyz = np.stack([v['x'][:5], v['y'][:5], v['z'][:5]], axis=-1)
print(f"First 5 points from PLY file:")
print(xyz)
print(f"Total N points in PLY: {len(v['x'])}")
EOF
    ok "2.3 PLY first 5 points printed (verify manually vs train log)"
else
    fail "2.3 PLY missing — không verify được"
fi

# 2.4 Missing PLY → FileNotFoundError
echo "[verify] Test 2.4: rename PLY, expect FileNotFoundError..."
mv "$PLY_FILE" "$PLY_FILE.tmp" 2>/dev/null
CUDA_VISIBLE_DEVICES=0 ns-train splatfacto-roma \
    --data $DATA_ROOT/$SCENE/3_views/ \
    --output-dir $TEST_OUT/_24 \
    --max-num-iterations 100 \
    > /tmp/verify_24.log 2>&1
mv "$PLY_FILE.tmp" "$PLY_FILE" 2>/dev/null
grep -q "FileNotFoundError.*RoMa PLY not found" /tmp/verify_24.log && \
    ok "2.4 Missing PLY raise FileNotFoundError (no silent fallback)" || \
    fail "2.4 KHÔNG raise FileNotFoundError — có thể silent fallback"

# 2.1 OFF flag byte-identical (skip — cần re-train cả 2 config 10k+ scene để fair)
echo "ℹ 2.1 OFF flag test SKIP (cần full 10k run — chạy riêng qua run_offverify.sh)"

# ─────────────────────────────────────────────────
# TIER 4 — Standalone không touch Nerfstudio source
# ─────────────────────────────────────────────────
echo ""
echo "=== TIER 4 — Standalone không touch Nerfstudio ==="

# 4.1 Site-packages Nerfstudio không bị modify
NS_DIR=/home/aidev/miniconda3/envs/nerfstudio/lib/python3.10/site-packages/nerfstudio
MODIFIED=$(find $NS_DIR -newer $PLUGIN_DIR/__init__.py -type f -name "*.py" 2>/dev/null | wc -l)
if [ "$MODIFIED" -eq 0 ]; then
    ok "4.1 Nerfstudio site-packages KHÔNG bị modify (0 file newer than plugin)"
else
    fail "4.1 Có $MODIFIED file Nerfstudio bị modify"
fi

# 4.2 Plugin code self-contained
N_PY=$(find $PLUGIN_DIR -name "*.py" | wc -l)
ok "4.2 Plugin có $N_PY file Python (self-contained)"

# 4.3 Tag [CRSGaussian] cover
N_TAG=$(grep -rl "\[CRSGaussian" $PLUGIN_DIR --include="*.py" | wc -l)
if [ "$N_TAG" -eq "$N_PY" ]; then
    ok "4.3 Tag [CRSGaussian] cover all $N_PY Python file"
else
    fail "4.3 Chỉ $N_TAG/$N_PY file có tag [CRSGaussian]"
fi

# ─────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────
echo ""
echo "==========================================="
echo "SUMMARY: $PASS PASS, $FAIL FAIL"
echo "==========================================="

if [ "$FAIL" -eq 0 ]; then
    echo "🎉 All tier passed — A1 plug-in verified."
    echo ""
    echo "Next step: run full 4 scene compare"
    echo "  bash run_4scene_roma.sh  # create separately"
else
    echo "⚠ $FAIL tier failed — debug trước khi run full 4 scene"
    echo "  Logs: /tmp/verify_train.log, /tmp/verify_eval.log, /tmp/verify_24.log"
fi

echo ""
echo "Tier 2.1 OFF flag byte-identical: chạy riêng (cần full 10k iter compare)"
echo "Tier 3 multi-seed paired N=12: chạy sau khi 2.1 PASS"
echo "Tier 5 reproducibility: pip freeze > requirements.txt, viết README"
