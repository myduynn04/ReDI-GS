#!/bin/bash
# ============================================================
# [B6 Viewer Smooth Setup] 3-way realtime viewer setup
# File: crsgaussian_plugin/scripts/setup_3way_viewer.sh
#
# Mục đích:
#   1. Convert anh's crsgaussian Gaussians → splatfacto-sparse format
#      cho 3 scene (leaves, trex, horns) → folder mới splatfacto_3view_crs_view/
#   2. Print 3 viewer commands sẵn để paste vào 3 terminal
#
# Lý do: View anh's Phase 22 Gaussians qua gsplat rasterizer (smooth 30-60 FPS)
#        thay vì Inria rasterizer (lag 14 FPS + crash).
#
# CAUTION: tạo folder mới splatfacto_3view_crs_view/ (KHÔNG đè vanilla baseline).
# ============================================================

set -e

cd /home/aidev/workspace/representation-3d/duyen/nerfstudio

export NERFSTUDIO_METHOD_CONFIGS="crsgaussian=crsgaussian_plugin:crsgaussian_method_spec,splatfacto-sparse=crsgaussian_plugin:splatfacto_sparse_method_spec,splatfacto-17=crsgaussian_plugin:splatfacto_17_method_spec,splatfacto-sparse-noabs=crsgaussian_plugin:splatfacto_sparse_noabs_method_spec"
export PYTHONPATH=/home/aidev/workspace/representation-3d/duyen/nerfstudio:$PYTHONPATH
export CORGS_SOURCE_PATH=/home/aidev/workspace/representation-3d/duyen/CoR-GS

SCENES="leaves trex horns"

echo "═══════════════════════════════════════════════════════════"
echo "  3-Way Smooth Viewer Setup"
echo "═══════════════════════════════════════════════════════════"

# ────────────────────────────────────────────────────────
# STEP 1 — Convert anh's crsgaussian ckpt cho 3 scene
# ────────────────────────────────────────────────────────

for SCENE in $SCENES; do
    SRC_DIR=$(ls -td outputs/splatfacto_3view/$SCENE/splatfacto-sparse/* 2>/dev/null | head -1)
    if [ -z "$SRC_DIR" ]; then
        echo "❌ $SCENE: splatfacto_3view source folder không tồn tại — skip"
        continue
    fi

    NEW_DIR=outputs/splatfacto_3view_crs_view/$SCENE/splatfacto-sparse/$(basename $SRC_DIR)

    echo ""
    echo "═══ $SCENE ═══"

    # Clone whole folder (config.yml + dataparser_transforms.json + ckpt struct)
    if [ ! -d "$NEW_DIR" ]; then
        mkdir -p $(dirname $NEW_DIR)
        cp -r $SRC_DIR $NEW_DIR
        echo "  Cloned folder: $NEW_DIR"
    else
        echo "  Folder đã tồn tại: $NEW_DIR (giữ — chỉ re-convert ckpt)"
    fi

    # [B6 viewer-fix CRITICAL] Sửa output_dir trong config.yml để ns reconstruct
    # ckpt path từ FOLDER MỚI (không phải hardcoded path từ folder gốc cloned).
    # KHÔNG sửa = ns viewer load ckpt từ SOURCE folder (vanilla baseline) thay
    # vì NEW folder (anh's converted Gaussians).
    CONFIG=$NEW_DIR/config.yml
    if [ -f "$CONFIG" ]; then
        [ ! -f "${CONFIG}.original" ] && cp $CONFIG ${CONFIG}.original
        sed -i 's|outputs/splatfacto_3view/|outputs/splatfacto_3view_crs_view/|g' $CONFIG
        echo "  Config output_dir patched: $CONFIG"
    fi

    # Find anh's crsgaussian ckpt
    CRS_CKPT=$(ls outputs/phase22_b4/$SCENE/crsgaussian/*/nerfstudio_models/step-000009999.ckpt 2>/dev/null | tail -1)
    if [ -z "$CRS_CKPT" ]; then
        echo "  ❌ KHÔNG tìm thấy crsgaussian ckpt cho $SCENE"
        continue
    fi

    # Convert + overwrite ckpt trong NEW_DIR
    python crsgaussian_plugin/scripts/convert_crs_to_splatfacto.py \
        --src $CRS_CKPT \
        --dst $NEW_DIR/nerfstudio_models/step-000009999.ckpt 2>&1 | tail -5
done

# ────────────────────────────────────────────────────────
# STEP 2 — Print 3 viewer commands sẵn
# ────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ SETUP DONE — 3 viewer commands sẵn cho mỗi scene"
echo "═══════════════════════════════════════════════════════════"

ENV_BLOCK='export NERFSTUDIO_METHOD_CONFIGS="crsgaussian=crsgaussian_plugin:crsgaussian_method_spec,splatfacto-sparse=crsgaussian_plugin:splatfacto_sparse_method_spec,splatfacto-17=crsgaussian_plugin:splatfacto_17_method_spec,splatfacto-sparse-noabs=crsgaussian_plugin:splatfacto_sparse_noabs_method_spec" && export PYTHONPATH=/home/aidev/workspace/representation-3d/duyen/nerfstudio:$PYTHONPATH && cd /home/aidev/workspace/representation-3d/duyen/nerfstudio'

for SCENE in $SCENES; do
    echo ""
    echo "─── 🎬 Scene: $SCENE ───"
    echo ""
    echo "  Terminal A — anh's recipe (port 7007, GPU 0):"
    echo "  ┌─ COPY PASTE ─────────────────────────────────────────"
    echo "  $ENV_BLOCK"
    echo "    RUN=\$(ls -td outputs/splatfacto_3view_crs_view/$SCENE/splatfacto-sparse/* | head -1)"
    echo "    CUDA_VISIBLE_DEVICES=0 ns-viewer --load-config \$RUN/config.yml --viewer.websocket-port 7007"
    echo "  └──────────────────────────────────────────────────────"
    echo ""
    echo "  Terminal B — vanilla AbsGS=ON (port 7008, GPU 1):"
    echo "  ┌─ COPY PASTE ─────────────────────────────────────────"
    echo "  $ENV_BLOCK"
    echo "    RUN=\$(ls -td outputs/splatfacto_3view/$SCENE/splatfacto-sparse/* | head -1)"
    echo "    CUDA_VISIBLE_DEVICES=1 ns-viewer --load-config \$RUN/config.yml --viewer.websocket-port 7008"
    echo "  └──────────────────────────────────────────────────────"
    echo ""
    echo "  Terminal C — vanilla no-AbsGS (port 7009, GPU 0 share):"
    echo "  ┌─ COPY PASTE ─────────────────────────────────────────"
    echo "  $ENV_BLOCK"
    echo "    RUN=\$(ls -td outputs/splatfacto_3view_noabs/$SCENE/splatfacto-sparse-noabs/* | head -1)"
    echo "    CUDA_VISIBLE_DEVICES=0 ns-viewer --load-config \$RUN/config.yml --viewer.websocket-port 7009"
    echo "  └──────────────────────────────────────────────────────"
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  📝 Hướng dẫn:"
echo "═══════════════════════════════════════════════════════════"
echo "  1. Chọn 1 scene (leaves / trex / horns)"
echo "  2. Mở 3 terminal SSH"
echo "  3. Copy-paste block A vào Terminal A, B vào Terminal B, C vào C"
echo "  4. VSCode PORTS tab → forward 7007, 7008, 7009"
echo "  5. Browser:"
echo "       http://localhost:7007 → anh's recipe (smooth gsplat)"
echo "       http://localhost:7008 → vanilla AbsGS=ON (smooth gsplat)"
echo "       http://localhost:7009 → vanilla no-AbsGS (smooth gsplat)"
echo ""
echo "  Đổi scene: Ctrl+C 3 viewer, đổi 'leaves' → 'trex'/'horns' trong RUN line"
echo "═══════════════════════════════════════════════════════════"
