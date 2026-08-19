#!/bin/bash
# ============================================================
# [CRSGaussian Phase 26] Kaggle master orchestrator — "chạy 1 phát".
# File: scripts/kaggle_run_all.sh (NEW)
#
# Chạy TOÀN BỘ chuỗi: setup env → dense init (RoMa v1) → place_init →
# smoke verify → ablation (baseline + Phase 26 modules) → gom kết quả
# thành 1 thư mục summary — không cần can thiệp tay giữa chừng.
#
# Có CHECKPOINT (touch file đánh dấu mỗi phase xong) → nếu Kaggle session
# bị ngắt (mất mạng, hết giờ, restart kernel), chạy lại chính lệnh này sẽ
# BỎ QUA các phase đã xong, tiếp tục từ chỗ dở dang — không phải chạy lại
# từ đầu.
#
# Prerequisite: scripts/kaggle_setup.sh đã chạy xong (2 conda env +
# submodule CUDA build) — script này TỰ GỌI kaggle_setup.sh ở Phase 0 nếu
# chưa thấy checkpoint, nên thực ra chỉ cần chạy đúng 1 lệnh duy nhất từ
# đầu:
#   !bash scripts/kaggle_run_all.sh
#
# Cấu hình qua biến môi trường (đều có default hợp lý, có thể để trống):
#   ABLATION_CONFIGS   danh sách CONFIG cho p26_ablation_run.sh, cách nhau
#                       khoảng trắng. Default: "trim_full trim_full_a1
#                       trim_full_c1 trim_full_a2" (baseline + 3 module
#                       Phase 26 riêng lẻ — KHÔNG gồm các tổ hợp/multi-seed
#                       để vừa 1 phiên Kaggle; scale lên sau khi biết chắc
#                       chạy được).
#   ABLATION_SCENES     danh sách scene. Default: toàn bộ 8 scene.
#   ABLATION_SEEDS       danh sách seed. Default: "42" (1 seed — multi-seed
#                       tốn gấp 3 thời gian, chỉ bật sau khi có tín hiệu).
#
# Ví dụ chạy nhỏ hơn để test nhanh trước (2 scene, 2 config):
#   ABLATION_SCENES="fern flower" ABLATION_CONFIGS="trim_full trim_full_a1" \
#       bash scripts/kaggle_run_all.sh
#
# Output cuối: results/p26_kaggle_summary/ (bảng PSNR tổng hợp mỗi config
# + toàn bộ log gốc) — nằm trong /kaggle/working nếu code chạy từ đó, nên
# tự động được Kaggle lưu lại khi notebook commit xong.
# ============================================================
set -uo pipefail   # KHÔNG dùng -e ở đây: 1 scene/config lỗi không được
                    # phép làm sập toàn bộ pipeline — muốn giữ kết quả các
                    # phần đã chạy được. Từng phase con tự set -e riêng.

STATE_DIR=".kaggle_run_state"
mkdir -p "$STATE_DIR" logs results/p26_kaggle_summary

ABLATION_CONFIGS=${ABLATION_CONFIGS:-"trim_full trim_full_a1 trim_full_c1 trim_full_a2"}
ABLATION_SCENES=${ABLATION_SCENES:-"fern flower fortress horns leaves orchids room trex"}
ABLATION_SEEDS=${ABLATION_SEEDS:-"42"}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

checkpoint_done() { [ -f "$STATE_DIR/$1.done" ]; }
mark_done() { touch "$STATE_DIR/$1.done"; }

export PATH="$HOME/miniconda3/bin:$PATH"
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

echo "############################################################"
echo "# PHASE 0 — Setup env (2 conda env + build CUDA submodule + copy data)"
echo "############################################################"
if checkpoint_done "phase0_setup"; then
    log "Phase 0 đã xong (checkpoint tồn tại) — bỏ qua."
else
    set -e
    bash scripts/kaggle_setup.sh
    set +e
    mark_done "phase0_setup"
fi

echo ""
echo "############################################################"
echo "# PHASE 1 — Dense init RoMa v1 (env roma_v1), tất cả scene"
echo "############################################################"
if checkpoint_done "phase1_preprocess"; then
    log "Phase 1 đã xong — bỏ qua."
else
    set -e
    N_GPU=$(nvidia-smi -L 2>/dev/null | wc -l)
    log "Phát hiện $N_GPU GPU."
    if [ "$N_GPU" -ge 2 ]; then
        log "Chạy song song 2 GPU (giống scripts/preprocess_all.sh)."
        conda run -n roma_v1 bash scripts/preprocess_all.sh 2>&1 | tee logs/kaggle_preprocess.log
    else
        log "Chỉ 1 GPU — chạy tuần tự từng scene (chậm hơn nhưng an toàn)."
        > logs/kaggle_preprocess.log
        for sc in fern flower fortress horns leaves orchids room trex; do
            log "  preprocess scene=$sc"
            SCENE=$sc conda run -n roma_v1 python scripts/preprocess.py \
                2>&1 | tee -a logs/kaggle_preprocess.log
        done
    fi
    set +e
    mark_done "phase1_preprocess"
fi

echo ""
echo "############################################################"
echo "# PHASE 2 — place_init.py (swap fused.ply <- fused.ply.romav1)"
echo "############################################################"
if checkpoint_done "phase2_place_init"; then
    log "Phase 2 đã xong — bỏ qua."
else
    set -e
    conda run -n redigs python scripts/place_init.py 2>&1 | tee logs/kaggle_place_init.log
    set +e
    mark_done "phase2_place_init"
fi

echo ""
echo "############################################################"
echo "# PHASE 3 — Smoke verify (scripts/p26_verify.sh, ~vài phút)"
echo "#          Nếu phase này lỗi, DỪNG — không chạy Phase 4 (ablation"
echo "#          thật 10k iter sẽ vô nghĩa nếu code có bug runtime)."
echo "############################################################"
if checkpoint_done "phase3_verify"; then
    log "Phase 3 đã xong — bỏ qua."
else
    set -e
    conda run -n redigs bash scripts/p26_verify.sh 2>&1 | tee logs/kaggle_verify.log
    set +e
    mark_done "phase3_verify"
fi

echo ""
echo "############################################################"
echo "# PHASE 4 — Ablation thật (10k iter/run). KHÔNG dừng pipeline nếu"
echo "#          1 (config, scene, seed) lỗi — log lỗi vào manifest, tiếp"
echo "#          tục phần còn lại, để không mất kết quả đã chạy được."
echo "#          CONFIGS=$ABLATION_CONFIGS"
echo "#          SCENES=$ABLATION_SCENES"
echo "#          SEEDS=$ABLATION_SEEDS"
echo "############################################################"
FAIL_MANIFEST="results/p26_kaggle_summary/failed_runs.txt"
> "$FAIL_MANIFEST"
for CONFIG in $ABLATION_CONFIGS; do
    CKPT="phase4_ablation_${CONFIG}"
    if checkpoint_done "$CKPT"; then
        log "Config $CONFIG đã xong — bỏ qua."
        continue
    fi
    log "Bắt đầu CONFIG=$CONFIG"
    CONFIG="$CONFIG" \
    SCENES_OVERRIDE="$ABLATION_SCENES" \
    SEEDS_OVERRIDE="$ABLATION_SEEDS" \
    GPU=0 \
    conda run -n redigs bash scripts/p26_ablation_run.sh \
        2>&1 | tee "logs/kaggle_ablation_${CONFIG}.log"
    RC=${PIPESTATUS[0]:-$?}
    if [ "$RC" -ne 0 ]; then
        log "CẢNH BÁO: CONFIG=$CONFIG kết thúc với lỗi (RC=$RC) — xem logs/kaggle_ablation_${CONFIG}.log"
        echo "$CONFIG: RC=$RC" >> "$FAIL_MANIFEST"
    else
        mark_done "$CKPT"
    fi
done

echo ""
echo "############################################################"
echo "# PHASE 5 — Gom kết quả: bảng PSNR/SSIM/LPIPS tổng hợp mỗi CONFIG"
echo "#          + copy toàn bộ log vào results/p26_kaggle_summary/"
echo "############################################################"
SUMMARY_FILE="results/p26_kaggle_summary/summary.txt"
{
    echo "Phase 26 Kaggle ablation summary — $(date)"
    echo "CONFIGS: $ABLATION_CONFIGS"
    echo "SCENES:  $ABLATION_SCENES"
    echo "SEEDS:   $ABLATION_SEEDS"
    echo ""
} > "$SUMMARY_FILE"

for CONFIG in $ABLATION_CONFIGS; do
    echo "=== CONFIG=$CONFIG ===" >> "$SUMMARY_FILE"
    # PREFIX="$CONFIG" — p26_ablation_run.sh đặt tên log
    # "${CONFIG}_seed{SEED}_{SCENE}.log", KHÔNG phải "A3_seed..." mặc định
    # của analyze.py (đó là convention của scripts/run.sh/trainer.sh).
    LOG_DIR="logs/p26_ablation/${CONFIG}" PREFIX="${CONFIG}" \
    SCENES="$ABLATION_SCENES" SEEDS="$ABLATION_SEEDS" \
    conda run -n redigs python scripts/analyze.py >> "$SUMMARY_FILE" 2>&1
    echo "" >> "$SUMMARY_FILE"
done

cp -r logs/p26_ablation results/p26_kaggle_summary/logs_p26_ablation 2>/dev/null
cp logs/kaggle_*.log results/p26_kaggle_summary/ 2>/dev/null

echo ""
echo "############################################################"
echo "# XONG TOÀN BỘ PIPELINE."
echo "# Xem kết quả: results/p26_kaggle_summary/summary.txt"
echo "# Log lỗi (nếu có): $FAIL_MANIFEST"
echo "# Toàn bộ log chi tiết: results/p26_kaggle_summary/logs_p26_ablation/"
echo "############################################################"
cat "$SUMMARY_FILE"
if [ -s "$FAIL_MANIFEST" ]; then
    echo ""
    echo "⚠️  Có config lỗi — xem $FAIL_MANIFEST:"
    cat "$FAIL_MANIFEST"
fi
