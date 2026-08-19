#!/bin/bash
# ============================================================
# [CRSGaussian Phase 26] Verify script — chạy TRƯỚC khi ablation thật.
# File: scripts/p26_verify.sh (NEW)
#
# Mục đích: gom mọi bước kiểm tra Phase 26 (A1 V_stability, C1 density-
# frequency modulation, A2 temporal regularization) thành 1 script chạy
# tuần tự trên server (A4000, conda env `redigs`) — không cần nhớ/gõ lại
# từng lệnh. Mỗi bước có comment giải thích NÓ KIỂM TRA GÌ và THẤT BẠI
# NGHĨA LÀ GÌ.
#
# Usage:
#   conda activate redigs
#   cd ReDI-GS   (sau khi transfer code lên server)
#   bash scripts/p26_verify.sh
#
# Dừng ngay (set -e) nếu bất kỳ bước nào fail — không chạy tiếp bước sau,
# để dễ khoanh vùng lỗi thay vì đọc log dài.
#
# Toàn bộ smoke-train dùng --iterations 100 (không phải 10000) — chỉ để
# bắt lỗi runtime/shape sớm (import fail, tensor shape mismatch, NaN),
# KHÔNG phải để đánh giá chất lượng. Đánh giá thật nằm ở
# scripts/p26_ablation_run.sh (10k iter, multi-scene, multi-seed).
# ============================================================
set -eo pipefail

SCENE=${SCENE:-fern}   # scene nhỏ, nhanh — đổi qua SCENE=xxx nếu cần
DATA_DIR="data/nerf_llff_data/${SCENE}"
SMOKE_ROOT="output/p26_verify_smoke"
LOG_DIR="logs/p26_verify"
mkdir -p "${SMOKE_ROOT}" "${LOG_DIR}"

echo "############################################################"
echo "# STEP 0 — Preconditions"
echo "############################################################"
echo "Scene dùng cho smoke test: ${SCENE} (${DATA_DIR})"
if [ ! -d "${DATA_DIR}" ]; then
    echo "LỖI: ${DATA_DIR} không tồn tại. Kiểm tra data đã đặt đúng chỗ chưa"
    echo "(xem README phần 'Reproducing the paper results')."
    exit 1
fi

echo ""
echo "############################################################"
echo "# STEP 1 — Unit test CŨ (regression check: A1/C1/A2 không phá gì cũ)"
echo "############################################################"
echo "--- test_update_crs.py (CRS formula gốc — verify V=None vẫn behavior cũ) ---"
python tests/test_update_crs.py
echo "--- test_reprojection_consistency.py ---"
python tests/test_reprojection_consistency.py
echo "--- test_depth_consistency.py ---"
python tests/test_depth_consistency.py
echo "--- test_crs_pruning.py ---"
python tests/test_crs_pruning.py

echo ""
echo "############################################################"
echo "# STEP 2 — Unit test MỚI (Phase 26 — A1/C1/A2 riêng lẻ, mock-based,"
echo "#          không cần data thật, chạy trong vài giây)"
echo "############################################################"
echo "--- test_v_stability.py (A1: EMA drift signal + geom freeze) ---"
python tests/test_v_stability.py
echo "--- test_temporal_reg.py (A2: temporal parameter loss) ---"
python tests/test_temporal_reg.py
echo "--- test_density_freq_modulate.py (C1: density-as-frequency modulation) ---"
python tests/test_density_freq_modulate.py

echo ""
echo "############################################################"
echo "# STEP 3 — Smoke train, TẤT CẢ flag Phase 26 OFF (default)"
echo "#          Mục đích: xác nhận code Phase 26 KHÔNG đụng baseline khi"
echo "#          tắt — nếu bước này lỗi, nghĩa là 1 trong các sửa đổi vào"
echo "#          train.py/crs_module.py/dropansh.py/gaussian_model.py đã lỡ"
echo "#          phá code path cũ (không phải lỗi ở module Phase 26 mới)."
echo "############################################################"
python train.py \
    --source_path "${DATA_DIR}" \
    --model_path "${SMOKE_ROOT}/baseline_off" \
    --eval -r 8 --n_views 3 --random_background \
    --iterations 100 --test_iterations 100 \
    --use_depth_prior --dav2_path ../Depth-Anything-V2 \
    2>&1 | tee "${LOG_DIR}/step3_baseline_off.log"
echo "STEP 3 xong — kiểm tra log kết thúc bằng dòng PSNR, không có traceback."

echo ""
echo "############################################################"
echo "# STEP 4 — Smoke train A1 ON (use_v_stability + geom freeze)"
echo "#          Mục đích: bắt lỗi runtime của V_stability signal + geom"
echo "#          freeze trong vòng lặp train.py thật (100 iter, nhanh)."
echo "############################################################"
python train.py \
    --source_path "${DATA_DIR}" \
    --model_path "${SMOKE_ROOT}/a1_on" \
    --eval -r 8 --n_views 3 --random_background \
    --iterations 100 --test_iterations 100 \
    --use_depth_prior --dav2_path ../Depth-Anything-V2 \
    --use_v_stability --v_stability_warmup 10 --crs_w_v 0.33 \
    --use_crs_modulated_geom_freeze --geom_freeze_start 10 --geom_freeze_tau 0.5 \
    2>&1 | tee "${LOG_DIR}/step4_a1_on.log"
echo "STEP 4 xong."

echo ""
echo "############################################################"
echo "# STEP 5 — Smoke train C1 ON (density-frequency modulation, cần"
echo "#          use_dropansh + use_opacity_decay làm prerequisite)"
echo "############################################################"
python train.py \
    --source_path "${DATA_DIR}" \
    --model_path "${SMOKE_ROOT}/c1_on" \
    --eval -r 8 --n_views 3 --random_background \
    --iterations 100 --test_iterations 100 \
    --use_depth_prior --dav2_path ../Depth-Anything-V2 \
    --use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2 \
    --use_opacity_decay --opacity_decay_factor 0.999 \
    --use_density_freq_modulate --density_freq_method voxel --density_freq_strength 1.0 \
    --opacity_decay_freq_modulate \
    2>&1 | tee "${LOG_DIR}/step5_c1_on.log"
echo "STEP 5 xong."

echo ""
echo "############################################################"
echo "# STEP 6 — Smoke train A2 ON (temporal parameter regularization)"
echo "############################################################"
python train.py \
    --source_path "${DATA_DIR}" \
    --model_path "${SMOKE_ROOT}/a2_on" \
    --eval -r 8 --n_views 3 --random_background \
    --iterations 100 --test_iterations 100 \
    --use_depth_prior --dav2_path ../Depth-Anything-V2 \
    --use_temporal_reg --temporal_reg_start_iter 10 --temporal_ema_beta 0.9 \
    --lambda_temporal_xyz 0.01 --lambda_temporal_shape 0.01 --temporal_crs_weighted True \
    2>&1 | tee "${LOG_DIR}/step6_a2_on.log"
echo "STEP 6 xong."

echo ""
echo "############################################################"
echo "# STEP 7 — Smoke train CẢ 3 module cùng lúc (A1+C1+A2, khớp"
echo "#          CONFIG=trim_full_a1c1a2 trong p26_ablation_run.sh nhưng"
echo "#          KHÔNG có trim_full base — chỉ test 3 module mới không"
echo "#          xung đột nhau khi bật đồng thời)"
echo "############################################################"
python train.py \
    --source_path "${DATA_DIR}" \
    --model_path "${SMOKE_ROOT}/a1c1a2_on" \
    --eval -r 8 --n_views 3 --random_background \
    --iterations 100 --test_iterations 100 \
    --use_depth_prior --dav2_path ../Depth-Anything-V2 \
    --use_v_stability --v_stability_warmup 10 --crs_w_v 0.33 \
    --use_crs_modulated_geom_freeze --geom_freeze_start 10 --geom_freeze_tau 0.5 \
    --use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2 \
    --use_opacity_decay --opacity_decay_factor 0.999 \
    --use_density_freq_modulate --density_freq_method voxel --density_freq_strength 1.0 \
    --opacity_decay_freq_modulate \
    --use_temporal_reg --temporal_reg_start_iter 10 --temporal_ema_beta 0.9 \
    --lambda_temporal_xyz 0.01 --lambda_temporal_shape 0.01 --temporal_crs_weighted True \
    2>&1 | tee "${LOG_DIR}/step7_all_on.log"
echo "STEP 7 xong."

echo ""
echo "############################################################"
echo "# ALL STEPS PASSED"
echo "# Log chi tiết: ${LOG_DIR}/"
echo "# Output smoke (xoá được, chỉ 100 iter, không dùng để đánh giá): ${SMOKE_ROOT}/"
echo "# Bước tiếp theo: chạy ablation thật qua scripts/p26_ablation_run.sh"
echo "############################################################"
