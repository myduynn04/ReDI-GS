#!/usr/bin/env bash
# ============================================================
# [CRSGaussian Phase 37 — S1] Runner 8 scene, SONG SONG 2 GPU
# File: scripts/p37_s1_run_all.sh  (KEEP LOCAL — server-only)
#
# Pre-requisite:
#   conda activate roma_v1
#   cd ~/workspace/representation-3d/duyen/CoR-GS
#
# Usage:
#   bash scripts/p37_s1_run_all.sh
#
# Chia việc: GPU 0 = 4 scene đầu, GPU 1 = 4 scene sau, chạy đồng thời.
# Chi phí ước lượng: ~1.1s/pair × 3 pair × 4 scene ≈ vài phút mỗi GPU.
#
# Output mỗi scene:
#   data/nerf_llff_data/<scene>/3_views/dense/fused.ply.romav1_p37
#   data/nerf_llff_data/<scene>/3_views/dense/fused.romav1.qinit.npz
#   logs/p37/s1_<scene>.log
#
# ⚠ KHÔNG đè fused.ply — script chỉ GHI THÊM, không swap init.
#   Phase 22 pipeline không bị ảnh hưởng.
# ============================================================
set -u   # KHÔNG dùng -e: một scene fail không được giết cả batch

LOG_DIR="logs/p37"
mkdir -p "${LOG_DIR}"

GPU0_SCENES="fern flower fortress horns"
GPU1_SCENES="leaves orchids room trex"

# ============================================================
# [CRSGaussian P37] PREFLIGHT — dừng NGAY nếu môi trường sai
# Lý do: lần chạy đầu 2026-08-06, `conda activate roma_v1` fail
#        (env không tồn tại) → script vẫn chạy tiếp trong env sai,
#        cả 8 scene chết vì ImportError trong 9 giây, và gate in ra
#        "MISSING" che mất nguyên nhân thật.
#        → Fail fast, báo đúng lỗi, đừng để nó trông như kết quả S1.
# ============================================================
echo "[P37 S1] preflight..."
echo "  conda env : ${CONDA_DEFAULT_ENV:-<none>}"
echo "  python    : $(command -v python || echo '<not found>')"

# ⚠ KHÔNG viết `python - <<'PY' || { ... }` nhiều dòng:
#   bash bắt đầu đọc thân heredoc từ DÒNG NGAY SAU dòng chứa `<<`, nên khối
#   ngoặc nhọn bị nuốt vào heredoc, `PY` đóng heredoc, dấu `{` không bao giờ
#   được đóng → "syntax error: unexpected end of file" (dính 2026-08-07).
#   Cách an toàn: chạy heredoc như lệnh đơn, bắt $? ở dòng riêng.
python - <<'PY'
import sys
ok = True
try:
    import torch
    print(f"  torch     : {torch.__version__}  cuda={torch.cuda.is_available()}  n_gpu={torch.cuda.device_count()}")
except Exception as e:
    print(f"  torch     : FAIL — {e}"); ok = False
for mod in ("romatch", "cv2", "plyfile", "PIL"):
    try:
        __import__(mod)
        print(f"  {mod:<10}: ok")
    except Exception as e:
        print(f"  {mod:<10}: FAIL — {e}"); ok = False
sys.exit(0 if ok else 1)
PY
PREFLIGHT_RC=$?

if [ "${PREFLIGHT_RC}" -ne 0 ]; then
  echo
  echo "❌ [P37 S1] PREFLIGHT FAIL — môi trường chưa sẵn sàng. KHÔNG chạy tiếp."
  echo "   Sửa env rồi chạy lại. Kiểm tra: conda env list"
  exit 1
fi

echo "[P37 S1] preflight OK"
echo

echo "============================================================"
echo "[P37 S1] start $(date '+%F %T')"
echo "  GPU 0: ${GPU0_SCENES}"
echo "  GPU 1: ${GPU1_SCENES}"
echo "  logs : ${LOG_DIR}/"
echo "============================================================"

run_group () {
  local gpu="$1"; shift
  for sc in "$@"; do
    echo "[GPU ${gpu}] ${sc} → ${LOG_DIR}/s1_${sc}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" SCENE="${sc}" \
      python scripts/p37_romav1_preprocess_qinit.py \
      > "${LOG_DIR}/s1_${sc}.log" 2>&1
    if [ $? -ne 0 ]; then
      echo "[GPU ${gpu}] ❌ ${sc} FAILED — xem ${LOG_DIR}/s1_${sc}.log"
    else
      echo "[GPU ${gpu}] ✅ ${sc} done"
    fi
  done
}

run_group 0 ${GPU0_SCENES} &
PID0=$!
run_group 1 ${GPU1_SCENES} &
PID1=$!

wait ${PID0} ${PID1}

# ── Đếm sidecar thật sự sinh ra trước khi gọi gate ──
# Nếu 0 file thì đây là lỗi HẠ TẦNG, không phải phán quyết S1.
# Phân biệt hai thứ này quan trọng: "signal suy biến" và "script chết"
# cho ra cùng một màn hình MISSING nếu không chặn ở đây.
N_NPZ=$(ls -1 data/nerf_llff_data/*/3_views/dense/fused.romav1.qinit.npz 2>/dev/null | wc -l)
echo "============================================================"
echo "[P37 S1] preprocess xong $(date '+%F %T') — sidecar sinh ra: ${N_NPZ}/8"
echo "============================================================"

if [ "${N_NPZ}" -eq 0 ]; then
  echo "❌ [P37 S1] KHÔNG có sidecar nào — LỖI HẠ TẦNG, KHÔNG PHẢI KẾT QUẢ S1."
  echo "   ⚠ ĐỪNG đọc đây là 'signal suy biến'. Xem lỗi thật:"
  echo "       head -30 ${LOG_DIR}/s1_fern.log"
  echo "       grep -inE 'error|traceback|modulenotfound' ${LOG_DIR}/s1_*.log"
  exit 1
fi

python scripts/p37_s1_qinit_stats.py 2>&1 | tee "${LOG_DIR}/s1_gate_verdict.log"

echo
echo "[P37 S1] HOÀN TẤT. Verdict: ${LOG_DIR}/s1_gate_verdict.log"
