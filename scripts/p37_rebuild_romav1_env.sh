#!/usr/bin/env bash
# ============================================================
# [CRSGaussian Phase 37] Dựng lại env RoMa v1
# File: scripts/p37_rebuild_romav1_env.sh  (KEEP LOCAL — server-only)
#
# Bối cảnh: env `roma_v1` biến mất (phát hiện 2026-08-06). 8 file
#   fused.ply.romav1 CÒN NGUYÊN → Phase 22 / 21.89 an toàn, nhưng
#   không chạy lại preprocess được → chặn H1 (docs/37) VÀ chặn
#   reproducibility của Phase 22.
#
# Công thức lấy NGUYÊN VĂN từ docs/04_decisions_log.md:2122-2127
# (mục [2026-05-25] Phase 22 Setup Bước 0-2), KHÔNG phải đoán:
#
#   - Repo: Parskatt/RoMa (CVPR 2024) → ~/workspace/representation-3d/duyen/RoMa
#           ⚠ TÊN VIẾT HOA "RoMa" — glob `*roma*` KHÔNG khớp, đừng tưởng mất
#   - Env : roma_v1, isolated
#   - torch==2.6.0 torchvision==0.21.0, index-url cu124 (driver server 12.4)
#   - Weights: roma_outdoor.pth 425MB + dinov2_vitl14_pretrain.pth 1.13GB ≈ 1.55GB
#
# 🔴 BẪY ĐÃ GHI TRONG LOG — TUYỆT ĐỐI KHÔNG LẶP:
#   `pip install fused-local-corr` kéo theo torch 2.11+cu13 → lệch driver 12.4
#   → hỏng cả env. Phase 22 dùng use_custom_corr=False (fallback thuần PyTorch
#   `shitty_native_torch_local_corr`, RoMa/romatch/utils/local_correlation.py:39)
#   nên KHÔNG CẦN gói đó. Script này KHÔNG cài nó.
#
# Usage:
#   bash scripts/p37_rebuild_romav1_env.sh 2>&1 | tee logs/p37/rebuild_env.log
#
# Script CHỈ tạo env mới tên roma_v1. KHÔNG đụng env romav2/corgs/nerfstudio.
# ============================================================
set -u

DUYEN="${DUYEN:-$HOME/workspace/representation-3d/duyen}"
ROMA_DIR="${ROMA_DIR:-${DUYEN}/RoMa}"
ENV_NAME="${ENV_NAME:-roma_v1}"
PY_VER="${PY_VER:-3.10}"
CKPT_DIR="$HOME/.cache/torch/hub/checkpoints"

mkdir -p logs/p37

hr () { printf '%s\n' "------------------------------------------------------------"; }
die () { echo; echo "❌ $*"; echo "   DỪNG — không làm gì thêm."; exit 1; }

# ============================================================
# [CRSGaussian P37] BOOTSTRAP CONDA — phải làm TRƯỚC MỌI THỨ
# Lý do: `conda` là shell FUNCTION do `conda init` định nghĩa trong
#        ~/.bashrc, CHỈ có trong shell tương tác. Chạy `bash script.sh`
#        là non-interactive → `conda: command not found`.
#        Lần chạy 2026-08-07 dính đúng lỗi này, và tệ hơn là bước kiểm
#        tra env sau đó báo "✅ chưa tồn tại" trong khi thật ra lệnh
#        vừa fail → FALSE PASS.
#        → Nạp conda.sh ngay đây, và die nếu không nạp được.
# ============================================================
echo "[P37 REBUILD] bootstrap conda..."
# Cho phép ép từ ngoài: CONDA_BASE=/path bash scripts/p37_rebuild_romav1_env.sh
CONDA_BASE="${CONDA_BASE:-}"
if [ -n "${CONDA_BASE}" ]; then
  echo "  nguồn: \$CONDA_BASE do người dùng ép → ${CONDA_BASE}"
elif [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
  CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
  echo "  nguồn: \$CONDA_EXE → ${CONDA_BASE}"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null)"
  echo "  nguồn: conda trong PATH → ${CONDA_BASE}"
else
  for cand in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" \
              "$HOME/mambaforge" "/opt/conda" "/usr/local/miniconda3"; do
    if [ -f "${cand}/etc/profile.d/conda.sh" ]; then
      CONDA_BASE="${cand}"
      echo "  nguồn: dò đường dẫn quen thuộc → ${CONDA_BASE}"
      break
    fi
  done
fi

[ -n "${CONDA_BASE}" ] || die "Không tìm thấy conda.
   Thử: echo \$CONDA_EXE   /   ls ~/miniconda3/etc/profile.d/conda.sh
   Rồi chạy lại với: CONDA_BASE=/duong/dan/conda bash \$0"

[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ] \
  || die "Thiếu ${CONDA_BASE}/etc/profile.d/conda.sh"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
command -v conda >/dev/null 2>&1 || die "source conda.sh xong vẫn không gọi được conda"
echo "  ✅ conda sẵn sàng: $(conda --version)"

echo
echo "============================================================"
echo "[P37 REBUILD] RoMa v1 env — $(date '+%F %T')"
echo "  repo = ${ROMA_DIR}"
echo "  env  = ${ENV_NAME}   python ${PY_VER}"
echo "  conda base = ${CONDA_BASE}"
echo "============================================================"

# ── BƯỚC 1: repo còn không ──
echo
echo "[1] Kiểm tra repo RoMa v1"
hr
if [ -d "${ROMA_DIR}" ]; then
  echo "  ✅ repo CÒN: ${ROMA_DIR}"
  git -C "${ROMA_DIR}" log --oneline -2 2>/dev/null | sed 's/^/     /'
  if [ -f "${ROMA_DIR}/romatch/utils/local_correlation.py" ]; then
    echo "  ✅ local_correlation.py có mặt"
    grep -n "shitty_native_torch_local_corr\|use_custom_corr" \
      "${ROMA_DIR}/romatch/utils/local_correlation.py" 2>/dev/null | head -4 | sed 's/^/     /'
  else
    echo "  ⚠ KHÔNG thấy romatch/utils/local_correlation.py — repo có thể không đầy đủ"
  fi
else
  echo "  ❌ KHÔNG thấy ${ROMA_DIR}"
  echo
  echo "  Tìm case-insensitive quanh workspace (phòng khi tên khác):"
  find "${DUYEN}" -maxdepth 1 -iname "*roma*" 2>/dev/null | sed 's/^/     /'
  echo
  die "Repo RoMa v1 không còn. Cần clone lại (CẦN INTERNET):
     git clone https://github.com/Parskatt/RoMa ${ROMA_DIR}
   Rồi chạy lại script này."
fi

# ── BƯỚC 2: weights còn cache không (phần tải lâu nhất) ──
echo
echo "[2] Kiểm tra weights cache"
hr
NEED_NET=0
for w in roma_outdoor.pth dinov2_vitl14_pretrain.pth; do
  if [ -f "${CKPT_DIR}/${w}" ]; then
    echo "  ✅ ${w}  ($(du -h "${CKPT_DIR}/${w}" | cut -f1))"
  else
    echo "  ⚠ THIẾU ${w} — sẽ tải lần chạy đầu (cần internet)"
    NEED_NET=1
  fi
done
[ "${NEED_NET}" -eq 1 ] && echo "  → lần smoke đầu tiên CẦN INTERNET (~1.55GB)" \
                        || echo "  → đủ weights, KHÔNG cần internet"

# ── BƯỚC 3: env đã tồn tại chưa ──
echo
echo "[3] Kiểm tra env ${ENV_NAME}"
hr
# Bắt output RIÊNG rồi mới xét — nếu chính lệnh fail thì phải die,
# KHÔNG được để pipeline nuốt lỗi rồi kết luận "chưa tồn tại" (false pass,
# đúng bug đã dính 2026-08-07).
ENV_LIST="$(conda env list 2>&1)" || die "conda env list thất bại:
${ENV_LIST}"

if printf '%s\n' "${ENV_LIST}" | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "  ⚠ env ${ENV_NAME} ĐÃ TỒN TẠI."
  echo "     Script này KHÔNG ghi đè. Muốn dựng lại sạch thì tự xoá trước:"
  echo "       conda env remove -n ${ENV_NAME}"
  die "env đã có — dừng để tránh phá state đang chạy."
fi
echo "  ✅ chưa tồn tại — tạo mới an toàn"

# ── BƯỚC 4: tạo env ──
echo
echo "[4] Tạo env ${ENV_NAME} (python ${PY_VER})"
hr
conda create -n "${ENV_NAME}" "python=${PY_VER}" -y || die "conda create thất bại"

# conda.sh đã source ở bootstrap đầu file → activate dùng được ngay
conda activate "${ENV_NAME}" || die "không activate được ${ENV_NAME}"
echo "  ✅ active: ${CONDA_DEFAULT_ENV}"
# Chốt chặn: đảm bảo python đang dùng LÀ python của env vừa tạo,
# không phải python của base rò rỉ vào.
PY_PATH="$(command -v python)"
echo "  python : ${PY_PATH}"
case "${PY_PATH}" in
  *"/envs/${ENV_NAME}/"*) : ;;
  *) die "python KHÔNG thuộc env ${ENV_NAME} (đang là ${PY_PATH}) — dừng để khỏi cài nhầm vào base" ;;
esac

# ── BƯỚC 5: torch PIN CHÍNH XÁC — trước mọi thứ khác ──
# Lý do phải cài torch TRƯỚC `pip install -e .`: setup.py của RoMa để torch
# không pin, nếu để pip tự giải sẽ kéo bản mới nhất → lệch driver cu124.
echo
echo "[5] Cài torch 2.6.0 + torchvision 0.21.0 (cu124) — PIN, không để pip tự chọn"
hr
pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124 \
  || die "cài torch thất bại"

python - <<'PY' || die "torch không dùng được CUDA"
import torch
print(f"  torch {torch.__version__}  cuda_build={torch.version.cuda}  avail={torch.cuda.is_available()}  n_gpu={torch.cuda.device_count()}")
assert torch.cuda.is_available(), "CUDA không khả dụng"
assert torch.version.cuda.startswith("12.4"), f"BUILD CUDA SAI: {torch.version.cuda} (cần 12.4)"
PY

# ── BƯỚC 6: cài romatch từ repo local ──
# 🔴 KHÔNG cài fused-local-corr. Xem header.
echo
echo "[6] pip install -e ${ROMA_DIR}"
hr
pip install --no-cache-dir -e "${ROMA_DIR}" || die "pip install -e RoMa thất bại"

# torch có bị gói khác kéo đổi phiên bản không — kiểm tra lại
python - <<'PY' || die "torch bị đổi phiên bản sau khi cài romatch — env hỏng, xoá và làm lại"
import torch
print(f"  torch SAU khi cài romatch: {torch.__version__}  cuda={torch.version.cuda}")
assert torch.__version__.startswith("2.6.0"), f"torch bị đổi thành {torch.__version__}"
assert torch.version.cuda.startswith("12.4"), f"cuda build bị đổi thành {torch.version.cuda}"
PY

# ── BƯỚC 7: deps mà p37 preprocess cần ──
echo
echo "[7] Deps cho p37 preprocess"
hr
pip install --no-cache-dir opencv-python-headless plyfile pillow numpy || die "cài deps thất bại"

# ── BƯỚC 8: verify API đúng như Phase 22 đã dùng ──
echo
echo "[8] Verify API"
hr
python - <<'PY' || die "API không khớp Phase 22 — KHÔNG chạy preprocess"
import inspect
import romatch
from romatch import roma_outdoor
print(f"  romatch      : {romatch.__file__}")
sig = inspect.signature(roma_outdoor)
has = "use_custom_corr" in sig.parameters
print(f"  roma_outdoor : OK   use_custom_corr={'✅ có' if has else '❌ KHÔNG có'}")
assert has, "thiếu use_custom_corr — sai bản repo, Phase 22 phụ thuộc flag này"
from romatch.utils.local_correlation import local_correlation
print("  local_correlation import OK")
for m in ("cv2", "plyfile", "PIL"):
    __import__(m); print(f"  {m:<12}: ok")
PY

echo
echo "============================================================"
echo "✅ [P37 REBUILD] XONG — env ${ENV_NAME} sẵn sàng"
echo "============================================================"
echo
echo "Bước tiếp — smoke 1 scene TRƯỚC khi chạy cả 8:"
echo "    conda activate ${ENV_NAME}"
echo "    cd ~/workspace/representation-3d/duyen/CoR-GS"
echo "    SCENE=fern python scripts/p37_romav1_preprocess_qinit.py 2>&1 | tee logs/p37/smoke_fern.log"
echo
if [ "${NEED_NET}" -eq 1 ]; then
  echo "⚠ Lần smoke đầu CẦN INTERNET để tải ~1.55GB weights."
fi
echo "Smoke OK rồi mới chạy:  bash scripts/p37_s1_run_all.sh"
echo
echo "🔑 KIỂM TRA REPRODUCIBILITY (quan trọng hơn cả H1):"
echo "   p37 sinh fused.ply.romav1_p37 — phải TRÙNG số điểm với fused.ply.romav1 gốc"
echo "   (Phase 22 ghi nhận 16.9k-26.4k điểm/scene, tổng 178k)."
echo "   Lệch nhiều = env dựng lại KHÔNG tương đương → Phase 22 không reproduce được."
