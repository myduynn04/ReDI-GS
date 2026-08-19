#!/bin/bash
# ============================================================
# [CRSGaussian Phase 26] Kaggle notebook setup — cài 2 conda env
# (`redigs` cho train, `roma_v1` cho dense init) + build 2 CUDA submodule,
# rồi copy data LLFF (arenagrenade/llff-dataset-full hoặc dataset tương tự
# cấu trúc) từ /kaggle/input (read-only) sang chỗ ghi được.
# File: scripts/kaggle_setup.sh (UPDATED)
#
# CHẠY TỪ 1 CELL trong Kaggle notebook, không phải máy local (không có
# GPU/CUDA). Dán nội dung dưới đây vào 1 cell bash (dùng %%bash hoặc
# !bash scripts/kaggle_setup.sh sau khi đã upload code ReDI-GS lên notebook).
#
# Trước khi chạy, cần trong Kaggle notebook:
#   1. Settings > Accelerator > chọn GPU (T4 x2 hoặc P100).
#   2. Add Input > tìm dataset LLFF (vd arenagrenade/llff-dataset-full) >
#      Add — Kaggle tự mount READ-ONLY vào /kaggle/input/<slug>/. KHÔNG
#      cần code kagglehub gì để "tải" nó — đã mount sẵn dưới dạng thư mục.
#      (Lưu ý: KaggleDatasetAdapter.PANDAS trong code mẫu Kaggle đưa ra là
#      để tải dữ liệu DẠNG BẢNG — CSV/DataFrame — không dùng được cho bộ
#      ảnh + file COLMAP nhị phân này. Bỏ qua đoạn code đó.)
#   3. Upload code ReDI-GS lên (Kaggle Dataset riêng hoặc Add Utility
#      Script). Không dùng được đường dẫn local /home/khanhtty/... — chỉ
#      tồn tại trên máy dev, không tồn tại trên Kaggle.
#
# ⚠️ QUAN TRỌNG — khác biệt Kaggle vs máy dev gốc (A4000):
#   - /kaggle/input LÀ READ-ONLY. preprocess.py ghi fused.ply.romav1 VÀO
#     TRONG thư mục scene, place_init.py ghi đè fused.ply — cả hai đều
#     cần quyền ghi. KHÔNG symlink thẳng dataset vào data/nerf_llff_data
#     (sẽ lỗi Permission denied ngay bước preprocess) — script này COPY
#     toàn bộ sang $KAGGLE_WORK/ReDI-GS/data/nerf_llff_data (ghi được).
#   - GPU khác kiến trúc (T4=compute 7.5, P100=compute 6.0, A4000=8.6).
#     torch.utils.cpp_extension tự detect kiến trúc đang chạy khi build —
#     không cần sửa gì, nhưng NẾU build lỗi liên quan "no kernel image is
#     available", set TORCH_CUDA_ARCH_LIST trước khi build (xem STEP 5).
#   - Kaggle session giới hạn 12h liên tục + quota GPU ~30h/tuần — không
#     đủ chạy multi-seed đầy đủ trong 1 lần (README ước ~5h/24-run trên
#     A4000; T4 chậm hơn). Dùng scripts/kaggle_run_all.sh (có checkpoint/
#     resume) thay vì chạy tay từng lệnh.
#   - Kaggle base image không có Miniconda/Python 3.8 mặc định → script
#     này cài Miniconda RIÊNG ở $HOME/miniconda3, không đụng Python hệ
#     thống Kaggle.
#   - `roma_v1` (dense init) và `redigs` (train) là 2 conda env TÁCH BIỆT
#     (Python 3.12/torch 2.6+cu124 vs Python 3.8/torch 2.1+cu121) — đúng
#     như README, không gộp chung.
#
# Usage (trong Kaggle notebook cell, sau khi code đã ở cwd hiện tại):
#   !bash scripts/kaggle_setup.sh
# ============================================================
set -eo pipefail

echo "############################################################"
echo "# STEP 0 — Verify GPU visible trong Kaggle notebook"
echo "############################################################"
nvidia-smi || { echo "LỖI: không thấy GPU. Bật Accelerator > GPU trong Settings trước."; exit 1; }

echo ""
echo "############################################################"
echo "# STEP 1 — Cài Miniconda riêng (không đụng Python hệ thống Kaggle)"
echo "############################################################"
if [ ! -d "$HOME/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
fi
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda --version

# Conda bản mới (26+) bắt buộc accept Terms of Service cho 2 channel mặc
# định trước khi tạo bất kỳ env nào non-interactive — environment.yml +
# environment_roma.yml đều dùng 2 channel này. Accept 1 lần, idempotent
# (chạy lại không lỗi nếu đã accept).
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# environment.yml / environment_roma.yml là `conda env export` ĐẦY ĐỦ,
# xuất trên server gốc (CPU AMD Zen2) — khoá cứng build-hash riêng cho
# kiến trúc đó (vd `_x86_64-microarch-level=3=3_zen2`). Trên máy khác
# kiến trúc (Kaggle không phải Zen2), conda không tìm được đúng build đó
# → solve fail. Cách chuẩn: "làm lỏng" — bỏ build-hash (chỉ giữ name=
# version), bỏ 3 gói meta không portable, bỏ dòng `prefix:` (trỏ path
# server cũ, có thể ghi đè `name:`). CHỈ áp dụng cho phần conda deps —
# phần `pip:` (torch==2.1.0+cu121 v.v.) giữ nguyên pin chính xác, vì đó
# là thứ quyết định tương thích với rasterizer CUDA tuỳ biến.
#
# 3 dòng pip local KHÔNG tồn tại trên PyPI (build từ submodules/ ở STEP 5,
# hoặc rác thừa từ môi trường gốc — evaluation==0.0.2 không được import ở
# đâu trong code) — phải loại khỏi requirements pip, nếu không conda env
# create sẽ báo "No matching distribution found".
#
# colmap + ceres-solver: verified KHÔNG được gọi ở đâu trong luồng RoMa v1
# (grep toàn bộ preprocess.py/place_init.py/train.py/dataset_readers.py —
# 0 lời gọi subprocess "colmap"). Package colmap khoá build "cuda_126"
# riêng, rất dễ solve-fail hoặc kéo theo CUDA toolkit lệch với torch pip
# cu121 — bỏ hẳn 2 gói này khỏi conda deps cho luồng Kaggle (chỉ cần nếu
# sau này muốn chạy tools/colmap_llff.py — MVS path — KHÔNG dùng trong
# kaggle_run_all.sh).
# torch/torchvision/torchaudio pin dạng "==X.Y.Z+cuNNN" — hậu tố +cuNNN
# là wheel CUDA riêng của PyTorch, CHỈ có trên index riêng của họ
# (download.pytorch.org/whl/cuNNN), KHÔNG có trên PyPI mặc định → pip
# thường sẽ báo "No matching distribution found". Phải chèn
# --extra-index-url đúng version CUDA vào đầu pip requirements list.
# $3 = URL index cần chèn (khác nhau giữa redigs=cu121 và roma_v1=cu124).
loosen_env_yml() {
    awk -v extra_index="$3" '
        BEGIN { in_pip = 0 }
        /^prefix:/ { next }
        /^  - pip:/ {
            in_pip = 1
            print
            if (extra_index != "") print "      - --extra-index-url " extra_index
            next
        }
        in_pip && /diff-gaussian-rasterization==0\.0\.0/ { next }
        in_pip && /simple-knn==0\.0\.0/ { next }
        in_pip && /evaluation==0\.0\.2/ { next }
        in_pip { print; next }
        /^  - _libgcc_mutex=/ { next }
        /^  - _openmp_mutex=/ { next }
        /^  - _x86_64-microarch-level=/ { next }
        /^  - colmap=/ { next }
        /^  - ceres-solver=/ { next }
        /^  - [^=]+=[^=]+=[^=]+$/ {
            n = split($0, parts, "=")
            if (n == 3) { print parts[1] "=" parts[2]; next }
        }
        { print }
    ' "$1" > "$2"
}

# Kiểm tra env "hoạt động thật" (import được package pip chính), không
# chỉ "có tồn tại tên". Lần chạy trước có thể đã tạo env NHƯNG pip fail
# giữa chừng (env vẫn có Python, chỉ thiếu torch/toàn bộ pip deps) — nếu
# chỉ check tên tồn tại, các lần chạy sau sẽ SKIP tạo lại, kẹt mãi với
# env dở dang. Phát hiện + xoá + tạo lại nếu env hỏng.
env_is_functional() {
    conda run -n "$1" python -c "import torch" >/dev/null 2>&1
}

echo ""
echo "############################################################"
echo "# STEP 2 — Tạo conda env 'redigs' (train, Python 3.8 + torch 2.1+cu121)"
echo "#          Mất khoảng 5-15 phút."
echo "############################################################"
if conda env list | grep -q "^redigs " && ! env_is_functional redigs; then
    echo "  env 'redigs' tồn tại nhưng KHÔNG hoàn chỉnh (thiếu torch) — có thể"
    echo "  do lần chạy trước lỗi giữa chừng. Xoá và tạo lại từ đầu."
    conda env remove -n redigs -y
fi
if ! conda env list | grep -q "^redigs "; then
    loosen_env_yml environment.yml /tmp/environment_redigs_kaggle.yml \
        https://download.pytorch.org/whl/cu121
    conda env create --file /tmp/environment_redigs_kaggle.yml
fi
conda run -n redigs python --version   # kỳ vọng: Python 3.8.20
conda run -n redigs python -c "import torch; print('torch:', torch.__version__, '| cuda available:', torch.cuda.is_available())"

# [CRSGaussian Phase 26] tensorboard KHÔNG có trong environment.yml gốc —
# train.py chỉ lưu ảnh so sánh render-vs-GT (save_images_test/) + log
# TensorBoard khi torch.utils.tensorboard.SummaryWriter import được
# (train.py:11-15, TENSORBOARD_FOUND). Thiếu package này → không có gì để
# visualize sau khi train xong, chỉ có số PSNR/SSIM/LPIPS dạng text. Cài
# riêng ở đây (không sửa environment.yml gốc, giữ nguyên trung thực với
# bản chị Duyên export).
conda run -n redigs pip install tensorboard

echo ""
echo "############################################################"
echo "# STEP 3 — Tạo conda env 'roma_v1' (dense init, Python 3.12 + torch 2.6+cu124)"
echo "#          Cần cho scripts/preprocess.py. Mất khoảng 5-10 phút."
echo "############################################################"
if conda env list | grep -q "^roma_v1 " && ! env_is_functional roma_v1; then
    echo "  env 'roma_v1' tồn tại nhưng KHÔNG hoàn chỉnh (thiếu torch) — xoá và tạo lại."
    conda env remove -n roma_v1 -y
fi
if ! conda env list | grep -q "^roma_v1 "; then
    loosen_env_yml environment_roma.yml /tmp/environment_roma_kaggle.yml \
        https://download.pytorch.org/whl/cu124
    conda env create --file /tmp/environment_roma_kaggle.yml
fi
conda run -n roma_v1 python --version   # kỳ vọng: Python 3.12.x
conda run -n roma_v1 python -c "import torch; print('torch:', torch.__version__, '| cuda available:', torch.cuda.is_available())"

echo ""
echo "############################################################"
echo "# STEP 4 — Build 2 CUDA submodule (rasterizer + simple-knn) trong env redigs"
echo "#          Đây là bước hay lỗi nhất khi đổi môi trường — nếu lỗi ở"
echo "#          đây, thường do nvcc không tìm thấy hoặc kiến trúc GPU"
echo "#          không khớp (xem STEP 5 nếu cần set TORCH_CUDA_ARCH_LIST)."
echo "############################################################"
# [CRSGaussian Phase 26] environment.yml KHÔNG có gói nào cung cấp trình
# biên dịch nvcc (chỉ có cuda-cudart = runtime library). Trên server gốc,
# nvcc có sẵn ở cấp hệ thống (cài ngoài conda, environment.yml không thấy
# được) — giả định vô hình, không đảm bảo đúng trên Kaggle. Kiểm tra chủ
# động trước khi build, tự cài cuda-nvcc khớp CUDA 12.1 (đúng version
# torch==2.1.0+cu121 đã build) nếu thiếu — tránh lỗi biên dịch mù mờ.
if ! conda run -n redigs which nvcc >/dev/null 2>&1; then
    echo "  nvcc không có sẵn — cài cuda-nvcc=12.1 vào env redigs (khớp torch cu121)."
    conda install -n redigs -y -c nvidia cuda-nvcc=12.1
fi
conda run -n redigs nvcc --version

# [ROLLBACK] Ép GCC 12 qua conda-forge làm CẢ 2 submodule fail (trước đó
# rasterizer build được, sau khi ép GCC 12 thì cả 2 đều fail) — compiler
# không phải nguyên nhân, bỏ bước này, dùng lại compiler mặc định của
# Kaggle (thứ đã giúp rasterizer build được ở lần chạy trước).
#
# Build 2 submodule với pip verbose (-v) — pip mặc định NUỐT output chi
# tiết của gcc/nvcc khi build fail (chỉ in "Failed building wheel", không
# có dòng lỗi compiler thật) → verbose để log thực sự chứa lỗi gốc, LƯU
# LOG RIÊNG + tự trích khi fail.
set +e
conda run -n redigs pip install -v submodules/diff-gaussian-rasterization-confidence 2>&1 | tee /tmp/build_rasterizer.log
RC1=${PIPESTATUS[0]}
conda run -n redigs pip install -v submodules/simple-knn 2>&1 | tee /tmp/build_simpleknn.log
RC2=${PIPESTATUS[0]}
set -e
if [ "${RC1}" -ne 0 ] || [ "${RC2}" -ne 0 ]; then
    echo ""
    echo "############################################################"
    echo "# LỖI BUILD CUDA SUBMODULE — trích dòng error/fatal quan trọng:"
    echo "############################################################"
    if [ "${RC1}" -ne 0 ]; then
        echo "--- rasterizer (RC=${RC1}) — 60 dòng quanh lỗi đầu tiên ---"
        grep -n -iE ": error|error:|fatal|unsupported gnu|undefined reference|no such file" /tmp/build_rasterizer.log | head -20
        echo "..."
        grep -n -iE ": error|error:|fatal|unsupported gnu|undefined reference|no such file" /tmp/build_rasterizer.log -A5 -B5 | tail -60
    fi
    if [ "${RC2}" -ne 0 ]; then
        echo "--- simple-knn (RC=${RC2}) — 60 dòng quanh lỗi đầu tiên ---"
        grep -n -iE ": error|error:|fatal|unsupported gnu|undefined reference|no such file" /tmp/build_simpleknn.log | head -20
        echo "..."
        grep -n -iE ": error|error:|fatal|unsupported gnu|undefined reference|no such file" /tmp/build_simpleknn.log -A5 -B5 | tail -60
    fi
    echo "############################################################"
    echo "# Log ĐẦY ĐỦ (nếu cần xem thêm): /tmp/build_rasterizer.log /tmp/build_simpleknn.log"
    echo "############################################################"
    exit 1
fi

echo ""
echo "############################################################"
echo "# STEP 5 — [CHỈ CHẠY NẾU STEP 4 LỖI với 'no kernel image is"
echo "#          available' hoặc tương tự] Set kiến trúc GPU tường minh"
echo "#          rồi build lại. T4=7.5, P100=6.0 — kiểm tra GPU thật đang"
echo "#          chạy qua nvidia-smi ở STEP 0 (dòng tên GPU) trước khi set."
echo "############################################################"
echo "  # Ví dụ (KHÔNG chạy mặc định, chỉ tham khảo nếu cần):"
echo "  # export TORCH_CUDA_ARCH_LIST='7.5'   # nếu Kaggle cấp T4"
echo "  # conda run -n redigs pip install --force-reinstall --no-deps submodules/diff-gaussian-rasterization-confidence"
echo "  # conda run -n redigs pip install --force-reinstall --no-deps submodules/simple-knn"

echo ""
echo "############################################################"
echo "# STEP 6 — Verify import thành công (chưa cần data thật)"
echo "############################################################"
conda run -n redigs python -c "
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from simple_knn._C import distCUDA2
import torch
print('rasterizer + simple-knn import OK')
print('torch.cuda device:', torch.cuda.get_device_name(0))
"

echo ""
echo "############################################################"
echo "# STEP 7 — COPY data LLFF từ /kaggle/input (read-only) sang chỗ"
echo "#          ghi được. Tự dò tìm thư mục dataset đã attach — không"
echo "#          cần bạn tự gõ tên chính xác."
echo "#          Kỳ vọng mỗi scene có: images/, sparse/0/*.bin,"
echo "#          poses_bounds.npy (đã verify cấu trúc này đủ chuẩn)."
echo "############################################################"
DATA_DST="data/nerf_llff_data"
EXPECTED_SCENES="fern flower fortress horns leaves orchids room trex"

if [ -d "$DATA_DST" ] && [ -f "$DATA_DST/fern/poses_bounds.npy" ]; then
    echo "  data/nerf_llff_data đã tồn tại + có poses_bounds.npy — bỏ qua copy."
else
    # Dò trong /kaggle/input mọi thư mục con có sẵn 'fern/poses_bounds.npy'
    # hoặc 'nerf_llff_data/fern/poses_bounds.npy' — không đoán tên dataset.
    SRC=""
    # maxdepth 7: một số Kaggle notebook mount dataset lồng sâu kiểu
    # /kaggle/input/datasets/<owner>/<slug>/nerf_llff_data/<scene>/ (đã xác
    # nhận thực tế, không phải /kaggle/input/<slug>/ phẳng như tài liệu
    # Kaggle mô tả) — quét sâu hơn thay vì đoán cấu trúc cố định.
    for cand in $(find /kaggle/input -maxdepth 7 -type d -name "fern" 2>/dev/null); do
        if [ -f "$cand/poses_bounds.npy" ]; then
            SRC=$(dirname "$cand")
            break
        fi
    done
    if [ -z "$SRC" ]; then
        echo "LỖI: không tìm thấy scene 'fern' có poses_bounds.npy dưới /kaggle/input."
        echo "  Kiểm tra lại đã Add Input đúng dataset chưa (Settings > Add Input)."
        find /kaggle/input -maxdepth 3 -type d 2>/dev/null
        exit 1
    fi
    echo "  Tìm thấy nguồn data tại: $SRC"
    mkdir -p "$DATA_DST"
    for sc in $EXPECTED_SCENES; do
        if [ -d "$SRC/$sc" ]; then
            echo "  copy $sc ..."
            cp -r "$SRC/$sc" "$DATA_DST/$sc"
        else
            echo "  CẢNH BÁO: thiếu scene '$sc' trong dataset nguồn — bỏ qua."
        fi
    done
fi

echo "  Kết quả:"
for sc in $EXPECTED_SCENES; do
    if [ -f "$DATA_DST/$sc/poses_bounds.npy" ]; then
        echo "    $sc: OK (poses_bounds.npy có mặt)"
    else
        echo "    $sc: THIẾU poses_bounds.npy"
    fi
done

echo ""
echo "############################################################"
echo "# STEP 8 — Clone Depth-Anything-V2 (sibling repo) + tải checkpoint ViT-L"
echo "#          train.py cần cho depth prior (utils/depth/depth_model.py:"
echo "#          'from depth_anything_v2.dpt import DepthAnythingV2'). Theo"
echo "#          README: clone làm sibling directory (../Depth-Anything-V2"
echo "#          tính từ ReDI-GS), checkpoint ~1.3GB tải từ HuggingFace."
echo "############################################################"
DAV2_DIR="../Depth-Anything-V2"
if [ ! -d "$DAV2_DIR/depth_anything_v2" ]; then
    echo "  Clone Depth-Anything-V2..."
    git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2.git "$DAV2_DIR"
else
    echo "  Depth-Anything-V2 đã có sẵn — bỏ qua clone."
fi

DAV2_CKPT="$DAV2_DIR/checkpoints/depth_anything_v2_vitl.pth"
if [ ! -f "$DAV2_CKPT" ]; then
    echo "  Tải checkpoint ViT-L (~1.3GB) từ HuggingFace..."
    mkdir -p "$DAV2_DIR/checkpoints"
    wget -q --show-progress -O "$DAV2_CKPT" \
        https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
else
    echo "  Checkpoint ViT-L đã có sẵn — bỏ qua tải."
fi
ls -lh "$DAV2_CKPT"

echo ""
echo "############################################################"
echo "# SETUP XONG."
echo "# Bước tiếp theo (dùng scripts/kaggle_run_all.sh để tự động hết,"
echo "# hoặc chạy tay từng bước):"
echo "#   1. conda run -n roma_v1 bash scripts/preprocess_all.sh   (dense init)"
echo "#   2. conda run -n redigs python scripts/place_init.py     (swap fused.ply)"
echo "#   3. conda run -n redigs bash scripts/p26_verify.sh        (smoke test)"
echo "#   4. conda run -n redigs bash scripts/p26_ablation_run.sh  (ablation thật)"
echo "# Miniconda cài ở \$HOME/miniconda3 — nếu Kaggle không giữ persistent"
echo "# storage cho notebook, phải chạy lại kaggle_setup.sh mỗi session mới."
echo "############################################################"
