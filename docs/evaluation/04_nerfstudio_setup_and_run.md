# Nerfstudio — Setup, Run & Compare Guide

> Hướng dẫn full pipeline: setup Nerfstudio trên Linux server → train 4 scene LLFF 3-view → compare với CRSGaussian Phase 22 trên 4 chiều: **tốc độ / định lượng / định tính / demo visual**.
>
> **Server**: Linux GPU (verified trên `aiserver.daotao.ai`, 2× NVIDIA RTX A4000, CUDA 12.4).
> **Local**: Windows + VS Code Remote SSH.
> **Verified working**: 2026-06-01.

---

## Phần 1 — Setup môi trường (lần đầu, ~30 phút)

### Bước 0 — Pre-check trên server

```bash
nvidia-smi
ls /usr/local/cuda
nvcc --version  # Nếu nvcc not found, sẽ set ở Bước 4
```

### Bước 1 — Tạo conda env riêng

```bash
conda create -n nerfstudio python=3.10 -y
conda activate nerfstudio

python --version  # Python 3.10.x
which python
```

### Bước 2 — Install PyTorch khớp CUDA 12.4

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

python -c "import torch; print('Torch:', torch.__version__, '| CUDA:', torch.version.cuda, '| OK:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"
# Expected: Torch: 2.5.1+cu124 | CUDA: 12.4 | OK: True | GPUs: 2
```

### Bước 3 — Fix NumPy + setuptools conflict

```bash
pip install "numpy<2"
pip install "setuptools<81" --force-reinstall

python -c "import pkg_resources; print('pkg_resources OK')"
```

### Bước 4 — Persist CUDA_HOME cho env

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/cuda.sh << 'EOF'
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
EOF

conda deactivate
conda activate nerfstudio

echo "CUDA_HOME=$CUDA_HOME"
which nvcc
nvcc --version  # Expected: release 12.4
```

### Bước 5 — Install Nerfstudio + ffmpeg

```bash
conda install -c conda-forge ffmpeg -y
ffmpeg -version | head -1

pip install nerfstudio
ns-train --help | head -5
```

### Bước 6 — Tạo working dir

```bash
cd /home/aidev/workspace/representation-3d/duyen/
mkdir -p nerfstudio/outputs nerfstudio/logs
cd nerfstudio
```

### ⚠️ Lưu ý dataset LLFF format không chuẩn Nerfstudio

CRSGaussian/CoR-GS đặt COLMAP files trong `triangulated/` thay vì `sparse/0/`. Khi chạy `ns-train splatfacto`:
```
colmap --colmap-path triangulated --images-path images
```

---

## Phần 2 — Smoke test 1 scene (verify setup OK)

### Bước 7 — Train fern 5k iter (~7-10 phút lần đầu)

```bash
cd /home/aidev/workspace/representation-3d/duyen/nerfstudio

CUDA_VISIBLE_DEVICES=0 ns-train splatfacto \
    --data /home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data/fern/3_views/ \
    --output-dir outputs \
    --max-num-iterations 5000 \
    --steps-per-eval-all-images 5000 \
    --vis tensorboard \
    --pipeline.model.use-absgrad False \
    colmap \
    --colmap-path triangulated \
    --images-path images \
    --eval-mode interval \
    --eval-interval 8 \
    2>&1 | tee logs/fern_smoke.log
```

Khi prompt downscale `[y/n]`: gõ **y** + Enter.

Lần đầu: im lìm 1-3 phút khi gsplat JIT compile 17 kernels. Sau đó progress bar.

```bash
# Eval explicit
RUN=$(ls -td outputs/3_views/splatfacto/* | head -1)
ns-eval --load-config $RUN/config.yml --output-path $RUN/eval.json
cat $RUN/eval.json
```

Expected fern 5k iter: PSNR ~17 dB, SSIM ~0.48, LPIPS ~0.38.

---

## Phần 3 — Mở viser viewer xem 3D

### Bước 8 — Khởi động viser

```bash
RUN=$(ls -td outputs/3_views/splatfacto/* | head -1)
ns-viewer --load-config $RUN/config.yml --viewer.websocket-port 7007
```

Giữ terminal SSH mở.

### Bước 9 — Forward port về Windows local

**Cách 1 — VS Code Port Forwarding (recommend):**
1. VS Code Remote SSH connected → panel dưới → tab "**Ports**"
2. Add Port → nhập `7007` → Enter
3. Right-click port → "Open in Browser"

**Cách 2 — SSH command line:**
```bash
# Windows PowerShell hoặc local zsh
ssh -L 7007:localhost:7007 aidev@aiserver.daotao.ai
```

### Bước 10 — Browser local + tương tác

URL: `http://localhost:7007` (Chrome/Edge 134+)

Mouse:
- Drag chuột trái: xoay
- Scroll: zoom
- Right-drag: pan

Panel phải:
- Tab **Control**: Max res, output type, show train cams
- Tab **Render**: Add Keyframe → camera path → MP4
- Tab **Export**: download PLY

**Max res setting:**
| Tình huống | Max res |
|------------|---------|
| Live xoay tương tác | 1024 |
| Screenshot tĩnh | 1920-2048 |
| Render video MP4 | 1920×1080 |

→ Max res cao = chi tiết render nét hơn, **KHÔNG** làm model tốt hơn (model fixed sau train).

### Bước 11 — Export PLY

```bash
ns-export gaussian-splat --load-config $RUN/config.yml --output-dir $RUN/export/
ls -lh $RUN/export/splat.ply  # ~10-50 MB
```

PLY chuẩn 3DGS → upload [superspl.at/editor](https://superspl.at/editor) hoặc dùng cho compare.

---

## Phần 4 — Chạy 4 scene benchmark 10k iter

### Bước 12 — Pre-generate downscaled images (FIX EOFError)

**Vấn đề**: Khi chạy script nohup background, nerfstudio prompt `[y/n]` để downscale → stdin đóng → EOFError. Phải tạo `images_4/` trước.

```bash
cd /home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data

for SC in fern horns fortress flower; do
    SRC=$SC/3_views
    if [ -d "$SRC/images_4" ] && [ -n "$(ls $SRC/images_4 2>/dev/null)" ]; then
        echo "✓ $SC: images_4 exists, skip"
        continue
    fi
    echo "Generating images_4 + images_2 for $SC..."
    mkdir -p $SRC/images_4 $SRC/images_2
    for IMG in $SRC/images/*.JPG; do
        [ -f "$IMG" ] || continue
        BASE=$(basename $IMG)
        ffmpeg -y -noautorotate -i $IMG -q:v 2 -vf scale=iw/4:ih/4 $SRC/images_4/$BASE 2>/dev/null
        ffmpeg -y -noautorotate -i $IMG -q:v 2 -vf scale=iw/2:ih/2 $SRC/images_2/$BASE 2>/dev/null
    done
    echo "✓ $SC done"
done

# Verify
for SC in fern horns fortress flower; do
    N=$(ls $SC/3_views/images_4/*.JPG 2>/dev/null | wc -l)
    echo "$SC: $N images in images_4/"
done
```

### Bước 13 — Tạo script 4 scene (FIX CUDA_HOME)

**Vấn đề**: nohup background không inherit `CUDA_HOME` từ activate.d → gsplat fail "No CUDA toolkit". Phải export CUDA_HOME inline trong script.

```bash
cd /home/aidev/workspace/representation-3d/duyen/nerfstudio

cat > run_4scene_10k.sh << 'EOF'
#!/bin/bash
# 4 scene LLFF 3-view, 10k iter, vanilla splatfacto, 2 GPU split
# Output: outputs_4scene_10k/<scene>/ + logs/4scene_10k/<scene>.log

# ── Critical: export CUDA_HOME cho gsplat JIT (nohup không inherit activate.d) ──
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

DATA_ROOT=/home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data
OUT_ROOT=/home/aidev/workspace/representation-3d/duyen/nerfstudio/outputs_4scene_10k
LOG_DIR=/home/aidev/workspace/representation-3d/duyen/nerfstudio/logs/4scene_10k
mkdir -p $OUT_ROOT $LOG_DIR

GPU0_SCENES=(fern horns)
GPU1_SCENES=(fortress flower)

run_scene() {
    local GPU=$1
    local SC=$2
    local LOG=$LOG_DIR/$SC.log

    # Skip if already done (resume-safe)
    if [ -f "$LOG" ] && grep -q "Training Finished" "$LOG" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] SKIP $SC (already done)"
        return 0
    fi

    echo "[$(date +%H:%M:%S)] START $SC on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
    ns-train splatfacto \
        --data $DATA_ROOT/$SC/3_views/ \
        --output-dir $OUT_ROOT \
        --experiment-name $SC \
        --max-num-iterations 10000 \
        --steps-per-eval-all-images 10000 \
        --vis tensorboard \
        --pipeline.model.use-absgrad False \
        colmap \
        --colmap-path triangulated \
        --images-path images \
        --eval-mode interval \
        --eval-interval 8 \
        2>&1 | tee $LOG
    echo "[$(date +%H:%M:%S)] DONE $SC"
}

# Parallel 2 GPUs
(
    for SC in "${GPU0_SCENES[@]}"; do
        run_scene 0 $SC
    done
) &
PID0=$!

(
    for SC in "${GPU1_SCENES[@]}"; do
        run_scene 1 $SC
    done
) &
PID1=$!

wait $PID0 $PID1
echo "[$(date +%H:%M:%S)] All 4 scenes done"
echo ""
echo "==== SUMMARY PSNR ===="
for SC in "${GPU0_SCENES[@]}" "${GPU1_SCENES[@]}"; do
    RUN=$(ls -td $OUT_ROOT/$SC/splatfacto/* 2>/dev/null | head -1)
    if [ -n "$RUN" ]; then
        ns-eval --load-config $RUN/config.yml --output-path $RUN/eval.json 2>&1 | tail -3
        PSNR=$(python3 -c "import json; print(json.load(open('$RUN/eval.json'))['results']['psnr'])" 2>/dev/null)
        echo "$SC: PSNR=$PSNR"
    fi
done
EOF

chmod +x run_4scene_10k.sh
```

Khác biệt vs script ban đầu:
- **Bỏ `set -e`** → 1 scene fail không kill toàn bộ
- **Export CUDA_HOME 2 lần**: top script + inline mỗi `ns-train`

### Bước 14 — Chạy nohup background

```bash
# Cleanup output cũ nếu có run fail trước đó
rm -rf /home/aidev/workspace/representation-3d/duyen/nerfstudio/outputs_4scene_10k
rm -rf /home/aidev/workspace/representation-3d/duyen/nerfstudio/logs/4scene_10k

# Chạy background
cd /home/aidev/workspace/representation-3d/duyen/nerfstudio
nohup ./run_4scene_10k.sh > /tmp/4scene_master.log 2>&1 &
echo "PID: $!"
disown

# Theo dõi tiến độ (Ctrl+C thoát tail, không kill script)
tail -f /tmp/4scene_master.log

# Hoặc đếm scene đã xong
ls logs/4scene_10k/*.log 2>/dev/null | wc -l
```

⏱ Tổng ~30-40 phút trên 2 GPU.

### Bước 15 — Export PLY 4 scene (sau khi script xong)

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

for SC in fern horns fortress flower; do
    RUN=$(ls -td /home/aidev/workspace/representation-3d/duyen/nerfstudio/outputs_4scene_10k/$SC/splatfacto/* | head -1)
    if [ -n "$RUN" ]; then
        ns-export gaussian-splat --load-config $RUN/config.yml --output-dir $RUN/export/
        echo "✓ $SC: $RUN/export/splat.ply"
    fi
done
```

### Bước 16 — Mở viser cho 1 scene defense demo

```bash
RUN=$(ls -td outputs_4scene_10k/horns/splatfacto/* | head -1)
ns-viewer --load-config $RUN/config.yml --viewer.websocket-port 7007
```

Forward port 7007 → browser local.

---

## Phần 5 — So sánh đa chiều splatfacto vs CRSGaussian Phase 22

> Phần này là **defense backbone**: 4 loại evidence cùng lúc.

### 5.1 — Tốc độ training (wall-clock)

Extract train time từ log:

```bash
# splatfacto 4 scene
for SC in fern horns fortress flower; do
    LOG=/home/aidev/workspace/representation-3d/duyen/nerfstudio/logs/4scene_10k/$SC.log
    if [ -f "$LOG" ]; then
        # Lấy 2 timestamp START và DONE từ master log
        START=$(grep "START $SC" /tmp/4scene_master.log | head -1 | awk '{print $1}')
        DONE=$(grep "DONE $SC" /tmp/4scene_master.log | head -1 | awk '{print $1}')
        echo "$SC: $START → $DONE"
    fi
done

# CRSGaussian Phase 22 train time
grep "TIMING.*Total training" /home/aidev/workspace/representation-3d/duyen/CoR-GS/output/p22_pilot_v1/A3_seed42_*/.../log* 2>/dev/null | head -4
```

**Bảng so sánh tốc độ** (điền sau khi chạy):

| Scene | Splatfacto 10k iter | CRSGaussian P22 10k iter | Δ time |
|-------|----------------------|---------------------------|--------|
| fern | TBD min | TBD min | TBD |
| horns | TBD min | TBD min | TBD |
| fortress | TBD min | TBD min | TBD |
| flower | TBD min | TBD min | TBD |
| **Average** | TBD | TBD | TBD |

Caveat: cùng GPU (A4000) + cùng iter (10k) mới fair. Phase 22 dùng RoMa init dense → có thể chậm hơn vì N_gauss cao hơn.

### 5.2 — Định lượng PSNR / SSIM / LPIPS

```bash
# splatfacto 4 scene từ eval.json
echo "Method,Scene,PSNR,SSIM,LPIPS"
for SC in fern horns fortress flower; do
    RUN=$(ls -td /home/aidev/workspace/representation-3d/duyen/nerfstudio/outputs_4scene_10k/$SC/splatfacto/* | head -1)
    if [ -f "$RUN/eval.json" ]; then
        python3 -c "
import json
d = json.load(open('$RUN/eval.json'))['results']
print(f'splatfacto,$SC,{d[\"psnr\"]:.3f},{d[\"ssim\"]:.4f},{d[\"lpips\"]:.4f}')
"
    fi
done

# CRSGaussian Phase 22 (từ logs P22)
# Nếu logs còn trên server:
for SC in fern horns fortress flower; do
    LOG=/home/aidev/workspace/representation-3d/duyen/CoR-GS/logs/p22_pilot_v1/A3_seed42_$SC.log
    if [ -f "$LOG" ]; then
        grep -E "Best test PSNR|SSIM|LPIPS" "$LOG" | head -3
    fi
done
```

**Bảng định lượng** (điền sau khi chạy):

| Scene | splatfacto vanilla 10k | CRSGaussian P22 10k | Δ PSNR |
|-------|------------------------|----------------------|--------|
| | PSNR / SSIM / LPIPS | PSNR / SSIM / LPIPS | |
| fern | TBD / TBD / TBD | TBD / TBD / TBD | TBD |
| horns | TBD / TBD / TBD | TBD / TBD / TBD | TBD |
| fortress | TBD / TBD / TBD | TBD / TBD / TBD | TBD |
| flower | TBD / TBD / TBD | TBD / TBD / TBD | TBD |
| **Average** | TBD | TBD | TBD |

### 5.3 — Định tính (qualitative side-by-side)

#### Cách 1: Render dataset views (recommend)

```bash
# splatfacto render 4 scene test views
for SC in fern horns fortress flower; do
    RUN=$(ls -td /home/aidev/workspace/representation-3d/duyen/nerfstudio/outputs_4scene_10k/$SC/splatfacto/* | head -1)
    ns-render dataset \
        --load-config $RUN/config.yml \
        --output-path $RUN/dataset_render \
        --split train+test
done
```

→ Output PNG ở `$RUN/dataset_render/test/` — render từ góc test view (novel view) so với GT.

#### Cách 2: Cùng góc nhìn cho cả 2 method

Để defense slide có ảnh side-by-side **cùng góc nhìn**:

1. Mở viser cho splatfacto (Bước 16)
2. Xoay đến góc đẹp → tab **Render** → Add Keyframe → Save camera path JSON
3. Mở viser cho CRSGaussian P22 (cần load thêm config, xem 5.4 dưới)
4. Load cùng JSON → render cùng góc → screenshot

**Slide layout đề xuất**:
```
+----------------+----------------+----------------+
|  GT (ảnh test) | splatfacto     | CRSGaussian P22|
|                |  PSNR 17.8     |  PSNR 22.5     |
+----------------+----------------+----------------+
       (lặp cho 4 scene: fern, horns, fortress, flower)
```

### 5.4 — Demo visual (viser / SuperSplat / MP4)

3 cách show 3D sống cho hội đồng, từ dễ đến chuyên nghiệp:

#### Cách A — Viser interactive (recommend cho live defense)
```bash
# Splatfacto
RUN_NS=$(ls -td outputs_4scene_10k/horns/splatfacto/* | head -1)
ns-viewer --load-config $RUN_NS/config.yml --viewer.websocket-port 7007

# CRSGaussian P22 (terminal khác, port khác)
# (Cần convert PLY P22 → nerfstudio format hoặc dùng SuperSplat thay)
```

Forward port 7007 → browser → hội đồng xoay scene.

#### Cách B — SuperSplat browser (đơn giản nhất)

1. Scp PLY về Windows local:
```powershell
# Splatfacto
scp aidev@aiserver.daotao.ai:/home/aidev/workspace/representation-3d/duyen/nerfstudio/outputs_4scene_10k/horns/splatfacto/*/export/splat.ply d:/horns_splatfacto.ply

# CRSGaussian P22
scp aidev@aiserver.daotao.ai:/home/aidev/workspace/representation-3d/duyen/CoR-GS/output/p22_pilot_v1/A3_seed42_horns/point_cloud/iteration_10000/point_cloud.ply d:/horns_p22.ply
```

2. Mở 2 tab browser [superspl.at/editor](https://superspl.at/editor), drag 2 PLY → so cạnh nhau

3. Hoặc upload + bấm "Publish" → có **link share** → mở trên laptop hội đồng

#### Cách C — Render MP4 flythrough

```bash
# Cần camera path JSON từ viser (Bước 10 tab Render)
ns-render camera-path \
    --load-config $RUN/config.yml \
    --camera-path-filename camera_path.json \
    --output-path $RUN/flythrough.mp4
```

→ MP4 1920×1080 embed slide PPT.

#### Cách D — Screenshot static cho slide

```bash
# Trong viser viewer, set Max res = 1920, xoay đến góc đẹp, F12 hoặc Snipping Tool
```

→ Paste vào slide grid 2×4 (2 method × 4 scene).

---

## Phần 6 — Defense narrative tổng hợp

Slide flow đề xuất:

| Slide | Content |
|-------|---------|
| 1 | "Bài toán: sparse-view 3D reconstruction 3 ảnh" — show 3 ảnh input |
| 2 | "Vanilla 3DGS production fail" — splatfacto qualitative + PSNR 17 dB |
| 3 | "CRSGaussian Phase 22 giải pháp" — 8 module: depth + CRS + opacity decay + RoMa init + ... |
| 4 | **Bảng định lượng** 4 scene (Phần 5.2) |
| 5 | **Bảng tốc độ** (Phần 5.1) — trade-off slower nhưng quality cao |
| 6 | **Qualitative grid** GT vs splatfacto vs CRSGaussian (Phần 5.3) |
| 7 | **Live demo viser** hoặc **SuperSplat link** — hội đồng tự xoay |
| 8 | Conclusion + future work |

---

## Common issues + fix

| Issue | Cause | Fix |
|-------|-------|-----|
| `pkg_resources not found` | setuptools 81+ removed | `pip install "setuptools<81" --force-reinstall` |
| `gsplat: No CUDA toolkit found` | CUDA_HOME chưa set | `export CUDA_HOME=/usr/local/cuda` |
| `ffmpeg: not found` | ffmpeg chưa cài env | `conda install -c conda-forge ffmpeg -y` |
| `EOFError: EOF when reading a line` (nohup) | Prompt downscale `y/n` không nhận stdin background | **Pre-generate `images_4/` trước** (Bước 12) |
| `gsplat fail trong script nohup` | nohup không inherit activate.d CUDA_HOME | **Export CUDA_HOME inline trong script** (Bước 13) |
| `images_4/xxx.JPG not found` | Folder downscale partial | `rm -rf <scene>/images_4 <scene>/images_2` rồi chạy lại |
| `AssertionError ns-render spiral` | spiral không support splatfacto | Dùng `ns-render dataset` hoặc viser camera-path |
| Browser localhost:7007 không load | Port forward chưa setup | VS Code Ports tab add 7007 |
| OOM khi train | GPU 0 share | `CUDA_VISIBLE_DEVICES=1` |

---

## Quick reference

```bash
# Activate
conda activate nerfstudio
export CUDA_HOME=/usr/local/cuda  # nếu chưa persist

# Pre-generate downscale (1 lần per scene)
# ... (xem Bước 12)

# Train 1 scene
CUDA_VISIBLE_DEVICES=0 ns-train splatfacto --data <path>/3_views/ \
    --output-dir outputs --max-num-iterations 10000 \
    --pipeline.model.use-absgrad False \
    colmap --colmap-path triangulated --images-path images \
    --eval-mode interval --eval-interval 8

# Eval
ns-eval --load-config <run>/config.yml --output-path <run>/eval.json

# Export PLY
ns-export gaussian-splat --load-config <run>/config.yml --output-dir <run>/export/

# View interactive
ns-viewer --load-config <run>/config.yml --viewer.websocket-port 7007

# Render dataset views (cho qualitative compare)
ns-render dataset --load-config <run>/config.yml --output-path <run>/dataset_render --split train+test

# Render MP4 từ camera path
ns-render camera-path --load-config <run>/config.yml \
    --camera-path-filename path.json --output-path flythrough.mp4
```

---

## Liên quan

- [01_real_world_comparison_plan.md](01_real_world_comparison_plan.md) — overall evaluation plan
- [02_brush_feasibility_assessment.md](02_brush_feasibility_assessment.md) — Brush alternative
- [03_nerfstudio_feasibility_assessment.md](03_nerfstudio_feasibility_assessment.md) — Nerfstudio plug-in feasibility
- CRSGaussian Phase 22 backbone: memory `project_phase22_roma_v1_pilot.md`
