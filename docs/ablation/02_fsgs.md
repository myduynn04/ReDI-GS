# Baseline 02 — FSGS (ICLR 2024)

> Zhu et al., "FSGS: Real-Time Few-Shot View Synthesis using Gaussian Splatting", ICLR 2024.
> [Paper](https://arxiv.org/abs/2312.00451) · [Project](https://zehaozhu.github.io/FSGS/) · base 3DGS.
> Mục tiêu: reproduce LLFF 3-view ở protocol ta (images_8 ≈ -r8, 10k) → unified eval + unified FPS.

**Trạng thái (2026-06-08): ✅ DONE — reproduced AVG PSNR 20.407 (paper ~20.43, −0.023 = trung thực).**

---

## 1. Method tóm tắt (cho paper)

Base 3DGS, **10k iter**, init = **COLMAP dense MVS fusion**, 3 cơ chế:
1. **Init dense**: COLMAP `patch_match_stereo` + `stereo_fusion` → `<scene>/3_views/dense/fused.ply` (~chục K điểm).
2. **Gaussian Unpooling**: densify hướng về novel view (proximity-guided).
3. **Monocular depth regularization**: **MiDaS DPT_Hybrid** depth → Pearson correlation loss trên (a) training views, (b) pseudo-views sample giữa các view (`sample_pseudo_interval`).

Paper LLFF 3-view PSNR ≈ **20.31** (số ours đã cite, 10k bucket). Đặc điểm: real-time render (paper nhấn FPS).

> Liên quan ta: ours cũng dùng depth-Pearson (mượn FSGS/LoopSparseGS). FSGS init COLMAP-MVS = giống
> CoR-GS baseline init (cùng `fused.ply`). Khác ta: ours Phase 22 dùng RoMa init (mạnh hơn).

---

## 2. Khác biệt setup so với Binocular3DGS (QUAN TRỌNG)

| Yếu tố | FSGS | Ghi chú rủi ro |
|---|---|---|
| Env | `environment.yml`: python 3.8, **torch 1.12.1 + cudatoolkit 11.6** | ⚠️ server chỉ có **CUDA 12.4** → build rasterizer torch-cu116 với nvcc 12.4 dễ FAIL. Cân nhắc env torch cu121 (như binocular3dgs) + cài deps tay |
| Rasterizer | `diff-gaussian-rasterization-confidence` (depth-enabled) + simple-knn | build `--no-build-isolation` |
| **Init ply** | `<scene>/3_views/dense/fused.ply` (COLMAP-MVS) | ⚠️ **8/8 scene đã bị swap = RoMa init của ta** (verified 2026-06-08, đều có `.colmap_mvs_backup`). FSGS phải dùng COLMAP-MVS gốc. **KHÔNG ghi đè data Phase 22** → dùng `prepare`: tạo data tree RIÊNG `FSGS/dataset_ablation/nerf_llff_data/` (symlink images/sparse/poses + copy MVS từ backup). Data gốc nguyên vẹn. |
| **Depth model** | **MiDaS DPT_Hybrid** qua `torch.hub` (load lúc import `depth_utils`, gọi mỗi camera trong `loadCam`) | cần internet 1 lần (~470MB DPT + repo MiDaS) hoặc torch-hub cache. `timm` bắt buộc |
| Resolution | đọc folder **`images_8`** (≈ ảnh//8) | ✓ khớp -r8 ta (cùng test pixel) |
| Iter / split | 10k · llffhold=8 + linspace(3) | ✓ khớp ta |
| Camera model | chỉ undistorted (PINHOLE/SIMPLE_PINHOLE/SIMPLE_RADIAL) | data ta OK |

---

## 3. Diagnostic server (chạy trước, paste output)

```bash
ROOT=/home/aidev/workspace/representation-3d/duyen
DATA=$ROOT/CoR-GS/data/nerf_llff_data

echo "== 1. FSGS repo + env =="
ls -d $ROOT/FSGS 2>/dev/null && echo "  FSGS repo OK" || echo "  FSGS repo MISSING (cần clone)"
conda env list | grep -i fsgs || echo "  env FSGS CHƯA có"

echo "== 2. fused.ply (init) — gốc COLMAP-MVS hay đã swap? =="
ls -la $DATA/fern/3_views/dense/fused.ply 2>/dev/null
ls -la $DATA/fern/3_views/dense/fused.ply.colmap_mvs_backup 2>/dev/null && echo "  -> CÓ backup = fused.ply hiện tại ĐÃ BỊ SWAP (cần restore cho FSGS)"

echo "== 3. images_8 có chưa =="
ls -d $DATA/fern/images_8 2>/dev/null && ls $DATA/fern/images_8 | wc -l

echo "== 4. submodule build trong env FSGS (nếu env tồn tại) =="
# đổi <fsgs_env> cho đúng:
# conda run -n <fsgs_env> python -c "import diff_gaussian_rasterization, simple_knn; print('built')" 2>&1 | tail -1

echo "== 5. MiDaS torch-hub cache =="
ls ~/.cache/torch/hub/ 2>/dev/null | grep -i midas || echo "  MiDaS chưa cache (cần internet 1 lần)"

echo "== 6. CUDA toolkit =="
ls -d /usr/local/cuda* 2>/dev/null
```

### Diễn giải kết quả → quyết định
| Mục | Nếu... | Thì... |
|---|---|---|
| 1 | env FSGS chưa có | tạo env (xem §4) |
| 2 | có `.colmap_mvs_backup` | **restore**: `cp fused.ply.colmap_mvs_backup fused.ply` cho cả 8 scene (FSGS cần MVS gốc) |
| 2 | KHÔNG có backup | fused.ply hiện = COLMAP-MVS gốc (an toàn) — HOẶC chưa từng có dense, phải gen (nặng) |
| 5 | MiDaS chưa cache | bật internet 1 lần khi chạy lần đầu |

---

## 4. Setup (điền sau diagnostic)

**Env** — 2 hướng (chọn theo §3 mục 6):
- **A (official)**: `conda env create -f environment.yml` (torch cu116) → cần `cuda-nvcc=11.6` (conda nvidia) để build.
- **B (khuyến nghị nếu chỉ có CUDA 12.4)**: tạo env torch cu121 (như binocular3dgs) + `pip install timm torchmetrics open3d opencv-python imageio matplotlib plyfile` + build rasterizer/simple-knn `--no-build-isolation` với `CUDA_HOME=/usr/local/cuda`.

(Lệnh chi tiết chốt sau diagnostic.)

---

## 5. Pipeline (sẽ vào `scripts/ablation/run_fsgs_llff.sh`)

Mỗi scene (KHÔNG có bước triangulate riêng — init = fused.ply có sẵn):
1. `python train.py -s $DATA/<scene> -m $OUT/<scene> --eval --n_views 3 --sample_pseudo_interval 1` (10k)
2. `python render.py -s $DATA/<scene> -m $OUT/<scene> --iteration 10000`
3. **unified eval**: `corgs/metrics.py -m $OUT/<scene>` (KHÔNG dùng FSGS metrics.py)
4. **unified FPS**: thêm vào `bench_fps_compare.sh` (render signature FSGS)
- Output: `FSGS/output/LLFF_ablation/<scene>/` · Log: `CoR-GS/logs/ablation/fsgs/`

---

## 6. Kết quả (điền sau khi chạy)

**Config**: images_8 (≈-r8), 10k iter, init COLMAP-MVS (data tree riêng), env `fsgs` (torch 2.4.1+cu121), MiDaS DPT_Hybrid. 1-seed.

| Scene | PSNR ↑ | SSIM ↑ | SSIM_sk ↑ | LPIPS ↓ | AVGE ↓ | N_gauss | Train(s) | FPS ↑ |
|-------|--------|--------|-----------|---------|--------|---------|----------|-------|
| fern | 21.896 | 0.7213 | 0.7166 | 0.2085 | 0.0919 | 184,370 | 2603 | 223.5 |
| flower | 20.383 | 0.6273 | 0.6226 | 0.2467 | 0.1243 | 522,152 | 2744 | 206.8 |
| fortress | 23.367 | 0.7258 | 0.7128 | 0.1694 | 0.0754 | 72,698 | 2768 | 290.2 |
| horns | 20.192 | 0.7157 | 0.7063 | 0.2289 | 0.1100 | 93,413 | 2492 | 259.5 |
| leaves | 17.629 | 0.6347 | 0.6498 | 0.2064 | 0.1293 | **1,051,981** | 3785 | 115.8 |
| orchids | 16.258 | 0.5115 | 0.5186 | 0.2596 | 0.1633 | 103,390 | 2439 | 322.5 |
| room | 21.720 | 0.8437 | 0.8319 | 0.1666 | 0.0777 | 47,110 | 2339 | 609.3 |
| trex | 21.810 | 0.8139 | 0.8082 | 0.1520 | 0.0765 | 184,770 | 2626 | 222.0 |
| **Avg** | **20.407** | **0.699** | **0.696** | **0.205** | **0.106** | **282,486** | **2724** | **281.2** |

**Reproduction**: 20.407 vs paper ~20.43 = **−0.023** → khớp.

**So với ours**: ours 21.918 = **+1.511 PSNR** vs FSGS, **SSIM/LPIPS/AVGE đều tốt hơn**, và **ít Gaussian hơn ~3×**
(ours 94,917 vs FSGS **282,486**) + **train ~4× nhanh** (676 vs 2724 s/scene).
- **FPS (unified)**: FSGS **281.2** > ours 175.4 → FSGS render nhanh hơn (dù nhiều Gaussian hơn) vì render path nhẹ
  (không tính confidence/depth như rasterizer ours). Cả hai real-time. **Infer cũng là trục FSGS nhỉnh hơn ours** (giống Binocular).
- FSGS densify khổng lồ (leaves **1.05M** Gaussian → OOM nếu GPU <10GB).

> Caveat: FSGS chậm-train + nặng do `sample_pseudo_interval=1` + Gaussian Unpooling. Train time đo trên GPU shared (~45 phút/scene).

---

## 7. Caveats cho paper
- FSGS native res = `images_8` (= -r8 ta) → so trực tiếp được.
- Init COLMAP-MVS (phải restore gốc, KHÔNG để dính RoMa/PDCNet+ của ta).
- MiDaS DPT_Hybrid = depth prior ngoài (giống ta dùng DAV2) → ghi rõ "external depth prior" khi so.
