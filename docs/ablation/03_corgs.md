# Baseline 03 — CoR-GS (ECCV 2024)

> Zhang et al., "CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization", ECCV 2024.
> Base codebase của CRSGaussian (ours xây trên CoR-GS). 2-field co-regularization + co-pruning.
> Init = COLMAP-MVS `3_views/dense/fused.ply` · 10k iter · -r8 · 3-view.

**Trạng thái (2026-06-08): ✅ DONE — reproduced AVG PSNR 20.110 = paper 20.11 (KHỚP CHÍNH XÁC).**

---

## 1. Method tóm tắt
- **2-field co-regularization**: train 2 Gaussian field song song, regularize chéo (point + rendering disagreement).
- **Co-pruning**: prune Gaussian mà 2 field bất đồng (Open3D nearest-neighbor).
- Init COLMAP-MVS, KHÔNG depth prior. Là **baseline ours xây lên** (ours thêm CRS + RoMa init + DropAnSH + ...).

Paper LLFF 3-view PSNR = **20.11** (ours đã cite ở fair 10k bucket).

---

## 2. Setup
- Repo riêng: `duyen/CoR-GS_goc` (CoR-GS gốc, KHÁC `duyen/CoR-GS` = codebase CRSGaussian).
- **Đã train sẵn** bởi user: `CoR-GS_goc/output/llff/<scene>/` — `resolution=8, n_views=3, eval`, iter 10000.
- Renders có sẵn `test/ours_10000/{gt,renders}` (renders kèm `_depth.png` — evaluator bỏ qua tự động vì chỉ duyệt theo tên file trong `gt/`).
- **KHÔNG cần train lại** → chỉ unified eval (`corgs/metrics.py`). Không tốn GPU training.

Lệnh eval: loop 8 scene `corgs python metrics.py -s x -m CoR-GS_goc/output/llff/<scene>`.

---

## 3. Kết quả (2026-06-08, -r8, 10k, unified eval)

| Scene | PSNR ↑ | SSIM ↑ | SSIM_sk ↑ | LPIPS ↓ | AVGE ↓ | N_gauss | FPS ↑ |
|-------|--------|--------|-----------|---------|--------|---------|-------|
| fern | 22.293 | 0.7450 | 0.7409 | 0.1959 | 0.0862 | 77,095 | _bench_ |
| flower | 19.862 | 0.6494 | 0.6431 | 0.2319 | 0.1327 | 83,047 | _bench_ |
| fortress | 23.461 | 0.7512 | 0.7375 | 0.1661 | 0.0730 | 51,525 | _bench_ |
| horns | 19.019 | 0.6890 | 0.6749 | 0.2490 | 0.1254 | 63,030 | _bench_ |
| leaves | 16.746 | 0.6201 | 0.6316 | 0.2141 | 0.1409 | 196,935 | _bench_ |
| orchids | 15.593 | 0.4952 | 0.4959 | 0.2669 | 0.1738 | 79,768 | _bench_ |
| room | 21.474 | 0.8506 | 0.8393 | 0.1540 | 0.0764 | 37,324 | _bench_ |
| trex | 22.430 | 0.8343 | 0.8295 | 0.1310 | 0.0681 | 56,712 | _bench_ |
| **Avg** | **20.110** | **0.704** | **0.699** | **0.201** | **0.110** | **80,680** | _bench_ |

> N_gauss đếm từ `point_cloud.ply` — CoR-GS 2-field, ply có thể chỉ là 1 field → **xấp xỉ**.
> Train time: models pre-trained, không đo (n/a). FPS: chờ `bench_fps_compare.sh` (thêm CoR-GS).

**Reproduction**: 20.110 vs paper **20.11** = **0.000** → khớp tuyệt đối.

---

## 4. So với ours (CoR-GS = base của CRSGaussian)

| Method | Iter | PSNR | SSIM | LPIPS | AVGE | N_gauss |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| CoR-GS (base) | 10k | 20.110 | 0.704 | 0.201 | 0.110 | 80,680 |
| **CRSGaussian (ours)** | 10k | **21.918** | **0.769** | **0.158** | **0.084** | **94,917** |
| **Δ ours − CoR-GS** | | **+1.808** | **+0.065** | **−0.043** | **−0.026** | +14,237 |

**Luận điểm mạnh nhất**: ours xây trực tiếp trên CoR-GS, **+1.808 PSNR** từ các đóng góp (RoMa dense-init + CRS +
DropAnSH + ...) ở **cùng 10k iter, cùng -r8**. Đây là ablation "base → full" rõ ràng nhất cho paper.

> CoR-GS dùng nhiều Gaussian hơn-or-ít-hơn tùy scene; ours +14k Gaussian nhưng +1.8 PSNR (RoMa init dày hơn).

---

## 5. Caveats
- CoR-GS = base codebase ours → so sánh này = "ablation cốt lõi" (đóng góp tổng của CRSGaussian).
- Cùng init COLMAP-MVS như FSGS native; ours đổi sang RoMa (Phase 22) = một phần của +1.808.
- Provenance: pre-trained `CoR-GS_goc/output/llff/`, eval unified 2026-06-08.
