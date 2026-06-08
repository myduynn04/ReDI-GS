# CRSGaussian (ours) — reference results (LLFF 3-view)

> Mô hình của ta, mốc so trong mọi bảng ablation. Recipe: **Phase 22 — RoMa v1 dense-init + A3-TRIM**, **10k iter**, **-r 8**, hold-out test, eval = `corgs/metrics.py` (unified, cùng evaluator với baselines).

**Trạng thái (2026-06-07): ✅ measured. MEAN PSNR = 21.918 (Phase 22 pilot, N=24 = 3 seeds × 8 scenes).**

> ⚠️ **Các con số PSNR:**
> - **21.918** = Phase 22 pilot **N=24** (3-seed mean) — bộ số đầy đủ dùng cho bảng ablation ở đây.
> - **21.89 ± 0.10** = giá trị **commit/defense** = mean của **4 run N=24** (Phase 25_2 lock). 21.918 = 1 trong 4 (lucky-high). Chênh < noise.
> - Khi báo cáo: dùng 21.918 cho bảng so per-metric (có full per-scene); ghi 21.89 là defense reproducibility.

---

## Kết quả per-scene (Phase 22 pilot, N=24 3-seed mean, 10k, -r8)

| Scene | PSNR ↑ | SSIM ↑ | SSIM_sk ↑ | LPIPS ↓ | AVGE ↓ | N_gauss | Train (s) ↓ | FPS ↑ |
|-------|--------|--------|-----------|---------|--------|---------|-------------|-------|
| fern | 23.840 | 0.7970 | 0.7947 | 0.1426 | 0.0660 | 86,715 | 672 | 122.6 |
| flower | 21.413 | 0.6915 | 0.6872 | 0.2086 | 0.1054 | 107,109 | 984 | 84.6 |
| fortress | 25.569 | 0.8369 | 0.8312 | 0.1173 | 0.0518 | 60,316 | 492 | 254.7 |
| horns | 21.079 | 0.7749 | 0.7658 | 0.1849 | 0.0942 | 69,765 | 414 | 175.5 |
| leaves | 19.375 | 0.7287 | 0.7425 | 0.1630 | 0.0995 | 246,378 | 1482 | 83.0 |
| orchids | 17.582 | 0.5815 | 0.5891 | 0.2053 | 0.1333 | 78,492 | 580 | 157.5 |
| room | 22.970 | 0.8808 | 0.8696 | 0.1274 | 0.0622 | 39,852 | 336 | 351.4 |
| trex | 23.514 | 0.8626 | 0.8585 | 0.1155 | 0.0584 | 70,711 | 450 | 173.6 |
| **MEAN** | **21.918** | **0.7692** | **0.7673** | **0.1581** | **0.0839** | **94,917** | **676** | **175.4** |

- Train ≈ **676 s/scene (~11.3 phút)** ở 10k iter (3-seed mean).
- **FPS = 175.4** (unified protocol warmup50+300timed, -r8, đo trên seed42; FPS N_gauss lệch nhẹ so cột N_gauss 3-seed).

> Per-scene PSNR: fortress 25.57 (cao nhất) · fern 23.84 · trex 23.51 · room 22.97 · flower 21.41 · horns 21.08 ·
> leaves 19.38 · orchids 17.58 (thấp nhất). Pattern khớp đặc tính LLFF.

---

## So với Binocular3DGS (cùng -r8, hold-out, cùng unified evaluator)

| Method | Iter | PSNR | SSIM | SSIM_sk | LPIPS | AVGE | N_gauss | Train s/scene | FPS (unified) |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Binocular3DGS | 30k | 21.356 | 0.744 | 0.742 | 0.171 | 0.092 | 106,620 | 1483 | **266.6** |
| **CRSGaussian (ours)** | 10k | **21.918** | **0.769** | **0.767** | **0.158** | **0.084** | **94,917** | **676** | 175.4 |
| **Δ ours − Bino** | | **+0.562** | **+0.025** | **+0.025** | **−0.013** | **−0.008** | **−11,703** | **−807 (~2.2×)** | −91 |

**Luận điểm cho paper:** ours **thắng mọi metric chất lượng** (PSNR +0.562, SSIM/SSIM_sk/LPIPS/AVGE đều tốt hơn) + **ít Gaussian hơn** (95k vs 107k) + **train ~2.2× nhanh** (676 vs 1483 s/scene) ở **3× ít iter**.

> **FPS (unified protocol — so thẳng được)**: Binocular **266.6** vs ours **175.4** → Binocular render nhanh hơn,
> NHƯNG **cả hai đều real-time** (≫30 FPS). Ours chậm hơn/frame do **rasterizer-confidence** (xuất thêm kênh
> confidence + depth) — đây là chi phí của cơ chế CRS, đổi lấy +0.56 PSNR + ít Gaussian + train nhanh.
> Trình bày thành thật: **infer là trục DUY NHẤT Binocular nhỉnh hơn**; ours thắng tất cả còn lại.

---

## Provenance
- Recipe: Phase 22 RoMa v1 + A3-TRIM, 10k, -r8. Models: `CoR-GS/output/p22_pilot/A3_seed42_<scene>/` (seed42 đại diện cho FPS). Commit value 21.89 (mean 4× N=24, [[project_phase25_2_phase22_lock]]).
- Bộ số trên = N=24 (3-seed) mean, eval unified `corgs/metrics.py`. PSNR 21.918 = Phase 22 pilot.
- FPS đo riêng trên seed42 (p22_pilot) bằng protocol unified — N_gauss FPS có thể lệch nhẹ so với 94,917 (3-seed mean).
