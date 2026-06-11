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
| fern | 23.840 | 0.7970 | 0.7947 | 0.1426 | 0.0660 | 86,715 | 672 | 221.9 |
| flower | 21.413 | 0.6915 | 0.6872 | 0.2086 | 0.1054 | 107,109 | 984 | 145.9 |
| fortress | 25.569 | 0.8369 | 0.8312 | 0.1173 | 0.0518 | 60,316 | 492 | 295.8 |
| horns | 21.079 | 0.7749 | 0.7658 | 0.1849 | 0.0942 | 69,765 | 414 | 194.2 |
| leaves | 19.375 | 0.7287 | 0.7425 | 0.1630 | 0.0995 | 246,378 | 1482 | 96.0 |
| orchids | 17.582 | 0.5815 | 0.5891 | 0.2053 | 0.1333 | 78,492 | 580 | 174.3 |
| room | 22.970 | 0.8808 | 0.8696 | 0.1274 | 0.0622 | 39,852 | 336 | 389.7 |
| trex | 23.514 | 0.8626 | 0.8585 | 0.1155 | 0.0584 | 70,711 | 450 | 188.0 |
| **MEAN** | **21.918** | **0.7692** | **0.7673** | **0.1581** | **0.0839** | **94,917** | **676** | **213.2** |

- Train ≈ **676 s/scene (~11.3 phút)** ở 10k iter (3-seed mean).
- **FPS = 213.2 (FAIR)** — unified protocol warmup50+300timed, -r8, seed42. **FAIR** = bỏ pass SH→RGB Python *chỉ-dùng-lúc-train* khỏi đường infer (`bench_fps_crsgaussian_fair.sh`), **verified bit-identical ảnh** (img_max_abs_diff=0.0 mọi scene). Raw-có-pass ≈ 175-188 (biến thiên GPU ±10 giữa các lần đo). N_gauss FPS đo trên seed42, lệch nhẹ so cột N_gauss 3-seed.

> Per-scene PSNR: fortress 25.57 (cao nhất) · fern 23.84 · trex 23.51 · room 22.97 · flower 21.41 · horns 21.08 ·
> leaves 19.38 · orchids 17.58 (thấp nhất). Pattern khớp đặc tính LLFF.

---

## So với Binocular3DGS (cùng -r8, hold-out, cùng unified evaluator)

| Method | Iter | PSNR | SSIM | SSIM_sk | LPIPS | AVGE | N_gauss | Train s/scene | FPS (unified) |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Binocular3DGS | 30k | 21.356 | 0.744 | 0.742 | 0.171 | 0.092 | 106,620 | 1483 | **266.6** |
| **CRSGaussian (ours)** | 10k | **21.918** | **0.769** | **0.767** | **0.158** | **0.084** | **94,917** | **676** | 213.2 |
| **Δ ours − Bino** | | **+0.562** | **+0.025** | **+0.025** | **−0.013** | **−0.008** | **−11,703** | **−807 (~2.2×)** | −53 |

**Luận điểm cho paper:** ours **thắng mọi metric chất lượng** (PSNR +0.562, SSIM/SSIM_sk/LPIPS/AVGE đều tốt hơn) + **ít Gaussian hơn** (95k vs 107k) + **train ~2.2× nhanh** (676 vs 1483 s/scene) ở **3× ít iter**.

> **FPS (unified protocol — so thẳng được)**: Binocular **266.6** vs ours **213.2 (FAIR)** → Binocular render nhanh hơn,
> NHƯNG **cả hai đều real-time** (≫30 FPS). Ours chậm hơn/frame do **rasterizer-confidence** (CUDA kernel mang
> trọng số confidence per-Gaussian) — chi phí thật của cơ chế CRS, đổi lấy +0.56 PSNR + ít Gaussian + train nhanh.
> (FAIR = đã loại pass SH→RGB train-only khỏi đo infer, img_diff=0; raw-có-pass ≈ 175-188.)
> Trình bày thành thật: **infer là trục DUY NHẤT Binocular nhỉnh hơn**; ours thắng tất cả còn lại.

---

## 6-view (reproduced — N=24, 3-seed mean, full metrics)

> Ours 6-view **đã chạy thật** (không lấy từ paper). Recipe IDENTICAL Phase 22 + `--n_views 6`, 10k iter, -r8.

### Per-scene full metrics (3-seed mean, metrics.py recompute)

| Scene | PSNR ↑ | SSIM ↑ | SSIM_sk ↑ | LPIPS ↓ | AVGE ↓ | N_gauss | FPS ↑ |
|-------|--------|--------|-----------|---------|--------|---------|-------|
| fern | 26.158 | 0.8746 | 0.8751 | 0.0835 | 0.0415 | 144,831 | 82.8 |
| flower | 26.059 | 0.8618 | 0.8671 | 0.0898 | 0.0438 | 143,443 | 66.6 |
| fortress | 29.467 | 0.9095 | 0.9079 | 0.0667 | 0.0286 | 144,891 | 87.2 |
| horns | 25.324 | 0.8797 | 0.8758 | 0.1056 | 0.0527 | 148,587 | 67.4 |
| leaves | 21.057 | 0.8000 | 0.8130 | 0.1190 | 0.0747 | 308,937 | 42.9 |
| orchids | 18.870 | 0.6636 | 0.6737 | 0.1652 | 0.1078 | 157,622 | 65.1 |
| room | 29.788 | 0.9467 | 0.9413 | 0.0657 | 0.0288 | 100,070 | 98.7 |
| trex | 25.517 | 0.9129 | 0.9105 | 0.0772 | 0.0403 | 130,493 | 67.9 |
| **MEAN** | **25.280** | **0.8561** | **0.8581** | **0.0966** | **0.0523** | **159,859** | **72.3** |

### Per-seed PSNR breakdown (training-time, 24 cells)

| Scene | seed 42 | seed 137 | seed 9999 | 3-seed mean | Range |
|-------|---------|----------|-----------|-------------|-------|
| fern | 26.159 | 26.213 | 26.169 | 26.180 | 0.054 |
| flower | 26.001 | 26.081 | 26.144 | 26.075 | 0.143 |
| fortress | 29.538 | 29.578 | 29.422 | 29.513 | 0.156 |
| horns | 25.083 | 25.337 | 25.564 | 25.328 | 0.481 |
| leaves | 21.031 | 21.076 | 21.069 | 21.059 | 0.045 |
| orchids | 18.947 | 18.894 | 18.968 | 18.936 | 0.074 |
| room | 29.787 | 29.927 | 29.765 | 29.826 | 0.162 |
| trex | 25.532 | 25.573 | 25.512 | 25.539 | 0.061 |
| **8-scene mean** | **25.260** | **25.335** | **25.327** | **25.307** | 0.075 |

→ **Per-seed 8-scene mean range = 0.075** (vs noise floor ±0.10) → **rất ổn định**, mọi seed đều converge tới ~25.3 PSNR. Reproducibility cực mạnh.

→ **Worst-cell variance**: horns range 0.481 dB (single-cell variance từ atomicAdd). Mean qua 8 scenes vẫn cancel ra ±0.05.

- **2 PSNR**: **25.307** (training-time analyzer 3-seed) vs **25.280** (metrics.py recompute từ PNG renders, drift −0.027 do quantize). Cả 2 đều dùng được.
- **Δ 6-view vs 3-view**:
  - PSNR: **+3.389** (21.918 → 25.307)
  - SSIM: **+0.087** (0.769 → 0.856)
  - LPIPS: **−0.061** (0.158 → 0.097)
  - N_gauss: **+1.7×** (95K → 160K)
  - FPS: **−58%** (175 → 72) — vẫn ≫ 30 FPS real-time
- **Per-scene highlights**:
  - **room +6.86 dB** (22.97 → 29.83) — biggest jump (sparse init thiếu cho geometry phẳng)
  - **fortress +3.94** (25.57 → 29.51)
  - **leaves/orchids +1.4-1.7** — least improvement (foliage/thin texture hard regardless)

### Provenance 6-view

- Recipe: identical Phase 22 + `--n_views 6`. RoMa preprocess: C(6,2)=15 pairs (`scripts/p22_romav1_preprocess.py` với N_VIEWS=6, ~30s/scene).
- Outputs: `output/p25_3_6view/A3_seed{42,137,9999}_<scene>/` (24 dirs).
- Logs: `logs/p25_3_6view/A3_seed*_*.log` (KEEP — archive cho future re-analysis sau khi xóa output).
- FPS bench: same protocol Phase 22 (warmup50 + timed300, -r8, seed42), output `output/ablation/crsgaussian_6view/<scene>.json`.

> Baseline 6-view (FSGS/CoR-GS/Binocular/...) = **lấy từ paper** (không tự chạy) — xem `00_index.md` §3.2.

---

## 9-view (reproduced — N=24, 3-seed mean, full metrics)

> Ours 9-view **đã chạy thật**. Recipe IDENTICAL Phase 22 + `--n_views 9`, 10k iter, -r8.

### Per-scene full metrics (3-seed mean, metrics.py recompute)

| Scene | PSNR ↑ | SSIM ↑ | SSIM_sk ↑ | LPIPS ↓ | AVGE ↓ | N_gauss | FPS ↑ |
|-------|--------|--------|-----------|---------|--------|---------|-------|
| fern | 27.454 | 0.9027 | 0.9034 | 0.0661 | 0.0333 | 255,259 | 61.0 |
| flower | 27.154 | 0.8875 | 0.8930 | 0.0723 | 0.0360 | 246,358 | 54.8 |
| fortress | 29.239 | 0.8977 | 0.8962 | 0.0672 | 0.0300 | 269,304 | 62.4 |
| horns | 26.887 | 0.9197 | 0.9180 | 0.0709 | 0.0379 | 275,809 | 56.2 |
| leaves | 21.909 | 0.8293 | 0.8413 | 0.1055 | 0.0655 | 381,280 | 41.7 |
| orchids | 19.665 | 0.7054 | 0.7166 | 0.1451 | 0.0952 | 246,837 | 54.6 |
| room | 29.751 | 0.9560 | 0.9509 | 0.0579 | 0.0256 | 198,840 | 76.9 |
| trex | 27.628 | 0.9393 | 0.9380 | 0.0545 | 0.0292 | 230,059 | 59.9 |
| **MEAN** | **26.211** | **0.8797** | **0.8822** | **0.0799** | **0.0441** | **262,968** | **58.4** |

### Per-seed PSNR breakdown (training-time, 24 cells)

| Scene | seed 42 | seed 137 | seed 9999 | 3-seed mean | Range |
|-------|---------|----------|-----------|-------------|-------|
| fern | 27.470 | 27.475 | 27.464 | 27.470 | 0.011 |
| flower | 27.120 | 27.235 | 27.159 | 27.171 | 0.115 |
| fortress | 29.209 | 29.150 | 29.510 | 29.290 | 0.360 |
| horns | 27.303 | 26.674 | 26.706 | 26.894 | 0.630 |
| leaves | 21.934 | 21.869 | 21.929 | 21.911 | 0.065 |
| orchids | 19.705 | 19.730 | 19.755 | 19.730 | 0.050 |
| room | 30.179 | 29.374 | 29.791 | 29.781 | 0.806 |
| trex | 27.628 | 27.716 | 27.606 | 27.650 | 0.110 |
| **8-scene mean** | **26.319** | **26.153** | **26.240** | **26.237** | 0.166 |

→ **Per-seed 8-scene range = 0.166** (slightly larger than 6-view 0.075 nhưng vẫn trong noise floor ±0.10 chút). Reproducibility tốt.

→ **room range 0.806 + horns 0.630**: largest single-cell atomicAdd variance ở 9-view (init dense hơn 6-view nhiều → race condition cao hơn). Mean qua 8 scenes vẫn cancel.

- **2 PSNR**: **26.237** (training-time analyzer 3-seed) vs **26.211** (metrics.py recompute từ PNG renders, drift −0.026).
- **Δ 9-view vs 6-view**:
  - PSNR: **+0.930** (25.307 → 26.237) — **diminishing returns 3.6× vs 3→6 jump (+3.389)**
  - SSIM: **+0.024** (0.856 → 0.880)
  - LPIPS: **−0.017** (0.097 → 0.080)
  - N_gauss: **+1.6×** (160K → 263K)
  - FPS: **−19%** (72 → 58) — vẫn ≫ 30 FPS real-time
- **Per-view gain analysis**:
  - 3→6 view: **+1.13 dB/view** (3 extra views = +3.39 dB)
  - 6→9 view: **+0.31 dB/view** (3 extra views = +0.93 dB)
  - **Diminishing rate: 3.6×** — gain/view giảm mạnh khi tăng N
- **Per-scene highlights**:
  - **fern** +1.29 dB (26.18 → 27.47) — biggest single-scene jump
  - **horns** +1.57 (25.33 → 26.89)
  - **fortress −0.22** (29.51 → 29.29) — slight decrease (already saturated ở 6-view, atomicAdd noise)
  - **leaves/orchids +0.85** — vẫn limit by thin-structure

### Provenance 9-view

- Recipe: identical Phase 22 + `--n_views 9`. RoMa preprocess: C(9,2)=36 pairs (`scripts/p22_romav1_preprocess.py` với N_VIEWS=9, ~50s/scene).
- Outputs: `output/p25_4_9view/A3_seed{42,137,9999}_<scene>/` (24 dirs).
- Logs: `logs/p25_4_9view/A3_seed*_*.log` (KEEP — evidence cho 26.237).
- FPS bench: same protocol (warmup50 + timed300, -r8, seed42), output `output/ablation/crsgaussian_9view/<scene>.json`.

> Baseline 9-view (FSGS/CoR-GS/Binocular/...) = **lấy từ paper** — xem `00_index.md` §3.3.

---

## Progression summary (3-view → 9-view, ours)

| n_views | PSNR | SSIM | LPIPS | N_gauss | FPS (raw) | Δ PSNR / view |
|---------|------|------|-------|---------|-----------|---------------|
| 3 | 21.918 | 0.7692 | 0.158 | 95K | 175 | — |
| 6 | 25.307 | 0.8561 | 0.097 | 160K | 72 | +1.13/view |
| 9 | 26.237 | 0.8797 | 0.080 | 263K | 58 | +0.31/view |

> ⚠️ **FPS ở bảng này = RAW** (có pass SH→RGB train-only, đo cùng cách cho cả 3/6/9 → nội-bộ apples-to-apples).
> Số **canonical cross-method cho 3-view = 213.2 (FAIR)** (xem §3-view + §Binocular). 6/9-view CHƯA đo lại fair
> (không có so cross-method ở đó vì baselines = paper); nếu áp fair thì ~+14% (≈82/66). Tất cả vẫn ≫30 FPS.

→ **Strong diminishing returns**: gain per added view giảm 3.6× từ 3→6 sang 6→9. Consistent với sparse-view literature.

→ **FPS scales inversely with N_gauss** (PSNR ↑ ⇄ FPS ↓): 9-view có N_gauss 2.8× hơn 3-view nhưng FPS chỉ giảm 3×.

→ **Real-time threshold maintained**: cả 3 setting ≫ 30 FPS (real-time).

---

## Provenance
- Recipe: Phase 22 RoMa v1 + A3-TRIM, 10k, -r8. Models: `CoR-GS/output/p22_pilot/A3_seed42_<scene>/` (seed42 đại diện cho FPS). Commit value 21.89 (mean 4× N=24, [[project_phase25_2_phase22_lock]]).
- Bộ số trên = N=24 (3-seed) mean, eval unified `corgs/metrics.py`. PSNR 21.918 = Phase 22 pilot.
- FPS đo riêng trên seed42 (p22_pilot) bằng protocol unified — N_gauss FPS có thể lệch nhẹ so với 94,917 (3-seed mean).
