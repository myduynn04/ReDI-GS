# Task Queue — CRSGaussian Implementation

> Làm theo thứ tự. KHÔNG nhảy cóc.
> Tick [x] sau khi xong và verify.
> Ghi kết quả số vào cột "Kết quả".

---

## Trạng thái legend
- `[ ]` Chưa làm
- `[~]` Đang làm
- `[x]` Xong + verified
- `[!]` Bị block — cần giải quyết trước

---

## PHASE 0 — Setup & Baseline

| # | Task | Kết quả | Ghi chú |
|---|------|---------|---------|
| T0.0 | **Grep verify ICP/gs1 không hardcoded** | [x] Pass | 18 matches gaussiansN, 10 gs1, 3 coprune, 1 open3d — tất cả CONDITIONAL (gated bởi args). Single-field gaussiansN=1 an toàn. |
| T0.1 | Cài environment CoR-GS | [x] Done | Server đã có conda env |
| T0.2 | Cài DepthAnything V2 + viết depth_model.py | [x] Done | Checkpoint vitl 1.3GB tại ../Depth-Anything-V2/checkpoints/. Wrapper tại utils/depth/depth_model.py |
| T0.3 | Chuẩn bị LLFF dataset | [x] Done | 8 scenes đã có trên server |
| T0.4 | Chuẩn bị DTU dataset | [ ] | Để sau |
| T0.X | ~~Sửa dataset_readers.py~~ | [x] SKIP | Không cần — depth_alignment.py đọc errors trực tiếp từ points3D.bin |
| T0.5 | **Baseline CoR-GS — LLFF 8 scenes 3-view (gaussiansN=2, --coreg, --coprune)** | [x] Avg PSNR=20.11 SSIM=0.704 LPIPS=0.201 | Per-scene: fern 22.29, flower 19.86, fortress 23.46, horns 19.02, leaves 16.75, orchids 15.59, room 21.47, trex 22.43 |
| T0.5b | **Single-field baseline — LLFF 3-view (gaussiansN=1, no coreg, no coprune)** | fern: PSNR=21.15 SSIM=0.678 LPIPS=0.232 | Chỉ fern chạy được, các scene khác OOM (không có co-pruning kiểm soát Gaussian count) |
| T0.6 | **Baseline CoR-GS — DTU scan24 3-view** | PSNR=? SSIM=? | |
| T0.6b | **Single-field baseline — DTU scan24 3-view** | PSNR=? SSIM=? | Tương tự T0.5b cho DTU |
| T0.7 | Visualize floater từ depth map | | Confirm floater visible trên cả 2-field và 1-field |
| T0.8 | **Chạy CoR-GS với DPT (baseline gốc)** | PSNR=? | Depth estimator ablation |
| T0.9 | **Chạy CoR-GS với DepthAnything V2** | PSNR=? | So sánh với T0.8 — nếu ≈ thì giữ DPT |

### Kết quả cần ghi sau Phase 0

| Run | gaussiansN | coreg | coprune | PSNR | SSIM | LPIPS |
|-----|-----------|-------|---------|------|------|-------|
| T0.5 — CoR-GS gốc | 2 | ✓ | ✓ | 20.11 | 0.704 | 0.201 |
| T0.5b — Single-field | 1 | ✗ | ✗ | ? | ? | ? |
| Gap (T0.5 - T0.5b) | — | — | — | ? | ? | ? |

> Gap > 0.3 dB → co-reg/co-prune quan trọng → CRS phải compensate
> Gap < 0.15 dB → single-field đủ mạnh → CRS có path rõ ràng hơn

---

## PHASE 1 — Depth Alignment

> **Prerequisite:** T0.2 (depth_model.py + DAV2) phải xong trước T1.2. T0.X bỏ — đọc errors trực tiếp từ bin trong depth_alignment.py.

| # | Task | Kết quả | Ghi chú |
|---|------|---------|---------|
| T1.1 | Đọc DepthRegularizedGS weighted alignment code | [x] SKIP | Đã đọc và hiểu approach. Quyết định dùng closed-form WLS thay Adam — ghi decisions_log. |
| T1.2 | Tạo `utils/depth/depth_alignment.py` | [x] Done | Closed-form WLS, weight=1/reproj_err. Scale âm (DAV2 inverse depth) — đúng kỳ vọng. |
| T1.3 | Tích hợp vào `train.py` | [x] Done | --use_depth_prior flag. Test: baseline không break (PSNR 20.94 vs 20.94). DAV2+align chạy OK: scale≈-0.09, depth_range=30.56 |
| T1.4 | Test: so sánh depth map trước/sau alignment | [x] Pass | Raw [0,328] → Aligned [16.9, 47.6]. Scale âm flip đúng chiều. Depth structure giữ nguyên. debug_depth/ verified trực quan. |
| T1.5 | Ablation A6: naive vs weighted | [ ] | Chạy fern scene — để sau khi có depth loss (Phase 3) |

> **Phase 1 Summary:** DAV2 vitl predict relative depth → closed-form WLS align với COLMAP (weight=1/reproj_err) → metric depth. aligned_depth_dict + depth_range sẵn sàng cho Phase 2 (D_i) và Phase 3 (depth loss). Baseline không bị break.

---

## PHASE 2 — CRS Module

| # | Task | Kết quả | Ghi chú |
|---|------|---------|---------|
| T2.1 | Đọc CoR-GS rendering disagreement logic | [x] Done | Không refactor — co-reg/co-prune cần 2 fields, không dùng cho single-field R_i. Chỉ tái dùng: render() trả về color (N,3) + visibility_filter. R_i viết mới. |
| T2.2 | Thêm `_crs_score` attribute vào gaussian_model.py | [x] Done | 5 chỗ sửa: create_from_pcd (init=0), get_crs property, densification_postfix (neutral), prune_points, capture/restore (backward-compatible). Test PASS: PSNR 20.94, 104 it/s. |
| T2.3 | Tạo `CoR-GS/crs_module.py` — D_i function | [x] Done | GPU projection, multi-camera avg, neutral 0.5 invisible. File: utils/crs/crs_module.py. Unit test 8/8 PASS. |
| T2.4 | Tạo `CoR-GS/crs_module.py` — R_i function | [x] Done | Hướng 1: GT color pairwise. File: utils/crs/crs_module.py. Unit test 6/6 PASS. |
| T2.5 | Tạo `CoR-GS/crs_module.py` — update_crs() | [x] Done | EMA trên logit space, scale=5.0, w1=w2=0.5 default. Unit test 6/6 PASS. Surface=0.74, floater=0.46, invisible=0.50. |
| T2.6 | Hook vào train.py — LOG ONLY (không đổi loss) | [x] Done | PSNR 21.10 (baseline 21.15, no regression). CRS range [0.14, 0.90]. <0.35=4572 (2.2%), >0.65=54679 (25.9%). Signal separation confirmed. |
| T2.7 | Validate: floater có CRS thấp hơn surface? | [x] Done | Floater CRS=0.316 opacity=0.903, surface CRS=0.820 opacity=0.432. Histogram bimodal (log). Floaters_only.ply confirmed. Pruning AND logic cần sửa → Option C ở T4.2. |

---

## PHASE 3 — Depth Loss

> **Nhắc lại:** CoR-GS gốc KHÔNG có depth loss. Đây là thêm hoàn toàn mới.
> **Quyết định:** Depth loss dùng fixed lambda. CRS chỉ điều khiển densify/prune (Phase 4). Xem decisions_log.

| # | Task | Kết quả | Ghi chú |
|---|------|---------|---------|
| T3.1 | Thêm `pearson_depth_loss()` vào loss_utils.py | [x] Done | Pearson correlation, scale/shift invariant, differentiable. Unit test 8/8 PASS. |
| ~~T3.2~~ | ~~Thêm `adaptive_depth_loss()` vào loss_utils.py~~ | REMOVED | Bỏ — depth loss fixed lambda, CRS tách biệt cho Phase 4. Xem decisions_log. |
| T3.3 | Tích hợp fixed depth loss vào train.py | [x] Done | PSNR 22.35 (+1.2 vs baseline 21.15). Vượt 2-field CoR-GS (22.29). N=73k (-65% vs no-depth 211k). Gap train-test giảm 14→11 dB. |
| T3.4 | **Ablation: depth loss only vs baseline** | [x] Done | Kết quả T3.3 = ablation A1 (depth loss only). PSNR 22.35 vs A0=21.15. |

---

## PHASE 4 — Position Constraint + Full Pruning

| # | Task | Kết quả | Ghi chú |
|---|------|---------|---------|
| T4.1 | ~~Position constraint~~ | [x] DISABLED | Implemented + tested, nhưng TẮT sau ablation: PSNR -3 dB vì DAV2 noise. Xem decisions_log 2026-04. |
| T4.2 | Sửa densify_and_prune() — CRS pruning Option C | [x] Done | (CRS<0.35 AND isolated) OR legacy. Params in OptimizationParams. Unit test 5/5 PASS. |
| T4.3 | **Ablation A4: Full CRSGaussian** | PSNR=? vs A0=? | LLFF fern |
| T4.4 | **Ablation A5: T_warmup=0** | PSNR=? | Confirm need warmup |

---

## PHASE 5 — Informed CRS₀ Initialization

> **Triết lý:** CRS₀ có ý nghĩa hình học ngay từ đầu → densification iter 500-1000 không blind.
> **Prerequisite:** Phase 1 DONE (aligned depth), Phase 2 DONE (CRS module), T4.2 DONE (pruning).
> **Xem chi tiết:** docs/09_informed_crs_init_plan.md

| # | Task | Kết quả | Ghi chú |
|---|------|---------|---------|
| T5.1 | Thêm arguments Informed CRS₀ vào arguments/__init__.py | [x] Done | 11 args mới trong ModelParams. Master switch --informed_crs_init default False. str2bool cho bool default=True. |
| T5.2 | Tạo utils/crs/crs_init.py — compute_informed_crs0() | [x] Done | 4 functions: q_reproj, q_depth, q_view, compute_informed_crs0. + _match_reproj_errors helper. |
| T5.3 | Sửa gaussian_model.py — create_from_pcd() nhận informed_crs0 | [x] Done | Optional param, backward-compatible. |
| T5.4 | Sửa gaussian_model.py — densification_postfix() inherit CRS | [x] Done | Conservative inherit: clip(η*parent, 0, 0.5). Gated --crs_densify_inherit. Chain qua split/clone/prune. |
| T5.5 | Sửa train.py — hook informed CRS₀ + eta | [x] Done | Option C: overwrite sau Scene(). Log Q distribution. eta truyền vào densify. |
| T5.6 | Tạo tests/test_informed_crs_init.py — 6 unit tests | [x] Done | 6/6 PASS trên server. |
| T5.7 | Verify baseline không break (informed_crs_init=False) | [ ] | Chạy trên server: PSNR phải = baseline cũ. |
| T5.8 | Log Q distribution (100 iter, --informed_crs_init) | [ ] | Validate γ=5.0 phù hợp với Q distribution thực tế. |

---

## PHASE 5b — Anti-Overfit: Pseudo-View Loss

> **Mục đích:** Giảm train-test gap (~16-18 dB) bằng cách ép Gaussians học geometry/appearance ở novel views, không chỉ training views.
> **Bottleneck đã xác định:** Densify_until ablation chứng minh dừng densify sớm KHÔNG giảm overfit (drop AVG chỉ ±0.05 dB) → root cause là model overfit photometric ở 3 training views, cần signal regularization mới.

### Sub-phase 5b.1 — Pseudo Depth Loss (Approach 1) — KHÔNG HOẠT ĐỘNG

| # | Task | Kết quả | Ghi chú |
|---|------|---------|---------|
| T5b1.1 | Tạo `utils/depth/depth_warping.py` — forward warp DAV2 depth | [x] Done | 3 functions: `find_nearest_training_cam`, `forward_warp_depth`, `compute_warp_coverage`. Convention `cam.R = C2W rot`, `cam.T = W2C trans`. |
| T5b1.2 | Verify camera convention (round-trip + gradient flow) | [x] PASS | 4/4 tests: round-trip 7e-15, coverage 86%, scale ratio 0.875, gradient norm 1.46e-2. |
| T5b1.3 | 6 unit tests synthetic | [x] PASS | 6/6 tests trong `tests/test_pseudo_depth_warp.py`. |
| T5b1.4 | Hook `--use_pseudo_depth_loss` vào train.py | [x] Done | Gated, default OFF. Tái dùng `RenderDict["depth_pseudo_co_gs0"]`. |
| T5b1.5 | Ablation 7 configs × 8 LLFF scenes (B0/PM/PL/PH/PU5/PE/PT) | [x] Done | **AVG @10k: PL=20.288 vs B0=20.230 (+0.058)** — không cải thiện, gain trong noise floor. |
| T5b1.6 | Verify H1-H4 hypotheses | [x] **H4 = ROOT CAUSE** | H1/H2/H3 đã verify OK ở session trước. **H4: pseudo cams chỉ cách training cams 0.3-3.68° (max!), 0/10000 cam pass threshold 5°**. Pseudo loss = duplicate training depth signal. |
| T5b1.7 | Ablation H4 fix? | SKIP | F1 (filter) bất khả thi vì max angle 3.68° < threshold. F2 (perturb) phức tạp + reference DAV2 redundant nên không đáng đầu tư. → Switch sang Approach 2. |

### Sub-phase 5b.2 — Pseudo Photometric Consistency (Approach 2)

> **Triết lý khác:** Reference KHÔNG phải DAV2 (đã dùng training depth loss → redundant) mà là **GT IMAGE** warped từ training cam. Detect floater qua **PARALLAX**: floater 3D ở vị trí sai → khi nhìn từ pseudo cam (dù chỉ 3°), parallax shift `~ depth × tan(3°)` đủ để L1 loss detect.

| # | Task | Kết quả | Ghi chú |
|---|------|---------|---------|
| T5b2.1 | Thêm `warp_image_forward()` vào `utils/depth/depth_warping.py` | [x] Done | Scatter RGB thay vì depth, collision nearest-wins. |
| T5b2.2 | Thêm 3 args vào ModelParams | [x] Done | `use_pseudo_photo_loss`, `lambda_pseudo_photo`, `pseudo_photo_start_iter`. Default OFF → không phá baseline. |
| T5b2.3 | Hook vào train.py — loss block sau pseudo depth block | [x] Done | Gated bởi 5 điều kiện AND. Tái dùng `RenderDict["image_pseudo_co_gs0"]`. L1 masked loss với valid mask broadcast (3,H,W). |
| T5b2.4 | Tạo `scripts/ablation_pseudo_photo.sh` | [x] Done | 5 configs: B0, AP2_005 (λ=0.005), AP2_01, AP2_02, AP2_05. Lambda thấp vì warp có boundary artifacts. |
| T5b2.5 | Smoke test fern (5 runs) | [ ] | Verify không crash, có log `loss/pseudo_photo`. |
| T5b2.6 | Full 8 LLFF scenes (40 runs, parallel 2 GPU) | [ ] | Sau khi smoke test pass. |

### Sub-phase 5b.3 — Pseudo DAV2 Self-Reference (Approach 1') — HOLD

> Chạy DAV2 trên rendered image tại pseudo cam → reference depth phụ thuộc content render thực. FSGS-style. **Hold cho đến khi Approach 2 cho kết quả** vì overhead DAV2 inference per-iter cao (+1.3GB GPU memory + ~50ms/call), risk OOM.

---

## PHASE 6 — GFS Metric [HOLD - deprioritized]

> **Tạm gác hoàn toàn.** Focus 100% vào C1 (CRS Module) và C2 (CRS-guided Densification) trước.
> Không implement metrics_dtu.py extension, compute_depth_rmse(), compute_floater_ratio().

| # | Task | Kết quả | Ghi chú |
|---|------|---------|---------|
| ~~T6.1~~ | ~~Mở rộng metrics_dtu.py — Depth RMSE~~ | HOLD | |
| ~~T6.2~~ | ~~Mở rộng metrics_dtu.py — Floater Ratio~~ | HOLD | |
| ~~T6.3~~ | ~~Validate: CRSGaussian vs CoR-GS trên DTU~~ | HOLD | |
| ~~T6.4~~ | ~~Scatter plot PSNR vs Depth RMSE~~ | HOLD | |

---

## PHASE 7 — Full Ablation Study

### Component ablation

| Config | D_i | R_i | Pos.const | W.align | CRS₀ | T_warmup | PSNR (fern) | PSNR (DTU avg) |
|--------|-----|-----|-----------|---------|------|----------|-------------|----------------|
| A0 — CoR-GS 2-field (T0.5) | ✗ | ✗ | ✗ | ✗ | N/A | N/A | 22.29 | |
| A0b — Single-field (T0.5b) | ✗ | ✗ | ✗ | ✗ | N/A | N/A | 21.15 | |
| A1 — Depth loss only (T3.3) | ✗ | ✗ | ✗ | ✓ | 0.5 | 1000 | **22.35** | |
| A2 — R_i only | ✗ | ✓ | ✗ | ✓ | f(reproj)/0.5 | 1000 | | |
| A3 — Full CRS, no pos | ✓ | ✓ | ✗ | ✓ | f(reproj)/0.5 | 1000 | | |
| A4 — Full CRSGaussian | ✓ | ✓ | ✓ (T_densify=500) | ✓ | f(reproj)/0.5 | 1000 | | |
| A5 — No warmup | ✓ | ✓ | ✓ | ✓ | f(reproj)/0.5 | 0 | | |
| A6 — Naive align | ✓ | ✓ | ✓ | ✗ | f(reproj)/0.5 | 1000 | | |

### CRS₀ Informed Initialization ablation (updated 2026-04)

| Config | q_reproj | q_depth | q_view | Inherit | PSNR (fern) | Ghi chú |
|--------|----------|---------|--------|---------|-------------|---------|
| C0 | ✗ | ✗ | ✗ | ✗ | | Baseline — informed_crs_init=False |
| C1 | ✓ | ✗ | ✗ | ✗ | | Chỉ reproj |
| C2 | ✗ | ✓ | ✗ | ✗ | | Chỉ depth |
| C3 | ✗ | ✗ | ✓ | ✗ | | Chỉ view |
| C4 | ✓ | ✓ | ✗ | ✗ | | reproj + depth |
| C5 | ✓ | ✗ | ✓ | ✗ | | reproj + view |
| C6 | ✗ | ✓ | ✓ | ✗ | | depth + view |
| C7 | ✓ | ✓ | ✓ | ✗ | | Tất cả, equal weights |
| C8 | ✓ | ✓ | ✓ | ✓ | | Tất cả + densify inherit |

### Hyperparameter sensitivity

| Param | Default | Values | PSNR (fern) |
|-------|---------|--------|-------------|
| T_densify | 500 | {500, 1000} | |
| T_warmup | **1000** | {500, 1000, 2000} | ← default đổi từ 2000 |
| epsilon_depth | 0.05 | {0.02, 0.05, 0.10} | |
| w1/w2 | 0.5/0.5 | {0.5/0.5, 0.7/0.3, 0.3/0.7, 0.6/0.4, 0.4/0.6, 0.8/0.2, 0.2/0.8} | |
| CRS scale factor | 5.0 | {3.0, 5.0, 8.0} | |
| EMA decay | 0.9 | {0.8, 0.9, 0.95} | |
| CRS update interval | 100 | {50, 100, 200} | |
| tau_crs | 0.35 | {0.25, 0.35, 0.45} | ← updated từ 0.20 |

> **Quy trình tìm bộ trọng số tốt nhất:**
> 1. Chạy fern 3-view với default (w1/w2=0.5/0.5, scale=5.0, ema=0.9)
> 2. Grid search w1/w2 trước (7 bộ) — giữ cố định scale, ema
> 3. Chọn w1/w2 tốt nhất → grid search scale (3 giá trị)
> 4. Chọn scale tốt nhất → grid search ema (3 giá trị)
> 5. Verify bộ tốt nhất trên 2-3 scenes LLFF khác (flower, room)

---

## PHASE 8 — Full Experiments

| # | Task | Status |
|---|------|--------|
| T8.1 | LLFF 8 scenes × baselines | |
| T8.2 | DTU 15 scenes × baselines | |
| T8.3 | Blender 8 scenes × baselines | |
| T8.4 | Compile Table 1 (PSNR/SSIM/LPIPS) | |
| T8.5 | Compile Table 2 (Depth RMSE / Floater Ratio) [HOLD] | |

---

## Notes / Blockers

*(Ghi vào đây khi gặp vấn đề)*

---

## Future Work / Ideas

- **Selective CRS update**: Hiện tại update_crs() tính D_i+R_i cho TẤT CẢ N Gaussians mỗi lần. Gaussians đã ổn định (không di chuyển) có D_i/R_i giống hệt → update thừa. Ý tưởng: chỉ update Gaussians mới sinh + Gaussians có gradient lớn. Tiết kiệm ~80% compute ở iter muộn. Chưa cần vì overhead hiện tại nhỏ (vài ms/lần, N=90k, K=3).