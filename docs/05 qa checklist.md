# QA Checklist — CRSGaussian

> Claude Code tự check list này sau mỗi task trước khi báo cáo "Done".
> Không tick = chưa xong. Không được chuyển task tiếp nếu còn ô trống.

---

## Phase 0 — Baseline

### T0.0 — Grep verify
- [ ] Chạy grep command, paste output đầy đủ
- [ ] Phân tích từng dòng output: conditional hay unconditional?
- [ ] Kết luận PASS hoặc FAIL với lý do cụ thể
- [ ] Nếu FAIL: đề xuất fix với line numbers chính xác
- [ ] Ghi kết quả vào 03_task_queue.md

### T0.X — Extract reprojection errors
- [ ] Đọc dataset_readers.py và báo cáo đúng tên hàm đọc COLMAP
- [ ] Xác định dòng đang discard errors (paste line number + code)
- [ ] Sửa tối thiểu — không sửa logic khác
- [ ] Script verify: shape, min, max, mean của reprojection_errors
- [ ] Giá trị mean trong range hợp lý (0.3–2.0 pixel)
- [ ] Ghi Pass/Fail + min/max/mean vào 03_task_queue.md

### T0.5 — 2-field baseline
- [ ] Command chạy được copy từ README/script, không tự viết
- [ ] PSNR nằm trong range 19.5–21.0
- [ ] Ghi đủ PSNR, SSIM, LPIPS
- [ ] Depth map đã được render và lưu vào output/
- [ ] Ghi vào 03_task_queue.md

### T0.5b — 1-field baseline
- [ ] Xác nhận gaussiansN=1 argument từ code (paste tên argument chính xác)
- [ ] Xác nhận coreg/coprune bị skip (paste if block từ train.py)
- [ ] Ghi đủ PSNR, SSIM, LPIPS
- [ ] Tính và ghi gap = T0.5 - T0.5b
- [ ] Ghi interpretation (gap lớn/vừa/nhỏ) vào 04_decisions_log.md
- [ ] Bảng so sánh 2-field vs 1-field đã được điền đầy đủ

---

## Phase 1 — Depth Alignment

### T1.1 — Đọc DepthRegularizedGS
- [ ] Tên file chứa alignment logic đã được xác định
- [ ] Function signature đã được paste chính xác
- [ ] Cách tính weight w(p) đã được giải thích
- [ ] Phương pháp solve (closed form / scipy) đã được xác định
- [ ] Edge cases đã được liệt kê

### T1.2 — depth_alignment.py
- [ ] File tạo đúng vị trí: `CoR-GS/depth_alignment.py`
- [ ] Function signature match với spec trong prompt
- [ ] Docstring đầy đủ (input, output, logic)
- [ ] Test 1 pass: s≈2.0, t≈3.0 (±0.1) với synthetic data
- [ ] Test 2 pass: không crash khi valid points < min_valid
- [ ] Test 3 pass: D_aligned.shape == D_dense.shape
- [ ] Output test được in ra rõ ràng
- [ ] Không sửa file nào khác

### T1.3 — Tích hợp train.py (structure only)
- [ ] depth_range được tính từ median(far) - median(near)
- [ ] depth_prior_dict được khởi tạo (dù còn empty)
- [ ] Training 10 iter không crash
- [ ] Print statements confirm structure OK
- [ ] Ghi Pass/Fail vào 03_task_queue.md

### T1.4 — depth_model.py + fill depth_prior_dict
- [ ] File tạo đúng vị trí: `CoR-GS/depth_model.py`
- [ ] DepthModel.predict() nhận đúng format input từ Camera object
- [ ] depth_prior_dict được fill cho tất cả training cameras
- [ ] s, t được log cho mỗi camera (xác nhận alignment hợp lý)
- [ ] depth_model bị del sau khi precompute xong (giải phóng VRAM)
- [ ] Visualize 1 depth map trước/sau alignment — scale trông hợp lý không?
- [ ] Training 50 iter không crash

---

## Phase 2 — CRS Module

### T2.1 — Đọc CoR-GS co-reg logic
- [ ] Mô tả cách CoR-GS enforce consistency (photometric? geometric?)
- [ ] Pseudo-view generation logic đã được giải thích
- [ ] compute_prune_mask() conditions đã được list đầy đủ
- [ ] Warp function nào tồn tại trong codebase đã được xác định
- [ ] Phần nào có thể tái dụng cho R_i đã được chỉ rõ

### T2.2 — _crs_score attribute
- [ ] List đầy đủ tất cả self._* attributes (tên + shape) trong __init__
- [ ] List tất cả properties hiện có
- [ ] densification_postfix() pattern đã được đọc (paste ví dụ 1 attribute)
- [ ] capture()/restore() tuple order đã được xác định
- [ ] Sửa đúng 6 chỗ: __init__, create_from_pcd, densification_postfix, prune_points, property, capture/restore
- [ ] KHÔNG dùng confidence attribute
- [ ] KHÔNG enable use_confidence trong rasterizer
- [ ] Unit test: get_crs.shape == (N,) ✓
- [ ] Unit test: get_crs range (0,1) ✓
- [ ] Unit test: densified Gaussians có CRS ≈ 0.622 ✓
- [ ] Unit test: prune_points giảm shape đúng ✓
- [ ] Unit test: capture/restore roundtrip ✓

### T2.3 — compute_depth_consistency (D_i)
- [ ] File: `CoR-GS/crs_module.py` được tạo
- [ ] Dùng 3D projection, KHÔNG dùng rendered depth per-pixel
- [ ] Có visibility filter (chỉ update visible Gaussians)
- [ ] Occluded Gaussians giữ giá trị cũ (D=0.5 nếu chưa có history)
- [ ] project_to_depth() helper đúng (z trong camera space)
- [ ] project_to_pixel() helper đúng (fx, fy từ FovX, FovY)
- [ ] Unit test: output shape [N] ✓
- [ ] Unit test: range [0,1] ✓
- [ ] Unit test: invisible → D=0.5 ✓
- [ ] Unit test: Gaussian đúng tại depth prior → D≈1.0 ✓

### T2.4 — compute_reprojection_consistency (R_i)
- [ ] Cache renders để tránh re-render nhiều lần
- [ ] warp_pixels() helper implement đúng (unproject + re-project)
- [ ] Xử lý warped pixels out-of-bounds (clamp)
- [ ] Invisible Gaussians: R=0.5 (neutral)
- [ ] Unit test: output shape [N] ✓
- [ ] Unit test: range [0,1] ✓
- [ ] Unit test: perfect geometry → R≈1.0 ✓

### T2.5 — update_crs() EMA
- [ ] EMA trên logit space (không phải trên sigmoid output)
- [ ] Update in-place với torch.no_grad()
- [ ] Unit test: sau 10 iter với D=R=0.8 → converge đến sigmoid(0.8)≈0.69 ✓
- [ ] Unit test: EMA decay đúng 10% sau 1 update với ema=0.9 ✓
- [ ] Unit test: không break gradient graph của parameters khác ✓

### T2.6 — Hook CRS vào train.py (LOG ONLY)
- [ ] Import crs_module đúng vị trí trong train.py
- [ ] T_warmup = 1000 (không phải 2000)
- [ ] CRS chỉ tính sau iteration > T_warmup
- [ ] Tính mỗi 100 iter
- [ ] Dùng torch.no_grad() wrapper
- [ ] Log: mean, std, pct_below_0.3, pct_above_0.7, histogram
- [ ] KHÔNG sửa loss
- [ ] KHÔNG sửa densify_and_prune()
- [ ] KHÔNG sửa compute_prune_mask()
- [ ] Training 1500 iter không crash ✓
- [ ] TensorBoard có CRS metrics từ iter 1100 trở đi ✓

### T2.7 — Validate CRS signal
- [ ] Chạy đủ 2000 iter với logging
- [ ] Screenshot histogram CRS tại iter 1100, 1500, 2000
- [ ] D_vals.std() > 0.05 (không collapse) ✓
- [ ] R_vals.std() > 0.05 (không collapse) ✓
- [ ] CRS distribution không flat (std > 0.05) ✓
- [ ] Ghi nhận: bimodal / unimodal / flat
- [ ] Ghi mean CRS tại 1000, 1500, 2000 vào 03_task_queue.md
- [ ] Kết luận rõ: GO (tiếp tục Phase 3) hay NO-GO (cần debug)

---

## Phase 3 — Adaptive Depth Loss

### T3.1 — Fixed depth loss
- [ ] pearson_depth_loss() trong loss_utils.py
- [ ] Docstring: input format, output range
- [ ] Lý do dùng Pearson (scale-invariant) đã được ghi trong docstring
- [ ] Unit test: identical tensors → loss = 0 ✓
- [ ] Unit test: anti-correlated → loss = 2 ✓
- [ ] Tích hợp vào train.py chỉ active khi T_densify ≤ iter < T_warmup

### T3.2 — Adaptive depth loss
- [ ] adaptive_depth_loss() trong loss_utils.py
- [ ] lambda_i = lambda_base * (2.0 - CRS_i) đúng
- [ ] CRS weights được project từ Gaussian space về pixel space
- [ ] Unit test: CRS=0 → lambda=2x ✓
- [ ] Unit test: CRS=1 → lambda=1x ✓
- [ ] Tích hợp vào train.py chỉ active sau T_warmup

### T3.3 — Ablation A3
- [ ] Config A3 được chạy: D_i + R_i, no position constraint
- [ ] PSNR được ghi vào ablation table trong 03_task_queue.md
- [ ] So sánh với A0 (baseline): ΔPSNR = ?
- [ ] Kết luận: depth loss có tác dụng không?

---

## Phase 4 — Position Constraint + Pruning

### T4.1 — Position constraint
- [ ] Constraint BẬT từ T_densify=500, không phải T_warmup
- [ ] epsilon_depth = 0.05 * depth_range (median based)
- [ ] Thử 3 lần, nếu không được → skip (không crash)
- [ ] Chỉ gọi project_to_depth (không cần render)
- [ ] Unit test: valid position được accept ✓
- [ ] Unit test: invalid position bị skip sau 3 attempts ✓
- [ ] Ablation: epsilon_depth ∈ {0.02, 0.05, 0.10} đã được setup

### T4.2 — CRS pruning
- [ ] Chỉ active sau T_warmup (có iter check)
- [ ] AND logic: CRS < 0.2 AND opacity < 0.005 AND isolated
- [ ] _compute_isolation_mask() implement kNN đúng
- [ ] KHÔNG thay đổi conditions hiện có
- [ ] Unit test: floater thỏa cả 3 → bị prune ✓
- [ ] Unit test: Gaussian mới (opacity thấp) không bị prune nhầm ✓

### T4.3 — Ablation A4 (Full CRSGaussian)
- [ ] Config A4 được chạy: full CRSGaussian
- [ ] PSNR được ghi vào ablation table
- [ ] Config A5 được chạy: T_warmup=0
- [ ] So sánh A4 vs A0 vs A3: ΔPSNR rõ ràng
- [ ] Kết luận: position constraint có tác dụng không?

---

## Tiêu chí GO/NO-GO cho mỗi Phase

| Phase | GO condition | NO-GO — phải làm gì |
|-------|-------------|---------------------|
| Phase 0 → 1 | T0.5b xong, gap đã biết | Không tiến nếu không có anchor numbers |
| Phase 1 → 2 | depth_prior_dict filled, s/t hợp lý, 50 iter không crash | Debug alignment nếu s<0 hoặc >10 |
| Phase 2 → 3 | T2.7 signal valid (D.std>0.05, R.std>0.05) | Debug T2.3 hoặc T2.4 nếu signal collapse |
| Phase 3 → 4 | A3 PSNR > A0 PSNR | Investigate depth loss nếu không improve |
| Phase 4 → Full | A4 PSNR > A3 PSNR | Investigate position constraint nếu không improve |

---

## Checklist review code (tự review sau mỗi function)

```
□ Type hints đầy đủ?
□ Docstring có input/output/range?
□ Edge cases được handle? (empty tensor, zero division, out-of-bounds)
□ torch.no_grad() ở đúng chỗ?
□ Không có memory leak? (render cache được clear?)
□ Shape assertions hoặc comments về expected shape?
□ Unit test cover happy path + edge cases?
□ Không hard-code hyperparameters — dùng arguments?
```