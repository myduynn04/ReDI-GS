# Decisions Log

> Ghi lại tất cả quyết định kỹ thuật quan trọng.
> Mục đích: tránh lặp lại tranh luận đã giải quyết, justify cho reviewer paper.

---

## Format

```
### [YYYY-MM] Tiêu đề quyết định
- **Quyết định:** Làm gì
- **Lý do:** Tại sao (kèm citation nếu có)
- **Thay thế đã cân nhắc:** Các option khác và lý do loại
- **Kết quả:** (điền sau khi có số thực nghiệm)
```

---

## Quyết định đã có

### [2026-03] Chọn CoR-GS làm codebase thay vì FSGS
- **Quyết định:** Dùng CoR-GS làm base, không phải FSGS
- **Lý do:** CoR-GS có (1) `metrics_dtu.py` sẵn cho DTU evaluation, (2) `compute_prune_mask()` với `torch.logical_or` chain dễ extend, (3) `diff-gaussian-rasterization-confidence` submodule export confidence per Gaussian, (4) rendering disagreement là basis cho R_i
- **Thay thế đã cân nhắc:**
  - FSGS: thiếu R_i và DTU eval, phải viết thêm ~2 tuần
  - SCGaussian: hybrid Gaussian phức tạp, GIM dependency nặng
  - LoopSparseGS: 35-45 phút/run → ablation không thực tế
  - DNGaussian: thiếu R_i và DTU eval
- **Kết quả:** Pending

---

### [2026-03] Bỏ Opacity Stability (O_i) khỏi CRS
- **Quyết định:** CRS = sigmoid(w1*D_i + w2*R_i), không có O_i
- **Lý do:** DropoutGS (CVPR 2025) chứng minh thực nghiệm: mô hình 10k Gaussian bị overfit nặng hơn 1k Gaussian. Floater có opacity **cao** và **ổn định** vì đang fit training view tốt. Dùng O_i sẽ bỏ sót đúng loại floater nguy hiểm nhất.
- **Thay thế đã cân nhắc:** CRS = sigmoid(w1*D_i + w2*R_i + w3*O_i) — loại vì bằng chứng từ DropoutGS
- **Kết quả:** Pending — sẽ verify bằng ablation visualizing CRS vs opacity cho floater

---

### [2026-03] Đổi T_warmup từ 2000 → 1000 iter

- **Quyết định:** T_warmup = 1000 (đổi từ 2000). Ablation vẫn test {500, 1000, 2000}.
- **Lý do:** CoR-GS sparse-view chỉ train 10k iterations. T_warmup=2000 chiếm 20% tổng training time và quan trọng hơn, densification thường kết thúc ở iter 5000-7000 — với T_warmup=2000, CRS chỉ ảnh hưởng densification trong ~3000-5000 iter. Với T_warmup=1000, CRS có thêm ~1000 iter effective window trên densification (+20-33%). Sparse-view scenes settle nhanh hơn dense-view (10k vs 30k total iter) nên 1000 iter warmup đủ để photometric loss converge sơ bộ.
- **Thay thế đã cân nhắc:**
  - T_warmup=2000: plan cũ — hợp lý cho dense-view 30k iter nhưng quá conservative cho 10k iter
  - T_warmup=500: quá sớm, R_i vẫn noisy khi scene chưa có structure rõ
  - T_warmup=3000: mất quá nhiều iter — CRS không có đủ thời gian ảnh hưởng densification
- **Kết quả:** Pending — ablation T_warmup ∈ {500, 1000, 2000}

---

### [2026-03] Dùng DepthAnything V2 thay DPT/MiDaS
- **Quyết định:** DepthAnything V2 (vitl) làm depth estimator. Nhưng cần ablation T0.8/T0.9 để verify — nếu DPT không kém nhiều thì giữ DPT.
- **Lý do:** General-purpose, không domain-sensitive như ZoeDepth (indoor/KITTI only). Tốt hơn MiDaS trên complex outdoor scenes (LLFF, DTU). Zero-shot transfer mạnh hơn.
- **Thay thế đã cân nhắc:**
  - DPT/MiDaS: dùng trong FSGS/CoR-GS baseline, kém hơn DAV2 về chất lượng depth
  - ZoeDepth: metric depth tốt nhưng domain-sensitive (fail trên outdoor)
  - Metric3D v2: tốt nhưng fragile, overhead lớn không cần thiết
- **Kết quả:** Pending — T0.8 vs T0.9

---

### [2026-03] Dùng weighted scale alignment thay naive least squares
- **Quyết định:** Align DepthAnything V2 với COLMAP points dùng weighted LS: w(p) = 1/reprojection_error(p)
- **Lý do:** Naive LS cho tất cả COLMAP points weight bằng nhau, nhưng points có reprojection error cao là unreliable. Weighted version ổn định hơn đặc biệt ở scenes textureless. Technique từ Chung et al. (CVPR Workshop 2024).
- **Thay thế đã cân nhắc:** Naive LS (không weight) — ablate trong A6
- **Kết quả:** Pending

---

### [2026-03] Multi-signal pruning dùng AND logic
- **Quyết định:** Prune khi (CRS < 0.2) AND (opacity < 0.005) AND (knn_dist > tau_isolated). CRS pruning chỉ active sau T_warmup.
- **Lý do:** OR logic sẽ prune nhầm Gaussian đang học nhưng chưa ổn định (opacity thấp do mới được tạo). AND logic chỉ prune khi cả ba signal đều xấu → conservative hơn → ít false positive hơn. CRS trước T_warmup là noise → pruning condition phải có iter check.
- **Thay thế đã cân nhắc:** CRS < tau_crs (chỉ dùng CRS) — quá aggressive ở giai đoạn đầu training
- **Kết quả:** Pending

---

### [2026-03] CRS₀ informed initialization thay vì neutral 0.5

- **Quyết định:** COLMAP points: CRS₀ = 1 - normalize(reprojection_error). Densified points: CRS₀ = 0.5.
- **Lý do:** COLMAP points có cơ sở hình học thực tế — reprojection error thấp = đáng tin hơn. Gán CRS₀ = 0.5 cho tất cả bỏ phí thông tin sẵn có từ SfM. Densified points chưa có signal → neutral default.
- **Thay thế đã cân nhắc:**
  - Alt-1: COLMAP=1.0, densified=0.5 — bỏ qua reproj quality
  - Alt-2: f(reproj), densified=0.3 — pessimistic default cho densified
  - Alt-3: f(reproj), densified=0.8 — optimistic default
  - Alt-4: tất cả=0.5 — baseline cũ, neutral nhưng uninformed
- **Kết quả:** Pending — ablation Alt-1 through Alt-4

---

### [2026-03] Position constraint bắt đầu từ T_densify=500 (không phải T_warmup)

- **Quyết định:** Position constraint BẬT ngay khi densification bắt đầu (T_densify=500), không đợi T_warmup=1000.
- **Lý do:** AD-GS (2025) chứng minh floater hình thành và tự khuếch đại ngay khi densification bắt đầu. Đợi đến T_warmup là quá muộn — 500 iter densification không kiểm soát đã tạo ra nhiều floater. Position constraint không phụ thuộc CRS (chỉ dùng depth prior), nên hoàn toàn có thể BẬT sớm.
- **Thay thế đã cân nhắc:**
  - T_densify = T_warmup = 1000: để floater khuếch đại 500 iter không kiểm soát
  - T_densify = 1000: compromise — ablate
- **Kết quả:** Pending — ablation T_densify ∈ {500, 1000}

---

### [2026-03] GFS metric (C3) tạm gác — focus C1+C2

- **Quyết định:** Không implement metrics_dtu.py extension, compute_depth_rmse(), compute_floater_ratio() cho đến khi C1+C2 có kết quả experiment rõ ràng.
- **Lý do:** C3 là contribution rủi ro — cần C1+C2 hoạt động tốt trước để có data chứng minh GFS cần thiết. Reviewer có thể dismiss metric mới nếu method chính không strong.
- **Thay thế đã cân nhắc:** Implement song song C1+C2+C3 — quá nhiều moving parts
- **Kết quả:** HOLD

---

### [2026-03] KHÔNG repurpose `confidence` attribute của CoR-GS cho CRS

- **Quyết định:** Thêm `_crs_score` attribute riêng biệt, không dùng lại `confidence` có sẵn trong gaussian_model.py.
- **Lý do:** `confidence` trong CoR-GS được truyền vào rasterizer qua `pipe.use_confidence` để weight alpha compositing trong rendering. Nếu repurpose thành CRS và bật `use_confidence=True`, rasterizer sẽ weight rendering theo CRS — thay đổi rendering behavior ngoài ý muốn và khó debug. CRS chỉ cần dùng để điều khiển densification/pruning, không cần ảnh hưởng rendering.
- **Thay thế đã cân nhắc:** Repurpose `confidence` — tiết kiệm code nhưng side effect nguy hiểm
- **Kết quả:** Pending

---

### [2026-03] Thêm Run 1-field (gaussiansN=1) làm baseline T0.5b

- **Quyết định:** Chạy thêm T0.5b — CoR-GS với gaussiansN=1, không co-reg, không co-prune — trước khi implement CRS.
- **Lý do:** CRSGaussian là single-field method. Nếu gap giữa 2-field (T0.5) và 1-field (T0.5b) lớn (>0.3 dB), CRS phải vừa bù gap vừa cải thiện — bar cao hơn. Nếu gap nhỏ (<0.15 dB), co-reg/co-prune của CoR-GS không đóng góp nhiều và single-field+CRS có path rõ ràng hơn. Không biết con số này trước khi implement là rủi ro lớn nhất về kỳ vọng.
- **Thay thế đã cân nhắc:** Chỉ chạy T0.5 (2-field) — bỏ qua thông tin về gap, set kỳ vọng sai
- **Kết quả:** Pending — T0.5b

---

### [2026-03] depth_range = median(far) - median(near)

- **Quyết định:** depth_range dùng để tính epsilon_depth và normalize D_i = median(far) - median(near) trên tất cả training cameras.
- **Lý do:** CoR-GS load near/far bounds từ poses_bounds.npy per camera. Mean dễ bị kéo lệch bởi outlier cameras (rất gần hoặc rất xa). Median ổn định hơn. Tính một lần ở đầu training, lưu vào scene object.
- **Thay thế đã cân nhắc:** mean(far) - mean(near) — kém robust hơn với outlier
- **Kết quả:** Pending

---

### [2026-03] SOTA context và target PSNR (updated)

- **Quyết định:** Ghi nhận SOTA LLFF 3-view hiện tại (tháng 3/2026). Target CRSGaussian: >21.0 competitive, >21.47 beat SOTA.

| Method | PSNR LLFF 3-view | Venue | Ghi chú |
|--------|-----------------|-------|---------|
| CoR-GS | 20.45 | ECCV 2024 | Base |
| DropGaussian | 20.76 | CVPR 2025 | |
| LoopSparseGS | 20.85 | TIP 2025 | |
| CuriGS | 21.10 | arXiv 2025 | |
| HBSplat | 21.13 | arXiv 2025 | |
| D2GS | 21.35 | arXiv 2025 | Train-test gap lớn (36.60 vs 20.26) — đáng ngờ |
| NexusGS | ~21.47 | CVPR 2025 Highlight | SOTA thực tế, dùng optical flow |

- **Ghi chú D2GS:** Train PSNR 36.60 vs test PSNR 20.26 — gap 16 dB là dấu hiệu overfit. Score-guided dropout của D2GS chưa đủ. CRSGaussian có thể beat D2GS về cả PSNR lẫn generalization nếu CRS hoạt động đúng.
- **Ghi chú NexusGS:** Phụ thuộc optical flow (FlowFormer++) — fail khi views có overlap thấp. Không fully end-to-end. Init-centric: không xử lý floater trong optimization loop.
- **Kết quả:** Pending

---

## Template cho decisions mới

### [YYYY-MM] Tiêu đề
- **Quyết định:**
- **Lý do:**
- **Thay thế đã cân nhắc:**
- **Kết quả:**

---

### [2026-03] CRS₀ = 0 (neutral) cho tất cả Gaussians — bỏ informed init

- **Quyết định:** Tất cả Gaussians khởi tạo _crs_score = 0 (logit) → CRS = 0.5 (neutral). Không dùng reprojection errors để init.
- **Lý do:** CRS không được dùng trước T_warmup → giá trị khởi tạo không có tác dụng thực tế. Sau T_warmup, mỗi 100 iter CRS sẽ được tính lại từ D_i và R_i thực tế → init value converge về 0 trong vài lần update đầu dù khởi tạo gì. Bỏ informed init đơn giản hóa code và loại bỏ dependency không cần thiết.
- **Thay thế đã cân nhắc:** CRS₀ = 1 - normalize(reproj_error) — bỏ vì không có tác dụng trước T_warmup
- **Kết quả:** Pending

---

### [2026-03] CRS scale factor = 5.0 để tận dụng full range sigmoid

- **Quyết định:** Áp dụng scale factor 5.0 trước sigmoid: crs_logit = 5.0 * (w1*D + w2*R - 0.5)
- **Lý do:** Không có scaling, sigmoid(w1*D + w2*R) với D,R ∈ [0,1] chỉ cho CRS ∈ [0.5, 0.73]. tau_crs = 0.35 không bao giờ trigger vì CRS minimum là 0.5 → pruning hoàn toàn không hoạt động. Với scale 5.0: logit ∈ [-2.5, 2.5] → CRS ∈ [0.08, 0.92] — floater có CRS ~ 0.08-0.25, surface có CRS ~ 0.75-0.92.
- **Thay thế đã cân nhắc:** Bỏ sigmoid dùng thẳng weighted average — đơn giản hơn nhưng không có smooth gradient. Scale {3.0, 8.0} — ablate nếu 5.0 không phù hợp.
- **Kết quả:** Pending — cần verify từ histogram T2.7

---

### [2026-03] Điều chỉnh ngưỡng tau_crs và tau_densify theo CRS range mới

- **Quyết định:** tau_crs = 0.35, tau_densify = 0.45 (tăng từ 0.2 và 0.4).
- **Lý do:** Với scale factor 5.0, CRS range là [0.08, 0.92]. Ngưỡng cũ tau_crs=0.2 nằm trong vùng rất thấp của range mới. tau_crs=0.35 tương đương khoảng 30% từ bottom của range — reasonable để detect floaters mà không quá aggressive. tau_densify=0.45 > tau_crs=0.35: đảm bảo chặn sinh con trước khi xóa, logic nhất quán.
- **Thay thế đã cân nhắc:** Giữ tau_crs=0.2 — không trigger vì CRS min ≈ 0.08 (floater cực đoan mới đạt)
- **Kết quả:** Provisional — chọn chính xác sau histogram T2.7. Ablate tau_crs ∈ {0.25, 0.35, 0.45}

---

### [2026-04] Dùng closed-form WLS cho depth alignment thay Adam optimizer

- **Quyết định:** Dùng closed-form weighted least squares để tìm scale, shift khi align monocular depth với COLMAP sparse depth.
- **Lý do:** Bài toán chỉ 2 tham số (scale, shift) → convex quadratic → có nghiệm chính xác trong 1 bước. Adam optimizer (như DepthRegularizedGS) là overkill — lặp hàng nghìn iter cho bài toán có closed-form solution. Depth ≤ 0 nếu xảy ra xử lý bằng clamp sau align.
- **Thay thế đã cân nhắc:** Adam optimizer (DepthRegularizedGS approach) — robust hơn nhưng chậm, không cần cho 2 tham số linear.
- **Kết quả:** Pending

---

### [2026-04] R_i range thực tế [0.333, 1.0] với K=3 cameras — CRS vẫn đủ rộng

- **Quyết định:** Ghi nhận R_i range thực tế và xác nhận tau_crs=0.35 vẫn hợp lý. Không cần điều chỉnh scale factor hay ngưỡng.
- **Lý do:** Với K=3 cameras LLFF, GT color pairwise (Hướng 1) cho R_i ∈ [0.333, 1.0]:
  - Floater hoàn toàn khác màu (worst case: 3 channels khác hết) → R = 1 - 2/3 = 0.333, KHÔNG phải 0.0
  - Surface hoàn toàn nhất quán → R = 1.0
  - Lý do R_min=0.333: L1 diff mean qua 3 RGB channels, worst case mỗi pair chỉ khác 2/3 channels
  
  Hệ quả cho CRS (logit = 5.0 * (0.5*D + 0.5*R - 0.5)):
  - Floater (D=0, R=0.333): avg=0.167 → logit=-1.67 → CRS=0.16
  - Surface (D=1, R=1.0): avg=1.0 → logit=2.5 → CRS=0.92
  - CRS thực tế range: [0.16, 0.92] — vẫn đủ rộng để detect floater
  - tau_crs=0.35 nằm giữa 0.16 (floater) và 0.92 (surface) — hợp lý
- **Thay thế đã cân nhắc:** Tăng scale factor (>5.0) để mở rộng CRS range — không cần vì [0.16, 0.92] đã đủ
- **Kết quả:** Confirmed bằng unit test T2.4. Verified bằng histogram thực nghiệm T2.7: CRS thực tế [0.14, 0.90].

---

### [2026-04] T2.7 — Kết quả thực nghiệm CRS trên fern 3-view (iter 3000)

- **Quyết định:** Ghi nhận kết quả T2.7 — CRS phân biệt được floater vs surface. Phase 2 PASS.
- **Lý do:** Kết quả thực nghiệm trên fern single-field, 3000 iter, --use_depth_prior:
  - CRS floater: mean=0.316, range [0.14, 0.32], N=2,634 (1.2%)
  - CRS surface: mean=0.820, range [0.82, 0.90], N=53,043 (24.3%)
  - CRS undecided: mean=0.497, N=145,739 (66.8%) — do densification nhanh hơn CRS update
  - **Opacity floater = 0.903** — xác nhận DropoutGS (CVPR 2025): floater có opacity CAO
  - **Opacity surface = 0.432** — thấp do alpha compositing nhiều Gaussians chồng nhau
  - Floaters_only.ply: điểm rải rác xa surface = floater thật ✓
  - Histogram bimodal (log scale): peak floater ~0.25, peak surface ~0.85
  - PSNR 21.10 (baseline 21.15) — CRS log-only không break training
  - 67% undecided sẽ giảm khi densification dừng (iter 10k) và CRS tiếp tục update
- **Thay thế đã cân nhắc:** N/A (observational)
- **Kết quả:** PASS — CRS hoạt động, sẵn sàng Phase 3

---

### [2026-04] Sửa pruning logic: AND → Option C (CRS+isolation OR opacity)

- **Quyết định:** Thay đổi pruning formula ở T4.2:
  ```
  CŨ:  prune = (CRS < tau_crs) AND (opacity < 0.005) AND (knn_dist > tau_isolated)
  MỚI: prune = (CRS < tau_crs AND knn_dist > tau_isolated) OR (opacity < 0.005)
  ```
- **Lý do:** T2.7 cho thấy floater có opacity = 0.903. Với AND logic cũ, điều kiện `opacity < 0.005` không bao giờ đúng cho floater → CRS pruning bị vô hiệu hóa hoàn toàn. Option C tách CRS pruning thành kênh riêng:
  - Kênh 1 (CRS): CRS thấp + isolated → prune (không cần opacity thấp)
  - Kênh 2 (legacy): opacity < 0.005 → prune (giữ behavior gốc của 3DGS)
  - Lý do giữ knn_dist: Gaussian đang học có CRS thấp nhưng nằm gần surface (có neighbors) → không prune nhầm. Floater thật = CRS thấp VÀ isolated.
- **Thay thế đã cân nhắc:**
  - Option A: CRS < tau OR (opacity<0.005 AND big_points) — CRS alone quá aggressive
  - Option B: CRS < tau alone — thiếu safety net, có thể prune Gaussian đang học
- **Kết quả:** Implement tại T4.2

---

### [2026-04] Bỏ adaptive_depth_loss — depth loss và CRS tách biệt hoàn toàn

- **Quyết định:** Depth loss dùng fixed lambda (không weight theo CRS). CRS chỉ điều khiển densification và pruning. Bỏ `adaptive_depth_loss()` khỏi plan.
  ```
  L_depth = lambda_base * pearson_depth_loss(rendered_depth, depth_prior)
  CRS → densification gating + multi-signal pruning (Phase 4)
  ```
- **Lý do:** Depth loss và CRS giải quyết 2 vấn đề khác nhau:
  - Depth loss: **correction** — kéo Gaussians về đúng depth
  - CRS pruning: **elimination** — xóa floater nếu vẫn tệ sau correction
  - Hai cơ chế bổ sung nhau, không cần overlap (adaptive weight)
  - Thiết kế sạch hơn: mỗi component có 1 nhiệm vụ rõ ràng
  - Ablation rõ ràng hơn: A1 (depth only) vs A2 (CRS only) vs A3 (cả hai) — không bị confound bởi adaptive weighting
- **Thay thế đã cân nhắc:** adaptive_depth_loss(lambda_base * (2 - CRS_i)) — phức tạp hơn, CRS per-Gaussian nhưng depth loss per-image → mismatch granularity, khó implement đúng
- **Kết quả:** T3.2 removed. T3.3 DONE — fern 3-view 3000 iter:
  - Test PSNR 22.35 (+1.20 vs baseline 21.15, vượt 2-field CoR-GS 22.29)
  - Train PSNR 33.34 (gap 10.99 vs 13.97 trước — less overfit)
  - N=73,575 (-65% vs no-depth 211k) — depth loss ngăn proliferation
  - CRS: mean=0.75, <0.35=876 (-81%), >0.65=56,336
  - Depth loss (correction) hoạt động mạnh. CRS pruning (Phase 4) chưa bật.

---

### [2026-04] Position constraint (T4.1) TẮT sau thực nghiệm

- **Quyết định:** Tắt position constraint hoàn toàn. Config cuối: depth loss + CRS pruning, KHÔNG có position constraint.
- **Lý do:** Thực nghiệm ablation trên fern 3-view 3000 iter:
  - Run A (depth loss + CRS pruning only): PSNR=22.37, N=72k ✓
  - Run B (depth loss + position constraint only): PSNR=19.33, N=59k ✗ (-3 dB)
  - Run C (cả hai): PSNR=19.11, N=55k ✗
  - Position constraint với epsilon=0.05*depth_range=1.53 units quá hẹp. DAV2 depth prior có noise → reject Gaussians hợp lệ ở vị trí hơi lệch surface. Kết quả: không đủ Gaussians → PSNR giảm mạnh.
  - Depth loss đã đủ correction (kéo Gaussians về đúng depth). CRS pruning đủ elimination (xóa floater). Position constraint thừa và gây hại.
- **Thay thế đã cân nhắc:**
  - Tăng epsilon (0.10, 0.15, 0.20) — có thể giảm hại nhưng vẫn reject một số Gaussians cần thiết
  - Chỉ áp dụng cho split (không clone) — clone copy parent position nên ít bị ảnh hưởng
  - Dùng soft constraint (penalty thay reject) — phức tạp hơn, chưa rõ lợi ích
- **Kết quả:** Ablation value cho paper: position constraint hại hơn lợi khi depth prior noisy. Đây là finding quan trọng — trái với AD-GS (2025) claim.