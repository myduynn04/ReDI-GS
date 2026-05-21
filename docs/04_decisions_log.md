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

### [2026-03] ~~CRS₀ = 0 (neutral) cho tất cả Gaussians — bỏ informed init~~ → SUPERSEDED

- **Quyết định cũ:** Tất cả Gaussians khởi tạo _crs_score = 0 (logit) → CRS = 0.5 (neutral).
- **Lý do cũ:** CRS không được dùng trước T_warmup → giá trị khởi tạo không có tác dụng thực tế.
- **SUPERSEDED bởi:** [2026-04] Informed CRS₀ Initialization (xem bên dưới).
- **Lý do thay đổi:** Iter 500-1000 densification chạy không kiểm soát khi CRS₀=0.5.
  Floaters sinh ra trước T_warmup, CRS chỉ detect được sau đó → quá muộn.
  Geometry info từ COLMAP + DAV2 alignment đã có sẵn → lãng phí nếu không dùng.

---

### [2026-04] Informed CRS₀ Initialization — 3-signal geometry prior

- **Quyết định:** COLMAP Gaussians nhận CRS₀ từ 3 geometry signals thay vì neutral 0.5.
  Densified Gaussians nhận conservative inherit từ parent (capped tại 0.5).
  Toàn bộ gated bởi `--informed_crs_init` (default False → behavior cũ).
- **Công thức COLMAP Gaussians:**
  ```
  q_reproj = 1 - clip(reproj_error / τ_r, 0, 1)           τ_r=2.5
  q_depth  = 1 - clip(|d_DAV2 - d_COLMAP| / depth_range, 0, 1)
  q_view   = (n_obs - 1) / max(N_train - 1, 1)
  Q_i = w_r*q_reproj + w_d*q_depth + w_v*q_view            default 1/3 mỗi cái
  ℓᵢ⁽⁰⁾ = γ * (Q_i - 0.5)                                  γ=5.0
  CRS₀ = sigmoid(ℓᵢ⁽⁰⁾)
  ```
- **Công thức Densified Gaussians:**
  ```
  CRS₀_child = clip(η * CRS_parent, 0, 0.5)               η=0.7
  ```
  Max=0.5: child không bao giờ trên neutral → phải "earn" CRS cao.
- **Lý do:**
  1. Iter 500-1000 densification chạy "mù" → floaters sinh tự do → CRS₀=0.5 quá muộn
  2. Geometry info đã có sẵn sau alignment (reproj errors, aligned depth, view count)
  3. 3 signals bổ sung nhau: reproj=SfM quality, depth=DAV2 agreement, view=stereo support
  4. Conservative inherit đảm bảo densified Gaussians không "free ride" từ parent
- **Thiết kế ablation-friendly:**
  - Master switch: `--informed_crs_init` (False → CRS₀=0.5 như cũ)
  - Component switches: `--crs_init_use_reproj`, `--crs_init_use_depth`, `--crs_init_use_view`
  - Tự normalize weights khi component bị tắt → 9 ablation configs
  - Densify inherit switch riêng: `--crs_densify_inherit`
- **Thay thế đã cân nhắc:**
  - CRS₀=0.5 tất cả (decision cũ) — bỏ vì lãng phí geometry info, iter 500-1000 không kiểm soát
  - Chỉ dùng reproj (1 signal) — thiếu depth agreement và view support
  - CRS₀=1.0 cho COLMAP (quá optimistic) — không phân biệt COLMAP point tốt/xấu
  - Inherit CRS_parent nguyên (không cap) — child ở vị trí khác, chưa proven
- **Kết quả:** Pending — 9 ablation configs (CONFIG 0-8)

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

---

### [2026-04] Pseudo Depth Loss (Approach 1) — KHÔNG hoạt động — root cause = pseudo cams không novel

- **Quyết định:** Bỏ pseudo depth loss (warp DAV2 depth từ training cam → pseudo cam). Switch sang Approach 2 (pseudo photometric).
- **Lý do thực nghiệm:**
  - Ablation 7 configs × 8 LLFF scenes (B0/PM/PL/PH/PU5/PE/PT): AVG @10k chỉ chênh ±0.06 dB so với baseline. PL (λ=0.02) tốt nhất với +0.058 dB — trong noise floor.
  - Per-scene phân cực: room/horns/fortress gain +0.3-0.6 dB, trex hại -1.5 dB. Không generalize.
- **Root cause (verified):** **Pseudo cams chỉ cách training cams 0.3-3.68° (max!), 0/10000 cam pass threshold 5°.**
  - Đo trên fern: distribution `[0.294°, 3.681°]`, mean=2.35°, median=2.48°
  - Toán: `arctan(spread/focal) ≈ arctan(1.5/30) ≈ 2.86°` — khớp empirical
  - Lý do: `generate_random_poses_llff()` sample position uniform trong **bbox training cams** với **fixed lookat** → forward direction bị giới hạn bởi spread của training cluster
  - LLFF forward-facing → 3 training cams cùng hướng `[~0.1, 0, 1]` → bbox nhỏ → pseudo cam là perturbation 3° của training cam, không phải view "novel"
- **Hệ quả lý thuyết:** Pseudo depth loss với reference = warped DAV2 depth = **duplicate training depth signal** với view shift 3° → không tạo regularization mới → không giảm overfit.
- **Thay thế đã cân nhắc:**
  - F1 (filter pseudo cam có angle > 5°): bất khả thi vì max=3.68°
  - F2a (tăng radii × N): rủi ro pseudo cam ra ngoài scene → render empty
  - F2b (perturb forward direction): khả thi nhưng reference DAV2 vẫn redundant → không đáng đầu tư
  - **F3: Switch sang pseudo PHOTOMETRIC** — reference = warped GT image (không phải DAV2) → signal độc lập với depth loss → CHỌN
- **Code retained:** `forward_warp_depth()` giữ lại, có thể tái dùng. `--use_pseudo_depth_loss` flag giữ default OFF, không xoá để có thể re-enable nếu cần.
- **Kết quả:** Ablation value cho paper — finding: pseudo cam interpolation trong LLFF không tạo true novel views; cần regularization mechanism khác.

---

### [2026-04] Pseudo Photometric Consistency (Approach 2) — đang test

- **Quyết định:** Implement pseudo photometric loss = L1 masked giữa rendered image tại pseudo cam vs warped GT image từ training cam gần nhất.
- **Cơ chế:**
  - Forward warp: GT image của training cam A → warped image tại pseudo cam P (dùng aligned depth của A)
  - Loss = `lambda * L1_masked(rendered_image_P, warped_GT_P)`
- **Tại sao có thể work dù pseudo cam chỉ 3° away:**
  - Floater là 3D point ở vị trí sai → khi nhìn từ pseudo cam (3°), parallax shift = `floater_depth × tan(3°)`
  - Với floater depth ~5 units → shift ~0.26 units trong view space → nhiều pixels mismatch
  - L1 per-pixel detect parallax error trực tiếp (không như Pearson scale-invariant)
- **Khác Approach 1 (depth):**
  - Reference = GT IMAGE (3 channels, never used as signal before), KHÔNG phải DAV2 depth
  - Signal độc lập với training depth loss → không redundant
  - L1 absolute, không phải Pearson statistical
- **Lambda thấp (`{0.005, 0.01, 0.02, 0.05}`):**
  - Warped GT có **artifacts ở boundaries** (forward warp scatter holes) → ngay cả model hoàn hảo vẫn có baseline L1 ~0.02-0.05
  - Lambda quá cao → ép Gaussians fit warp artifacts → harm
- **Thay thế đã cân nhắc:**
  - Approach 1' (DAV2 trên rendered image, FSGS-style): hold do overhead +1.3GB GPU + ~50ms/iter, risk OOM
  - SSIM thay L1: chưa cần, L1 đơn giản hơn để verify trước
- **Kết quả:** Pending — smoke test fern 5 configs đầu tiên.

---

### [2026-04] Track A: `_features_rest` là culprit, KHÔNG phải DC
- **Quyết định:** Identify `_features_rest` (higher-order SH) là primary source của train-test gap
- **Lý do:**
  - Hypothesis ban đầu trên fern-only đánh giá sai (fern Δ = +0.08 dB, min trong 8 scenes)
  - Full 8 scenes: SH1 (degree=1, cắt rest từ 45→9 params) cho +0.20 dB AVG
  - Exp A2 (DC-only freeze): chỉ +0.053 dB ≈ noise → DC không phải culprit
  - Chênh lệch full freeze (+0.32) vs DC-only (+0.05) = +0.27 dB → đến từ rest
- **Thay thế đã cân nhắc:**
  - "DC là culprit" (hypothesis sai): bị reject bởi Exp A2 full 8 scenes
  - "SH nói chung là culprit" (quá mơ hồ): cần isolation chính xác để viết paper
- **Kết quả:** ✅ CONFIRMED. `_features_rest` đóng góp ~85% overfit. Chi tiết docs/10_track_a_b_results.md section 2.

### [2026-04] Track A: A1 = sh_degree=1 + freeze_sh_after=1000
- **Quyết định:** Config A1 (SH1 + Freeze) làm best baseline cho Track B
- **Lý do:**
  - +0.379 dB AVG (tốt hơn SH1 only +0.196 và Freeze1k only +0.321)
  - Gap 13.74 dB (giảm 5 dB từ B0 18.77)
  - 6/8 scenes win, validated full 8 LLFF
- **Thay thế đã cân nhắc:**
  - SH2 (degree=2): +0.04 dB, gần noise — không đủ effect
  - Chỉ SH1: yếu hơn combined
  - Chỉ Freeze: yếu hơn combined
- **Kết quả:** ✅ A1 thành best config cho Track B baseline. Limitation: hại fortress (-0.18) và trex (-0.06) vì scene có specular thật.

### [2026-04] Track B: Uniform dropout > Targeted (phản trực giác)
- **Quyết định:** Chọn B1β (uniform dropout 0.2 từ iter 1000) làm final config
- **Lý do:**
  - +0.355 dB AVG trên A1 baseline, 7/8 wins
  - Targeted SH-norm (B3β) chỉ +0.150, Hybrid (B4β) chỉ +0.070
  - Hybrid B4α thậm chí âm (-0.07) — counter-productive
  - 4 giải thích: (a) universal risk trong sparse, (b) feedback loop, (c) legitimate specular bị target, (d) overlap với CRS pruning
- **Thay thế đã cân nhắc:**
  - B3 SH-norm: lý thuyết target đúng culprit (confirmed từ Track A), nhưng thực tế kém
  - B4 Hybrid: combine cả position + color signals, nhưng over-regularize Gaussian borderline useful
- **Kết quả:** ✅ CONFIRMED. B1β winner full 8 scenes. Counter-intuitive finding đáng paper (Section 3.6 Insight 1 trong docs/10).

### [2026-04] Track B: Post-warmup timing > Immediate
- **Quyết định:** Dropout start_iter = 1000 (sau warmup) thay vì 0 (từ đầu)
- **Lý do:**
  - Nhất quán 3 modes: Δ(β-α) trung bình +0.137 dB
  - Uniform: +0.156, SH-norm: +0.114, Hybrid: +0.140
  - Warmup 1000 iter cho geometry settle + CRS active + densification ổn định
  - Dropout sau scene hình thành > dropout trong lúc xây dựng
- **Thay thế đã cân nhắc:**
  - Iter 0 (Co-Adapt nguyên bản): phá vỡ learning ban đầu
  - Iter 500 (sau densify bắt đầu): chưa test, future ablation
  - Iter 1500+ (quá muộn): có thể miss overfit window, chưa test
- **Kết quả:** ✅ iter 1000 confirmed optimal qua 3 modes.
  **UPDATE (B1 start_iter sweep full 8 scenes):** Ablation {0, 500, 1000, 1500, 2000}
  cho thấy inverted-V pattern với 1000 là peak:
  - s0: -0.156 dB (too early)
  - s500: -0.115 dB (early)
  - **s1000: best** (synchronized với SH freeze)
  - s1500: -0.134 dB (too late)
  - s2000: -0.093 dB (too late)
  Bonus: tất cả configs monotone tăng đến 10k (không decay) — dropout loại bỏ late-stage overfit.
  Chi tiết: docs/10_track_a_b_results.md section 3.7.

### [2026-04] Final method: A1 + B1β cumulative
- **Quyết định:** Method cuối = CRS + A1 (sh_degree=1 + freeze 1k) + B1β (uniform dropout 0.2 từ 1k)
- **Lý do:**
  - Cumulative từ CoR-GS gốc: 20.08 → 20.96 = **+0.88 dB**
  - Gap: ~24 dB → 9.64 dB = **giảm 60%**
  - 2 cơ chế (A1 + B1β) ĐỘC LẬP, additive, không overlap:
    - A1 attacks higher-order SH drift
    - B1β attacks co-adaptation
  - Synergy scene-level: A1 hại fortress (-0.18) → B1β cứu (+0.93 trên A1)
- **Thay thế đã cân nhắc:** Từng component riêng lẻ đều kém hơn combined
- **Kết quả:** ✅ Final method for paper. Chi tiết docs/10_track_a_b_results.md section 3.7.

### [2026-04] Diagnostic Gap: Residual 9.64 dB chủ yếu do viewpoint novelty
- **Quyết định:** Residual gap là structural bound, không cần thêm SH regularization
- **Lý do:**
  - T4 (DC vs rest): Δ = +0.004 to +0.089 → rest NEUTRAL trên 4/4 scenes
  - T5 (Angular dist): test views cách 22-43° → FAR, information limited
  - T2 (SH diverge): test/train ratio 0.49-2.08x → LOW divergence, SH đã stable
  - T6 (Pareto): top 10% → 37% error → DISTRIBUTED, targeted fix vô ích
- **Thay thế đã cân nhắc:**
  - Cross-view SH consistency loss: ❌ SH đã stable → thêm loss không giúp
  - SH-norm dropout (B3): ❌ SH norm không còn correlated với error
  - CRS-guided SH learning rate: ❌ Error distributed → targeted kém uniform
  - Adaptive dropout: ⚠️ Diminishing returns (gap chủ yếu do missing information)
- **Kết quả:** ✅ Regularization pipeline hiện tại (A1+B1β) đã extract near-maximum gain.
  Improvement tiếp theo cần **additional information sources** (dense init, better pseudo-view,
  longer training) thay vì stronger regularization.
  Chi tiết: docs/10_track_a_b_results.md section 11.
- **⚠️ SUPERSEDED 2026-04-18:** Claim "near-maximum gain" bị Phase 1 DropAnSH phản bác
  (+0.16 dB gain). Regularization ceiling là ~21.12 dB (D1), không phải ~20.96 (B1b).
  Xem entry dưới.

---

### [2026-04-18] Phase 1 DropAnSH: D1 beats A1+B1β — backbone pivot
- **Quyết định:** Chuyển backbone từ A1+B1β sang D1 (pure DropAnSH, sh=3, no freeze, no B1)
- **Lý do:**
  - 4 configs × 8 LLFF scenes (n_views=3, 10k iter):
    - B0 (A1+B1β) = 20.96 AVG (prior best)
    - **D1 (pure DropAnSH) = 21.12 AVG** (+0.16 dB vs B0, +0.66 vs CoR-GS)
    - D2 (A1+DropAnSH stack) = 21.03 (A1 freeze conflicts với DropAnSH SH degree dropout)
    - D3 (A1+B1β+anchor) = 20.64 (over-regularize: B1 uniform + anchor cùng giải co-adaptation)
  - D1 win 6/8 scenes, lose chỉ fern (-0.13) và horns (-0.03)
- **Thay thế đã cân nhắc:**
  - Giữ A1+B1β: reviewer sẽ bắt bỏ qua +0.16 dB gain có thực
  - Chỉ dùng SH degree dropout của DropAnSH: chưa biết component nào dominant, cần Phase 2
  - Lock A1+B1β vì "đơn giản hơn": thực tế D1 đơn giản hơn (không cần freeze, không cần sh=1, không cần B1)
- **Kết quả:** ✅ D1 là new backbone. A1+B1β labeled "dominated by D1, reference only".
  Chi tiết: docs/10 Section 12.

### [2026-04-18] Multi-seed verify lock-in: D1 signal thật, không phải CUDA noise
- **Quyết định:** Confirm D1 > B1b là signal thật, không cần flip lại backbone
- **Lý do:**
  - 3 seeds (42, 123, 2024) × 3 weak scenes (orchids/leaves/horns):
    - orchids: D1 = 16.70 ± 0.13, B1b = 16.60, Δ = +0.10 (NOISE, trong 1σ)
    - leaves: D1 = 18.49 ± 0.07, B1b = 18.31, Δ = +0.18 (SIGNAL)
    - horns: D1 = 19.95 ± 0.12, B1b = 19.81, Δ = +0.14 (SIGNAL)
  - Std trong scene 0.07-0.13 dB → model stable, không nhạy seed
  - 2/3 SIGNAL, 1 NOISE, 0 FLIP → D1 ≥ B1b robust
  - AVG gap +0.14 (3 scene) khớp +0.16 (8 scene) → consistent
- **Thay thế đã cân nhắc:** Chỉ 1 seed (Phase 1 gốc) — không đủ để loại noise
- **Kết quả:** ✅ D1 lock-in. Tiến tới Phase 2.

### [2026-04-18] Conceptual reframing: "SH cần regularize liên tục, không phải chặt"
- **Quyết định:** Cập nhật root cause narrative cho paper
- **Lý do:**
  - Track A đúng: `_features_rest` là culprit (không phải DC)
  - Track A sai một phần: "cần chặt (freeze + low sh_degree)" — quá thô
  - Phase 1 chỉ ra: SH bậc cao **cần continuous stochastic regularization**
    - Freeze = binary gate, chặn SH học → bỏ lỡ cơ hội fine-tune khi geometry settle
    - DropAnSH SH degree dropout = stochastic, SH vẫn học nhưng "thỉnh thoảng vắng mặt" → không memorize
- **Thay thế đã cân nhắc:**
  - Giữ "SH cần chặt" narrative: Phase 1 evidence phủ nhận (D1 với sh=3+no freeze > A1)
  - Narrative phức tạp hơn ("mix freeze + dropout"): D2 result (-0.09) chứng minh ngược
- **Kết quả:** ✅ New narrative. Phản ánh trong paper Section 12.6 của docs/10.

### [2026-04-18] Phase 2 plan: DropAnSH component ablation
- **Quyết định:** Tách D1 thành D1-A (anchor only) và D1-S (SH degree only) trên 3 scenes
- **Lý do:**
  - D1 bật cả 2 cơ chế (anchor dropout pa=0.02 + SH degree dropout psh=0.2)
  - Phase 3 (CRS integration) cần biết component nào dominant để CRS-hóa đúng chỗ
  - Nếu anchor dominant → Phase 3 là CRS-guided anchor selection
  - Nếu SH degree dominant → Phase 3 là CRS-modulated SH dropout
- **Thay thế đã cân nhắc:**
  - Nhảy thẳng Phase 3 với cả 2: không biết CRS-hóa chỗ nào có tác dụng
  - Ablation 8 scenes: tốn thời gian, 3 scenes (fern/fortress/trex) đủ cover easy/medium/hard
- **Kết quả:** 🔄 Pending — 6 runs × ~10 min, 2 GPU parallel ~30 min wall-clock

### [2026-04-18] Phase 4 future idea: SH-reliability signal trong CRS
- **Quyết định:** Ghi nhận ý tưởng user đề xuất, defer đến sau Phase 3
- **Lý do:**
  - Hiện tại CRS = position signal (D_i depth + R_i reprojection). Thêm S_i (SH reliability)
    làm CRS thực sự multi-dimensional (position + color) → novelty mạnh hơn, phân biệt
    với DropAnSH thuần
  - Integration vào CRS logit: `crs = scale × (w1·D_i + w2·R_i + w3·S_i - 0.5)`
- **Thay thế đã cân nhắc:**
  - Naive `||c_rest||` magnitude proxy: ❌ không phân biệt view-independent-valid (matte wall)
    vs memorize (specular drift). Magnitude nhỏ/lớn đều có thể là 1 trong 2.
  - Proxy tốt hơn: SH directional variance, train-novel render divergence, SH gradient
    variance late-stage
  - SH pruning condition riêng (tách khỏi CRS): ❌ làm CRS thành multi-signal phức tạp, khó
    paper framing
- **Kết quả:** 📋 Saved to memory (project_sh_reliability_crs_idea.md). Implement chỉ khi
  Phase 3 CRS-guided có tín hiệu dương. Warmup w3=0 đến iter ~3000 vì SH chưa ổn định.

---

### [2026-04-21] Phase 2 fair rerun: Phase 1 D1 = 21.12 là batch luck, thực là 20.95
- **Quyết định:** Revise D1 baseline từ 21.12 (Phase 1) xuống 20.95 (Phase 2 fair batch)
- **Lý do:**
  - Phase 2 chạy D1/D1-A/D1-S trong cùng batch trên 8 LLFF scenes
  - D1 fair = 20.95, lệch -0.17 so với Phase 1 batch (21.12)
  - 4/8 scene lệch > 0.2 dB giữa 2 batch (room -0.66, fortress -0.39, orchids -0.25, trex -0.21)
  - Batch variance (~0.15-0.20) > multi-seed std trong cùng batch (0.07-0.13)
- **Thay thế đã cân nhắc:**
  - Giữ Phase 1 number: reviewer sẽ bắt inconsistency khi thấy fair rerun khác
  - Chạy thêm multi-seed cho D1 full 8 scenes: tốn ~90 min, đã có multi-seed trên 3 hard scenes đủ
- **Kết quả:** ✅ Docs/10 Section 13.2 cập nhật D1=20.95. Section 12.9 thêm pointer SUPERSEDED.
  Multi-seed verify (Section 12.4) vẫn valid trên 3 hard scenes (+0.14 dB confirmed), nhưng
  không lan ra AVG 8 scenes.

### [2026-04-21] Phase 2 ablation: Anchor dropout dominant, SH degree dropout ngoại biên
- **Quyết định:** Anchor dropout là cơ chế chính trong DropAnSH, SH degree dropout bổ sung nhỏ
- **Lý do:**
  - Phase 2 3-scene: D1-A (anchor only) = 23.52 > D1 (both) = 23.40 > D1-S (SH deg only) = 22.46
  - D1-S NET HARMFUL khi standalone: thêm SH deg dropout không anchor mất 0.93 dB
  - Full 8-scene verify: D1-A = 20.93 ≈ D1 = 20.95 (tied, chỉ khác 0.02 dB)
  - D1-A wins 5/8 scene, D1 cứu **room** (+0.58) — SH deg dropout giúp specular indoor
- **Thay thế đã cân nhắc:**
  - Chọn D1-A là final (đơn giản hơn): risk lose room scene
  - Chọn D1 full (cứu room): thêm 3 flags, marginal AVG gain
- **Kết quả:** ✅ Lock D1 full là backbone — no catastrophic scene. D1-A = viable simpler variant.

### [2026-04-21] Phase 2b decomposition: Regularization pipeline đã chạm ceiling ~20.95
- **Quyết định:** Dừng optimize regularization axis, chuyển sang augmentation axis (opacity decay, extend iter, dense init)
- **Lý do:**
  - 5 configs (D1, D1-A, E1, E2, E4) test full 8 LLFF scenes
  - Tất cả AVG trong 0.1 dB: E4=21.02, D1=20.95, D1-A=20.93, E2=20.93, E1=20.92
  - Per-scene winner count: E1 (4/8), E4 (2/8), D1 (1), D1-A (1)
  - E4 (pure uniform, no anchor, no SH drop) tied với DropAnSH — đáng chú ý
  - Ceiling do base pipeline (COLMAP sparse + 10k iter + CRS + informed init) quyết định
- **Thay thế đã cân nhắc:**
  - Test thêm anchor sampling strategies: đã đủ data points, thêm không break ceiling
  - Test larger dropout rates: diminishing returns, có risk hại
- **Kết quả:** ✅ Ceiling ~20.95 confirmed. Roadmap pivot:
  - Phase 2c Opacity decay (G5 zombie Gaussian, G3 late-stage) → +0.3-0.8 dB
  - Phase 2d Extend iter 10k→30k (G8 budget) → +0.5-1.5 dB
  - Phase 5 Dense init PDCNet+ (G2 coverage) → +1.0-3.0 dB
  - Phase 4 SH-reliability CRS (G9 novelty) → +0.2-0.5 dB + defensible contribution

### [2026-04-21] Room scene outlier pattern: anchor-only hurts indoor specular
- **Quyết định:** Ghi nhận room scene cần treatment đặc biệt — chưa ra production change
- **Lý do:**
  - Room PSNR ranking: E4 (22.69) > E2 (22.38) > D1 (22.21) >> D1-A (21.62) > E1 (21.46)
  - Anchor-only configs (D1-A, E1) thua nặng room
  - SH regularization (E2 freeze, E4 uniform+sh=3) giúp room
  - Hypothesis: indoor specular-heavy scene → anchor cluster drop phá specular coherence
- **Thay thế đã cân nhắc:**
  - Scene-conditional regularizer (detect indoor → disable anchor): overengineering
  - Reduce anchor rate pa cho room: hack, không generalize
- **Kết quả:** 📋 Noted. Nếu opacity_decay Phase 2c không giúp room → revisit.

### [2026-04-21] Phase 3 CRS-guided anchor DEFERRED pending 6/9 view evaluation
- **Quyết định:** Hoãn Phase 3 — test trên ceilinged baseline uninformative
- **Lý do:**
  - Regularization ceiling ~20.95 trên 3-view — mọi CRS-guided variant khả năng cao trong noise (±0.1 dB)
  - Track B đã chỉ "uniform > targeted" trên position signal — CRS-guided anchor có risk tương tự
  - Phase 3 gain thực sự chỉ đo được khi baseline được raise (dense init hoặc extend iter)
  - 6/9 view evaluation cần trước để hiểu method scaling behavior
- **Thay thế đã cân nhắc:**
  - Chạy Phase 3 trên D1 + A1+B1β + E4 baseline (3 variants): rủi ro marginal gain, weaken paper story
  - Skip Phase 3 hoàn toàn: mất novelty contribution chính
- **Kết quả:** 📋 Saved to memory (project_crs_guided_anchor_idea.md). Resume khi:
  - (a) Đã có kết quả 6/9 view eval, HOẶC
  - (b) Baseline raise lên ~22-23 dB qua Phase 2c/2d/5

### [2026-04-21] Compute cost awareness từ Phase 2c
- **Quyết định:** Mọi comparison Phase 2c trở đi PHẢI ghi training time + N_Gaussian + peak VRAM
- **Lý do:**
  - Một số mechanism (opacity decay, dense init) thay đổi N_Gaussian đáng kể → không fair so
    PSNR alone
  - Phase 2b đã quan sát N_Gauss range 39k-232k giữa scene × config
  - Paper reviewer thường hỏi trade-off quality vs compute
- **Kết quả:** Saved to memory (feedback_measure_compute_cost.md). Parse scripts từ Phase 2c
  phải extract các metric này.

---

### [2026-04-21] Phase 2d Stage A density-aware: NEGATIVE — pivot to CRS-guided
- **Quyết định:** Abandon density-aware dropout direction (voxel + covariance). Pivot
  Phase 3 CRS-guided anchor selection.
- **Lý do:**
  - D1-A-V (voxel binning density): -0.17 dB AVG vs D1-A baseline 20.93 → 20.76
  - Thua 6/8 scenes (flower -1.02, leaves -0.52 nặng nhất — scene có texture chi tiết)
  - Thắng chỉ 2/8 (room/horns — scenes yếu baseline)
  - D1-A-C (covariance overlap) crashed do Bhattacharyya inf với degenerate Gaussian covariance
  - Compute cost: +8.3% wall-clock, +7.2% N_Gaussian → double loss
  - **Root cause:** density counts spatial proximity structure-only. Không phân biệt
    "dense legitimate" (flower petals cần nhiều Gauss render texture) vs "dense co-adapted"
    (floater cluster). Signal structure không adequate cho floater detection.
- **Thay thế đã cân nhắc:**
  - Fix covariance crash (SVD pseudo-inverse): effort không justified khi voxel đã fail
  - Try rendering top-K Stage B (CUDA mod): 1-2 ngày dev trên axis đã proven NEGATIVE
  - Combine density × CRS (reuse voxel code): becomes Phase 3β, đang test song song với 3α
- **Kết quả:** ✅ Direction abandoned. Phase 3 CRS-guided (pure + combined) đang chạy.
  Stage B CUDA top-K deferred permanently. Cleanup pending user approval.
  Chi tiết: docs/10 Section 14.

### [2026-04-21] Phase 3αβ CRS-guided anchor: parallel test pure CRS vs CRS × voxel
- **Quyết định:** Test 2 CRS-guided strategies song song:
  - 3α: `p(anchor) ∝ (1 - CRS_i)` — pure CRS signal
  - 3β: `p(anchor) ∝ voxel_density × (1 - CRS)` — combined quality × structure
- **Lý do:**
  - CRS = D_i + R_i là QUALITY signal, phân biệt floater (low CRS) vs surface (high CRS)
  - Addresses Stage A root cause: không drop dense legitimate surface (high CRS protects)
  - Reuse `utils/regularizer/density_voxel.py` từ Stage A cho 3β
  - **Risk reminder:** Track B B2 (uniform dropout × (1-CRS)) đã fail trước → CRS-guided
    UNIFORM không work. Nhưng ANCHOR khác UNIFORM: cluster drop quanh low-CRS có thể
    tạo áp lực khác với single-Gauss drop. Chưa biết có escape B2 failure không.
- **Thay thế đã cân nhắc:**
  - Chỉ 3α: miss chance stack 2 signals
  - Chỉ 3β: không biết CRS alone có đủ không
  - Serial (3α → 3β): lose parallel efficiency khi 2 GPU available
- **Kết quả:** ✅ Completed 2026-04-21. Verdict MIXED (effectively FLAT).
  - 3α (CRS-guided anchor): AVG +0.075 dB vs D1-A — trong noise, driven by room +0.919 outlier
  - 3β (CRS × voxel combined): AVG -0.202 dB, compute +122.9% — rejected
  - Ceiling 20.95 CHƯA break. D1 backbone locked.
  - Defensible claim: "CRS-guided anchor benefits indoor floater-prone scenes"
  - NOT defensible: "CRS-guided anchor > uniform anchor overall"
  - Chi tiết: docs/10 Section 14.9

### [2026-04-21] Phase 3 MIXED → priority elevate Phase 2c + Phase 5
- **Quyết định:** Tiếp tục break-ceiling attempts với 2 axis mới:
  - Phase 2c Opacity decay (HIGH priority, cheap)
  - Phase 5 Dense init PDCNet+ (HIGH priority, biggest lever)
- **Lý do:**
  - Phase 3 không delivered novelty boost kỳ vọng
  - Ceiling 20.95 trên regularization axis đã confirmed qua Phase 2b + Phase 3
  - Cần axis khác: continuous pruning pressure (opacity decay) hoặc init coverage (dense init)
  - Phase 5 importance ELEVATED: trước Phase 3 planning đánh giá 70% verify, giờ cần để compete
    SOTA numbers vì Phase 3 không cho edge
- **Thay thế đã cân nhắc:**
  - Skip 2c, thẳng Phase 5: miss cheap win, Phase 5 có thể fail
  - Phase 4 SH-reliability trước: novelty track nhưng không break ceiling PSNR
  - Multi-seed verify 3α: 3α room outlier đã biết, verify chỉ confirm scene-specific
- **Kết quả:** 🔄 Phase 2c prompt đã viết, launch độc lập. Phase 5 staged (Stage 1 infra có thể
  start song song). Phase 4 sau khi có PSNR baseline competitive.

---

### [2026-04-24] Phase 2c Opacity Decay: FIRST CEILING BREAK (+0.19 dB)
- **Quyết định:** Lock D1-O999 là new backbone (DropAnSH + opacity decay 0.999)
- **Lý do:**
  - D1-O999 = 21.13 AVG vs D1 baseline 20.95 → **+0.19 dB ngoài noise ±0.15**
  - Lần đầu config phá ceiling regularization 20.95 (confirmed through Phase 2b 5 variants)
  - 6/8 scene wins (room +0.40, fortress +0.39, orchids +0.30, trex +0.26)
  - Chỉ leaves rớt nhẹ -0.14 (fine texture cần Gaussian đầy đủ)
  - Compute overhead chỉ +5% time, N_Gauss gần như không đổi → double win
  - Mechanism fill gap "zombie Gaussian" mà CRS + legacy opacity không bắt
- **Thay thế đã cân nhắc:**
  - Factor=0.995 (Binocular3DGS default): +0.17 dB (tốt nhưng kém 0.999)
  - Factor=0.99 aggressive: -0.58 dB catastrophic
  - Extend densify (Binocular3DGS recipe): -0.24 dB, N_Gauss explode 2-8×
- **Kết quả:** ✅ BACKBONE UPDATED. D1 → D1-O999. Mọi experiment sau dùng backbone này.
  Chi tiết: docs/10 Section 15.

### [2026-04-24] Budget-scaling insight — đóng góp NOVEL supporting
- **Quyết định:** Opacity decay factor MUST scale với training budget — derive rule
- **Lý do:**
  - Binocular3DGS dùng 0.995 cho 30k iter → catastrophic khi dùng 10k (-0.58 dB)
  - Ta derive: 0.999 optimal cho 10k
  - Rule of thumb: `factor ≈ exp(ln(target_survival) / n_iter)` với target=0.01
  - Không paper nào note điều này → SUPPORTING novelty
- **Thay thế đã cân nhắc:**
  - Copy nguyên config Binocular3DGS: fail thảm với 10k budget
  - Grid search ngẫu nhiên: không có principled rule
- **Kết quả:** ✅ Paper-worthy insight ("first to derive budget-aware decay factor
  for sparse-view 3DGS"). Supporting contribution, không phải main.

### [2026-04-24] Extend-densify REJECTED cho sparse-view 10k
- **Quyết định:** KHÔNG dùng `densify_until_iter = iterations` style của Binocular3DGS
- **Lý do:**
  - D1-O995E test: PSNR -0.24, N_Gauss explode 2.6× avg (leaves 7.7×!)
  - Binocular3DGS cần extend vì 30k budget + decay aggressive 0.995
  - Sparse-view 10k budget + gentle decay 0.999 KHÔNG cần compensate population
  - CRS pruning + DropAnSH anchor đã maintain Gaussian health
- **Thay thế đã cân nhắc:**
  - Copy nguyên Binocular3DGS recipe: catastrophic
  - Partial extend (densify_until 7500): chưa test, likely cũng hại
- **Kết quả:** ✅ Negative result đáng paper — "Binocular3DGS's extend-densify
  does not generalize to sparse-view 10k training".

### [2026-04-24] CRS isolation ablation running — critical defensibility test
- **Quyết định:** Test D1-noCRS-O999 (opacity decay ON, CRS pruning OFF) TRƯỚC khi
  commit paper narrative với CRS-weighted decay hoặc bất kỳ CRS extension nào
- **Lý do:**
  - Risk: opacity decay có thể làm thay CRS's job (kill floater through gradient-less decay)
  - Nếu CRS redundant với decay → paper story sập (CRS là main contribution của paper)
  - 3 tests trước (Track B B2, Phase 3α, Phase 2d Stage A) đã cho thấy pattern
    "CRS weighting selection mechanism → marginal/fail" → cần chứng minh CRS có value
  - Cheap (8 runs × 6 min = 48 min, 2 GPU parallel ~30 min)
- **Thay thế đã cân nhắc:**
  - Skip ablation, commit direct: risk crisis nếu reviewer bắt bug
  - Multi-seed verify D1-O999 trước: answer question khác, không giải quyết CRS-decay synergy
- **Kết quả:** ⚠️ COMPLETED — Paper crisis triggered:
  - D1-noCRS-O999 = 21.21 dB AVG > D1-O999 = 21.13 dB
  - CRS pruning REMOVED → +0.08 dB tốt hơn (within noise nhưng không hại)
  - **CRS-as-pruning REDUNDANT với opacity decay** trên backbone hiện tại
  - → Triggered CRS diagnostic arc (xem entries [2026-04-25] đến [2026-05-04])
  Chi tiết: docs/10 Section 15.10, docs/11 Section 1-2.

---

### [2026-04-25] Tier A Diagnostic Suite — quantitative CRS signal characterization
- **Quyết định:** Implement diagnostic methodology để characterize CRS signal quality TRƯỚC khi
  thử thêm mechanism. Không tweak hyperparams nữa, phải hiểu signal.
- **Lý do:**
  - 3 CRS-mechanism failures (B2, Phase 3α, Phase 2d) + CRS isolation redundancy
    → cần evidence quantitative thay vì hand-wave "CRS có signal"
  - Nếu signal có ceiling → mọi mechanism mới sẽ fail → cần biết để pivot
  - Tier A = 4 quantitative tests trên 8 LLFF scenes:
    - **A1 BC bimodality** (Sarle's coefficient): D, R có shape signal không
    - **A2 D-R correlation**: 2 components có orthogonal không
    - **A3 synthetic floater discrimination**: perturb 0.3×depth_range, đo CRS response
    - **A4 occlusion contamination**: count Gaussian-view pairs với gauss_z > render_z×1.05
- **Thay thế đã cân nhắc:**
  - Skip diagnostic, thử mechanism mới: rủi ro lặp lại failure pattern
  - Single-scene smoke: sparse signal, không generalize
- **Kết quả:** ✅ Tier A done 8 scenes. Findings:
  - **D bimodal** (BC > 0.555 in 6/8 scenes) → signal có shape
  - **R bimodal** mostly (5/8 scenes)
  - **D-R orthogonal** (correlation < 0.3 in all scenes) → 2 signals capture different info
  - **A3 floater discrimination**: ✅ CRS DOES drop after perturbation (median Δ = -0.18)
  - **A4 occlusion contamination**: 36.5% R_i samples bị occluder colors → R noisy
  - **Conclusion**: Signal có discriminative power nhưng R contaminated. Mechanism failure
    không giải thích được hoàn toàn bởi signal quality alone.
  Chi tiết: docs/11 Section 2 (Tier A diagnostic).

---

### [2026-04-26..30] Six CRS mechanism variants — exhaustive ceiling test
- **Quyết định:** Test 6 CRS architectural variants để confirm/reject "mechanism failure"
  thay vì giả thuyết "signal failure"
- **Variants tested (8 LLFF scenes mỗi cái):**
  1. **6 CRS pruning ablations** (tau ∈ {0.20, 0.25, 0.30, 0.35, 0.40, 0.45}): all marginal/dead
  2. **C1 — occlusion-aware R**: depth-test trước khi sample GT pixel → AVG **−0.064 dB**
  3. **C1.5 — depth_range-relative tolerance sweep**: best −0.038 dB (still negative)
  4. **F-invisible — visibility-streak counter**: 1.3% catch rate, marginal
  5. **D1G — CRS-gated densification**: AVG +0.028 dB, **REDUNDANT with DECAY** (synergy −0.132)
  6. **hC — Hybrid RC × D fusion** (proxy RC = opacity × radii × coverage): AVG **−0.079 dB**
  7. **RNRC L3 — differentiable α-coupling** (proxy RC + EMA + α_eff = α × CRS.detach()):
     AVG +0.074 dB, STACK +0.163 < DECAY +0.186 alone
- **Pattern observed:** Δ ceiling +0.07 ± 0.10 dB **across mọi mechanism dùng D+R signal**.
  Bất kể gate / differentiable / loss-modulating → cùng plateau.
- **Conclusion:** Bottleneck là **SIGNAL D+R**, không phải mechanism class.
- **Thay thế đã cân nhắc:**
  - Stop sau 3 failures: dữ liệu chưa đủ rule out mechanism
  - Test thêm variants: diminishing returns sau 6 architectural axes
- **Kết quả:** ✅ Comprehensive negative evidence — defendable claim "consistency-based CRS
  fundamentally redundant with opacity regularization on this backbone".
  Chi tiết: docs/11 Section 3-4.

---

### [2026-04-30] Literature survey — breakthrough direction analysis
- **Quyết định:** Survey 8-15 sparse-view 3DGS papers (2024-2026) để identify novel CRS axis
  chưa exhausted, thay vì proposing variant #7 trên cùng axis
- **Lý do:**
  - 6 variants × 8 scenes ceiling +0.07 → cần fresh insight từ literature
  - Cần distinguish "axis exhausted" vs "mechanism class exhausted" vs "signal exhausted"
- **Survey scope:** ICO-GS, CoMapGS, BinocularGS, CuriGS, DropAnSH, DropGaussian, CoR-GS,
  3DGS-MCMC, PUP 3D-GS, UNG-GS, MVGSR, WildGS-SLAM, Predictive-PU-GS, TriaGS, Opt3DGS, Intern-GS
- **Findings:**
  - **SOTA**: ICO-GS 22.20 dB (cycle-depth + feature-MPC), CoMapGS 21.10 (MASt3R covisibility)
  - **CoMapGS gain +0.65 dB** purely từ per-pixel covisibility loss weight (CoR-GS base)
  - **ICO-GS gain +0.7+ dB** từ cycle-depth filter (vs plain monodepth)
  - **Architectural axes CRS chưa touched**:
    1. **Loss-path** (per-pixel reliability weight) — CoMapGS prove +0.65 với weaker signal
    2. **SH-degree gating per-Gaussian** — DropAnSH random +0.42, targeted chưa thử
    3. **Cross-view feature consistency loss** — ICO-GS pattern, CRS-soft mask chưa thử
    4. **Gradient-magnitude scaling** — chưa paper nào làm per-Gaussian
  - **Saturated axes**: opacity scaling (DECAY chiếm), densification gate (D1G fail), prune gate
- **Thay thế đã cân nhắc:**
  - Skip survey, thử variant #7: lặp pattern, rủi ro confirmation bias
  - Use survey để clone ICO-GS: không novel, chỉ catch up
- **Kết quả:** ✅ 4 candidate breakthrough proposals identified. Top recommendation:
  **Loss-path mechanism với signal upgrade**. Chi tiết: docs/11 Section 5.

---

### [2026-05-02] Signal redesign — D_cycle thay D_DAV2
- **Quyết định:** Replace D_DAV2 với D_cycle (cycle-depth consistency qua training views)
- **Lý do D_DAV2 broken:**
  - Phụ thuộc DAV2 (single-view monodepth, biased)
  - Chỉ check 1 view tại a time → no multi-view consistency enforcement
  - Static (DAV2 1 lần preprocessing, không update với training state)
- **Lý do D_cycle better:**
  - Dùng rendered depth từ chính Gaussian field → internal, no external prior
  - Cycle qua 2+ views → enforce multi-view consistency built-in
  - Dynamic (update mỗi CRS update qua rendered depth)
  - Occluded points tự nhiên break cycle → handle Gap #2 (occlusion contamination)
- **Công thức:**
  ```
  For pair (a, b):
    P_a = project(Gaussian → cam_a)
    d_a = rendered_depth(P_a)
    P_b = unproject_then_project(P_a, d_a, cam_a → cam_b)
    d_b = rendered_depth(P_b)
    P_a' = project_back(P_b, d_b, cam_b → cam_a)
    cycle_error = ||P_a − P_a'||
  D_cycle_i = exp(-mean_pairs(cycle_error_i) / σ)
  ```
  σ ≈ 5.0 pixels. Warmup 1000 iters (rendered depth chưa stable).
- **Thay thế đã cân nhắc:**
  - Foundation model signals (DINOv2 feature, MASt3R correspondence): cost cao
    (3-4 GB GPU memory + setup overhead). Defer Tier 1 nếu D_cycle work.
  - D_cycle với DAV2 fusion: complexity tăng, không clear win
- **Kết quả:** Pending — implement Tier 2-min (xem entry tiếp theo).

---

### [2026-05-04] Tier 2-min — DUAL upgrade: D_cycle + Loss Reweighter (Phase 7 plan)
- **Quyết định:** Test attempt với DUAL upgrade:
  - **Signal**: D → D_cycle (xem entry [2026-05-02])
  - **Mechanism**: gate prune → per-pixel loss reweighter (orthogonal axis với DECAY)
- **Mechanism details:**
  ```
  CRS_pix(p) = Σᵢ Tᵢ(p) · αᵢ(p) · CRSᵢ        (alpha-composite per-Gaussian CRS)
  w(p) = γ + (1-γ) · CRS_pix(p)               γ ≈ 0.5
  L_recon = Σ_p w(p) · |I_render(p) - I_GT(p)|
  ```
  Stop-gradient trên CRS_pix để tránh degenerate cycle.
- **Lý do dual upgrade:**
  - Single-axis tweak (signal only OR mechanism only) → không address ceiling root cause đa nhân
  - Address 4.5/6 gap simultaneously: #2 (cycle break occlusion), #3 (D_cycle dynamic),
    #4 (loss path gradient flow), #5 (no DAV2), #6 (pairwise aggregation)
  - Literature double precedent: D_cycle (ICO-GS +0.7), loss reweighter (CoMapGS +0.65)
- **Compute analysis:**
  - Training: ~+10% slowdown (D_cycle compute mỗi 100 iters)
  - GPU memory: +0 (no foundation model, no feature cache)
  - Render FPS inference: baseline (CRS chỉ dùng training)
  - Model storage: baseline (CRS không saved trong PLY)
- **Commitment criteria (BINARY):**
  - Δ_T2M ≥ +0.20 → 🟢🟢 BREAKTHROUGH, defendable contribution
  - Δ_T2M +0.10 ~ +0.20 → 🟢 Solid, ship hoặc upgrade Tier 1 (DINOv2/MASt3R)
  - Δ_T2M 0 ~ +0.10 → 🟡 Marginal, pivot recipe paper
  - Δ_T2M ≤ 0 → 🔴 Dead, pivot decisively. KHÔNG đề xuất variant #8.
- **Ablation matrix (24 NEW runs, ~1.5-2h trên 2 GPU):**
  - DCYCLE: --use_d_cycle --crs_prune (signal upgrade alone)
  - LWEIGHT: --use_loss_reweight (mechanism upgrade alone, D+R cũ)
  - TIER2MIN: cả 2 flags (full)
  - B0 reuse từ logs cũ
- **Thay thế đã cân nhắc:**
  - Tier 1 Full FAMR (DINOv2 + MASt3R): cost 4-5 ngày, 3-4 GB GPU memory bump,
    risk OOM trên consumer GPU
  - Tier 3 Vanilla A (mechanism only, D+R cũ): probability thấp hơn (~30% vs ~40%),
    không address signal ceiling
  - Pivot ngay không Tier 2-min: bỏ lỡ chance signal upgrade
- **Kết quả:** Implemented + executed → xem [2026-05-05] Phase 7 result entries.
  Chi tiết: docs/11 Section 6 (Tier 2-min plan).

---

### [2026-05-05] Phase 7 Tier 2-min EXECUTED — Phase 5 weak backbone result
- **Quyết định:** Run Tier 2-min ablation (4 configs) trên Phase 5 weak backbone (B0=20.234)
- **Implementation:** `utils/crs/d_cycle.py` (cycle-depth helpers), `utils/crs/crs_module.py`
  thêm `render_crs_map` qua color-swap trick (override_color path), `train.py` loss reweighter
  block. KHÔNG sửa CUDA.
- **Result (8 LLFF scenes):**
  | Config | AVG | Δ vs B0 | Per-scene |
  |--------|-----|---------|-----------|
  | B0 | 20.234 | 0 | reference |
  | DCYCLE alone | 20.359 | **+0.125** ⭐ | FIRST CRS variant với clean positive across 9 attempts |
  | LWEIGHT alone | 20.228 | −0.007 | mechanism neutral |
  | TIER2MIN combined | 20.269 | +0.035 | synergy −0.083 (redundant) |
- **Compute:** +0.1-0.2% slowdown across configs, GPU memory unchanged.
- **Key findings:**
  1. **D_cycle là first signal upgrade clean positive** — beats 8 prior CRS variants ceiling +0.07
  2. **Loss reweighter mechanism alone neutral** trên Phase 5 weak backbone
  3. **Combined TIER2MIN destructive** — synergy −0.083, LW interferes với D_cycle effect
  4. **Per-scene heterogeneity high**: D_cycle wins floater-prone (trex +0.345, horns +0.327),
     fails detail/photometric (room −0.067 với LW −0.699 catastrophic)
- **Verdict:** 🔴 STOP per +0.15 commitment threshold (DCYCLE +0.125 < +0.15) trên backbone này.
  Nhưng **D_cycle is real signal** → cần verify scale lên strong backbone.
- **Pending:** Phase 7 Stage 1 trên D1-O999 strong backbone.

---

### [2026-05-06] Phase 7 Stage 1 — D_cycle on D1-O999 strong backbone
- **Quyết định:** Test Tier 2-min components trên D1-O999 strong backbone (=21.13 với CRS pruning)
  với 3-config attribution control:
  - TIER1_DAV2_GATE: D_DAV2 + R_old + CRS prune (reference)
  - TIER1_DC_GATE: D_cycle + R_old + CRS prune (signal isolation)
  - TIER1_DC_LW: D_cycle + R_old + Loss Reweighter (signal + mechanism, gate replaced)
- **Lý do attribution control:** Phase 6 ceiling failure could be EITHER signal OR mechanism
  bottleneck. Cần 3-config để tách (Issue 2 raised by execution session review).
- **Result (8 LLFF scenes):**
  | Config | AVG | Δ vs reference | Note |
  |--------|-----|----------------|------|
  | TIER1_DAV2_GATE | 21.178 | 0 | matches Phase 6 D1-O999 ~21.13 + run variance |
  | TIER1_DC_GATE | 21.163 | **−0.015** | D_cycle FLIPS NEGATIVE on strong backbone |
  | TIER1_DC_LW | 21.228 | +0.050 | Combined slightly positive |
  | vs No-CRS (21.21) | — | +0.018 | within noise |
- **Critical finding:** **D_cycle effect flips sign across backbones**:
  - Phase 5 weak: +0.125 (positive)
  - D1-O999 strong: −0.015 (negative)
  - → D_cycle fills regularization gap on weak backbone, REDUNDANT với DropAnSH+DECAY trên strong backbone
- **Per-scene pattern:** D_cycle helps floater-prone (horns +0.106, room +0.139), hurts detail
  (fern −0.070, leaves −0.138, fortress mixed)
- **Attribution diagnostic:**
  - Δ_DC_GATE (signal, gate kept): −0.015 (signal alone dead trên strong backbone)
  - Δ_LW_DC (mechanism, D_cycle kept): +0.065 (LW helps but small)
  - Δ_combined: +0.050 (within noise vs no-CRS)
- **Compute:** DC_GATE +2.7%, DC_LW +4.8% slowdown — reasonable.
- **Verdict:** 🟡 Marginal per analyzer. **Honest interpretation: H_both_dead** —
  signal + mechanism đều saturated trên strong backbone với D+R formula intact.
- **Hypothesis cho Phase 8:** R contamination 36.5% may be diluting D_cycle. Fix R first.

---

### [2026-05-07] Phase 8 — Formula Redesign + SH Path 🎉 BREAKTHROUGH
- **Quyết định:** 3 sub-phases simultaneously với 5-config ablation trên D1-O999 backbone:
  - **8a: R_visible** (visibility-aware reprojection consistency, fix 36.5% occlusion contamination)
  - **8b: S_stability** (SH coefficient EMA variance signal — multi-dim CRS adding color path)
  - **8c: CRS-modulated SH freeze** (per-Gaussian targeted freeze replace global Track A1)
- **Implementation:**
  - `utils/crs/sh_stability.py` (NEW): EMA variance tracking
  - `utils/crs/sh_freeze.py` (NEW): per-Gaussian gradient zeroing on _features_rest
  - `utils/crs/crs_module.py`: R_visible logic + multi-component CRS formula
  - `train.py`: hooks for S update + CRS-mod SH freeze (after backward, before optimizer.step)
- **Formula upgrade:**
  ```
  CRS_new = sigmoid(scale × (w_d·D_cycle + w_r·R_visible + w_s·S_stability − threshold))
  ```
  3 components address 3 known gaps simultaneously.
- **Result (8 LLFF scenes, 5 configs × 8 = 40 runs):**
  | Config | AVG | Component changed | Δ |
  |--------|-----|-------------------|---|
  | OLD | 21.178 | reference | 0 |
  | FIX_R_DAV2 | 21.068 | + R_visible (fix R) | **−0.111** ❌ R alone HURTS |
  | FIX_R_DC | 21.169 | + D_cycle (with clean R) | +0.102 ✅ D works in clean R |
  | FIX_RS | 21.146 | + S_stability | −0.023 ⚪ S adds nothing |
  | **FULL** | **21.335** | + CRS-mod SH freeze | **+0.189** ✅ BIGGEST winner |
  | **Δ_FULL vs OLD** | — | combined Phase 8 | **+0.156 dB** |
  | **Δ_FULL vs No-CRS (21.21)** | — | first beat no-CRS | **+0.125 dB** ⭐ |
- **Compute:** FULL +3.7% slowdown, GPU memory unchanged, render FPS unchanged.
- **Per-scene pattern (FULL vs OLD):** 5/8 wins
  - Big wins: room +0.544, horns +0.361, fortress +0.344
  - Small wins: flower +0.103, leaves +0.019
  - Marginal loss: fern −0.075, orchids −0.040, trex −0.005
  - Pattern: scenes có nhiều SH-driven color drift benefit nhất từ targeted freeze
- **🎉 KEY MILESTONE:** **First CRS variant beat no-CRS recipe by meaningful margin** (+0.125 dB).
  After 9 prior CRS attempts ceiling at +0.07, Phase 8 finally exceeds.
- **Attribution insights:**
  1. **Δ_R = −0.111 dB SURPRISE:** R_visible alone hurts! Visibility filter aggressive →
     too many Gaussians get neutral fallback → data loss > noise reduction
  2. **Δ_D = +0.102 dB VINDICATED:** D_cycle works when R is clean (Stage 1 R noise was diluting)
  3. **Δ_S = −0.023 dB DISAPPOINTING:** SH stability EMA variance không add useful info
  4. **Δ_M = +0.189 dB WINNER:** CRS-modulated SH freeze (per-Gaussian targeted) > global freeze
- **Verdict:** 🟡 SOLID — close DOC-GS gap (21.38 vs 21.335 = -0.045, within noise), still −0.865
  to ICO-GS SOTA. **First defendable CRS contribution** for paper.
- **Pending decisions:**
  1. Drop R or simplify formula? (Phase 9 Test 1)
  2. SH freeze mechanism universal? (Phase 9 Test 2 cross-backbone A1+B1β)
  3. Stack with big lever (dense init / feature MPC) for SOTA? (future Phase 10)
- **Chi tiết:** docs/11 Section 7 (Phase 8 result + attribution).

---

### [2026-05-08] Phase 9 EXECUTED — Simplification + Cross-backbone (3 hypotheses ALL non-trivial)
- **Quyết định:** 5-config ablation test 3 hypotheses từ Phase 8 attribution:
  - **H1 (drop R):** R_visible component có thể đang dilute D signal — test D-only formula
  - **H2 (drop S):** S_stability adds nothing (Δ_S = −0.023) — simplify recipe
  - **H3 (cross-backbone):** CRS-mod SH freeze universal mechanism? Test on A1+B1β backbone
- **Test 1 — D1-O999 simplification (3 NEW configs, FULL reuse):**
  | Tag | Components | Tests |
  |-----|-----------|-------|
  | FULL (reuse Phase 8) | D + R + S + CRS-mod-freeze | reference 21.335 |
  | FULL_NoS | D + R + CRS-mod-freeze | H2: drop S |
  | D_ONLY_FREEZE | D + CRS-mod-freeze (no R, no S) | H1: drop R |
  | D_ONLY_GATE | D + CRS prune (no R, no S, no SH freeze) | isolate D alone |
- **Test 2 — A1+B1β cross-backbone (1 NEW config):**
  | Tag | Backbone | Components | Tests |
  |-----|----------|-----------|-------|
  | A1B1_BASELINE | A1+B1β + global SH freeze | reference (~20.96) |
  | A1B1_BEST | A1+B1β (NO global freeze) + R_visible + D_cycle + CRS-mod freeze | H3: replace global with selective |
- **Implementation needs:**
  - `--disable_r_signal` flag: skip R compute, w_d=1 in formula (D-only support)
  - `--disable_global_sh_freeze` flag: ignore Track A1 freeze when CRS-mod active (cho A1+B1β backbone test)
  - 2 small flag additions to arguments + small logic changes in update_crs / train.py
- **Cost:** 32 NEW runs (4 NEW × 8 scenes + A1B1_BASELINE + A1B1_BEST × 8 scenes) ~100 phút 2 GPU
- **Verdict tree:**
  - H1 confirmed (drop R win ≥ +0.10) → simplify to D-only recipe
  - H2 confirmed (drop S neutral ±0.05) → drop S, cleaner formula
  - H3 confirmed (A1B1_BEST > baseline ≥ +0.15) → SH freeze universal mechanism
  - Best across all configs > 21.45 → close BinocularGS, push toward SOTA
- **Result Test 1 (D1-O999 simplification, 8 scenes):**
  | Config | AVG | Δ vs FULL | Note |
  |--------|-----|-----------|------|
  | **FULL (Phase 8 reference)** | **21.335** | 0 | **BEST** — recipe locked |
  | D_ONLY_GATE (D + gate prune, no SH freeze) | 21.242 | −0.093 | drop SH freeze hurts |
  | FULL_NoS (drop S) | 21.200 | −0.135 | **drop S HURTS more than alone-effect predicted** |
  | D_ONLY_FREEZE (drop R + S) | 21.159 | −0.176 | drop both worst |
- **Result Test 2 (A1+B1β cross-backbone, 8 scenes):**
  | Config | AVG | Δ |
  |--------|-----|---|
  | A1B1_BASELINE | 20.932 | reference (~Track A+B 20.96 ✓) |
  | A1B1_BEST | 20.983 | +0.051 |
- **Verdict:**
  - ❌ **H1 REJECTED**: R contributes (Δ_NoR = −0.041 in leave-one-out)
  - ❌ **H2 REJECTED**: S contributes via synergy (Δ_NoS = −0.135 — much worse than Phase 8 alone-effect Δ_S = −0.023)
  - 🟡 **H3 PARTIAL**: SH freeze mechanism works on A1+B1β but smaller gain (+0.051 vs +0.189 on D1-O999)
- **🔑 KEY INSIGHT — Sequential vs Leave-one-out attribution differs:**
  - Phase 8 sequential: Δ_S = −0.023 (S adds nothing **alone**)
  - Phase 9 leave-one-out: Δ_NoS = −0.135 (S **synergize** với mechanism)
  - → All 4 components (D, R, S, mechanism) **have non-trivial synergy** in FULL recipe
  - → Combination > sum of parts. **Don't simplify.**
- **Compute insight:**
  | Config | AVG train (s) | Backbone |
  |--------|---------------|----------|
  | A1B1_BEST | 192.0 (FASTER!) | A1+B1β |
  | A1B1_BASELINE | 205.2 | A1+B1β |
  | D1-O999 FULL | 360.7 | D1-O999 (DropAnSH overhead) |
  → A1+B1β backbone ~2× faster nhưng PSNR thấp hơn 0.35 dB. D1-O999 đáng overhead.
- **Final conclusion**: Phase 8 FULL recipe (D_cycle + R_visible + S_stability + CRS-mod SH freeze trên D1-O999 backbone) là **optimal**. **CRS axis exhausted** ở 21.335 dB.
- **SH freeze backbone-aware**: bigger gain trên D1-O999 (DropAnSH) than A1+B1β (sh=1+global freeze). Mechanism effective khi backbone không có SH-control sẵn.

---

### [2026-05-09] CRS axis EXHAUSTED — Phase 10 needed for SOTA gap
- **Quyết định**: Lock Phase 8 FULL recipe. Pivot Phase 10 với orthogonal axis.
- **State sau Phase 9:**
  - Best CRS variant: **21.335 dB** (Phase 8 FULL = D_cycle + R_visible + S_stability + CRS-mod freeze on D1-O999)
  - vs No-CRS (21.21): **+0.125 dB** (first CRS contribution defendable)
  - vs DOC-GS (21.38): **−0.045** (within noise, essentially tied)
  - vs BinocularGS (21.44): **−0.105**
  - vs ICO-GS SOTA (22.20): **−0.865** (still big gap)
- **CRS axis saturation evidence:**
  - 9 architectural variants tested (Phase 6-9)
  - Phase 8 FULL achieves +0.125 — meaningful but capped
  - Phase 9 confirms simplification hurts → recipe optimal
  - Cross-backbone test: SH freeze partial universal, smaller gains elsewhere
- **For SOTA gap closure** cần lever ngoài CRS:
  - **Phase 10A — Dense init** (DUSt3R/MASt3R): +1.0-3.0 dB expected, 1-2 ngày impl
  - **Phase 10B — Feature MPC** (DINO consistency): +0.3-0.7 dB, 2-3 ngày impl
  - **Phase 10AB — Stack**: combined +1.5-3.5 dB, 3-4 ngày
- **Trade-off awareness:**
  - Dense init = orthogonal lever, dilutes "CRS contribution" narrative
  - Feature MPC = ICO-GS path, novelty boundary unclear
  - Pure CRS path: Phase 8 +0.125 đã là maximum, không thể push thêm trong scope
- **Pending decision**: User chọn Phase 10 path (dense init / feature MPC / stack / khác).

---

### [2026-05-07] Phase 10A — DUSt3R Dense Init FAIL hard, axis DEAD
- **Quyết định**: Bỏ hẳn hướng foundation-model dense init (DUSt3R/MASt3R) cho initial PC. Phase 10A code giữ default OFF, sẽ cleanup sau.
- **Implementation:**
  - DUSt3R clone local Windows + install env riêng trên server
  - Pre-compute dense PC 8 scenes via `scripts/precompute_dust3r.py` (~25 min cache build)
  - 16-run ablation (AUGMENT/REPLACE × 8 scenes) trên Phase 8 FULL backbone
  - 6-run diagnostic (FILTER/DENSIFY/BOTH × orchids/leaves) cho hyperparameter tune
- **Phase 10A main results:**
  | Config | AVG PSNR | Δ vs P8_FULL | Init Gauss | Final Gauss | Time |
  |--------|----------|--------------|------------|-------------|------|
  | P8_FULL (ref) | 21.335 | 0 | ~3K | 87K | 361s |
  | AUGMENT | 20.436 | **−0.898** | 63K | 125K | 498s (+38%) |
  | REPLACE | 17.805 | **−3.529** | 49K | 134K | 548s (+52%) |
- **Diagnostic results (Option B — 2 scenes × 3 configs):**
  - Best: orchids FILTER (conf 3.0, max 10K, dedupe 0.05) Δ=−0.074 (within noise of 0, NOT ≥ +0.05)
  - DENSIFY scaling consistently hurts (over-aggressive dropansh_pa 0.05)
  - leaves catastrophic on all 3 configs (textureless foliage, DUSt3R fails)
  - **Ceiling ≈ −0.07 dB even with optimal filter** → systematic failure, not tuning
- **Verdict:** Phase 10A axis DEAD.
  - ❌ H1 (filter alone): closest to noise floor but never ≥ +0.05 → REJECTED
  - ❌ H2 (densify scaling): hurts both scenes → REJECTED
  - ❌ H3 (combined): cancellation effect → REJECTED
- **Root cause hypothesis:**
  - DUSt3R points alignment với COLMAP frame có residual error không khắc phục được bằng hyperparameter
  - Phase 8 recipe calibrated cho ~3K sparse init points → 50K dense init phá vỡ densify/prune dynamics
  - DUSt3R noise (especially textureless/foliage scenes) inflates Gaussian count nhưng không cải thiện accuracy
- **Decision:** PIVOT khỏi initial-PC axis. Foundation-model dense init nói chung BỎ HẲN (DUSt3R, MASt3R đều cùng class).
- **Cleanup:** xóa `output/p10a/`, DUSt3R env, checkpoint, source clone, cache. Code Phase 10A giữ tạm với default OFF cho paper "we tried this" reference.

---

### [2026-05-09] cuDNN+atomicAdd variance discovery + Multi-seed protocol locked
- **Quyết định**: Mọi ablation từ Phase 11 trở đi phải multi-seed (3 seeds × 8 scenes paired). Min detectable Δ ≈ ±0.10 dB.
- **Discovery**: 3 runs serial cùng seed=42 cùng code → range **1.31 dB** trên room (21.65, 22.96, 22.64). Variance source:
  - 3DGS rasterizer dùng `diff-gaussian-rasterization` với CUDA `atomicAdd` trong forward/backward
  - atomicAdd race condition → float sum không associative → bit-by-bit divergence
  - 10k iter accumulate → PSNR ±0.5-1.3 dB single-scene
  - `cudnn.deterministic=True` + `cudnn.benchmark=False` KHÔNG đủ (atomicAdd hardware-level non-det)
- **Hệ quả retrospective**:
  - Phase 8 paper 21.335 là 1 sample (có thể lucky cao)
  - "Drift -0.20 vs paper" trong p8_rerun = noise, không phải code regression
  - Bisect attempts unfeasible (variance > drift signal)
- **Variance bands locked cho project**:
  | Setup | Variance | Min Δ detectable |
  |-------|----------|------------------|
  | 1 scene × 1 seed | ±1.3 dB | ±1.0 dB |
  | 8 scenes × 1 seed (avg) | ±0.46 dB | ±0.30 dB |
  | 8 scenes × 3 seeds (paired) | ±0.10 dB | ±0.10 dB |
- **Multi-seed pattern**: 3 seeds (42, 137, 9999) × 8 scenes × {A0, A1} = 48 runs ~2.5h on 2 GPUs
- **Paired Δ comparison**: A0 và A1 cùng (seed, scene) → cancel common-mode variance
- **Verdict format**: N samples, Δ_mean ± SEM, 95% CI, significant if 0 ∉ CI

---

### [2026-05-09] Phase 11 Step 1 (Covisibility Reweight) — MARGINAL → KEEP code default OFF
- **Quyết định**: Step 1 cross-batch effect ≈ 0, marginal positive trong batch 2. Keep code default OFF, document as MARGINAL.
- **Multi-seed batch 1 (Step 1 alone)**:
  - N=24, Δ_mean = **−0.028 dB**, SEM = 0.047, 95% CI [−0.120, +0.063]
  - Per-seed Δ_8avg: −0.013, −0.100, +0.028 (mixed direction)
  - Verdict: NOT significant
- **Multi-seed batch 2 (Step 1 in 3-config factorial A0/A1/A2)**:
  - N=24, Δ_S1 mean = **+0.055 dB**, SEM = 0.039, 95% CI [−0.021, +0.132]
  - Per-seed Δ_8avg: +0.059, +0.100, +0.007 (all positive, borderline)
  - Verdict: borderline (p_one-tail ≈ 0.08)
- **Cross-batch combined (N=48)**: Δ_mean ≈ **+0.0135 dB** → effectively zero
- **Per-scene pattern**: room +0.355 (batch 2 only) but A0_std=0.51 (noisiest scene) → flip-flop direction across batches → noise dominate
- **Decision**: MARGINAL — code không reject hoàn toàn, không commit. Default OFF, để paper writeup note "we explored covisibility reweight, signal within noise floor".

---

### [2026-05-09] Phase 11 Step 2 (Perceptual DINO same-view) — REJECTED
- **Quyết định**: Reject Step 2. Cleanup pending (KHÔNG block Step 4/5).
- **Multi-seed**: N=24, Δ_mean = **−0.046 dB**, SEM = 0.056, 95% CI [−0.155, +0.064]
  - Per-seed Δ_8avg: −0.016, +0.047, −0.168 (inconsistent direction)
  - Verdict: NOT significant, slight negative
- **Per-scene pattern**: 5/8 negative, 3/8 small positive. **room −0.347** worst hit (vs +0.20 trong Step 1 → flip-flop confirm noise)
- **Implementation**: timm.create_model('vit_small_patch14_dinov2.lvd142m') — official Meta weights, Python 3.8 compatible

---

### [2026-05-09] Phase 11 Stack (Step 1 + Step 2) — REJECTED, no synergy
- **Quyết định**: Stack S1+S2 không synergize. Phase 7 LWEIGHT precedent confirmed: similar mechanisms (per-pixel/patch reweight) không stack additive.
- **Multi-seed factorial N=24**:
  - Δ_S1 (A1−A0) = +0.055 ± 0.039
  - **Δ_Stack (A2−A0) = −0.007 ± 0.043** (effectively zero)
  - **Δ_Synergy (A2−A1) = −0.063 ± 0.052** (Step 2 trending HURT Step 1)
- **Implication**: 2 perceptual-class mechanisms cancel each other. Feature/perceptual class **exhausted** trên Phase 8 FULL backbone.
- **Pivot direction**: Anti-overfit + geometric mechanism (different axis):
  - Step 4: Cross-view feature MPC (DINO + depth warping) — geometric, ICO-GS adapted
  - Step 5: TV depth edge-preserving regularizer — direct floater suppression

---

### [2026-05-09] Test blur diagnosis → anti-overfit framework
- **Quyết định**: Pivot khỏi pixel-level loss enhancements (edge-aware photometric REJECTED before test). Anti-overfit + geometric path.
- **User observation**: Color tone ≈ GT trên test renders, nhưng **blur ở high-detail regions**
- **Diagnosis**:
  - Train PSNR ~36-38 dB vs Test ~21 dB → gap 15-18 dB = structural overfit
  - Test blur causes ranked:
    1. **Floater render lệch** (Gaussian wrong-depth) — main culprit
    2. Gaussian scale over-large → over-smooth
    3. SH memorize training colors → no generalize
- **REJECTED hướng pro-overfit** (would widen gap):
  - Edge-aware photometric (gradient-weighted L1) — pro-overfit, increases train detail learning
  - Sobel gradient loss — same class
  - Per-pixel L1 depth — Phase 4.1 đã reject -3 dB
  - MS-SSIM — still pixel-level training view
- **APPROVED hướng anti-overfit**:
  - **Step 4 cross-view MPC** (~25-35%): depth warping forces correct Gaussian position → eliminate floater
  - **Step 5 TV depth edge-preserving** (~30-40%): smooth depth field nơi image smooth, giữ edge tại object boundary → direct floater suppression
- **Decision tree với stack negative warning**:
  - Combined ≥ max(alone) + 0.05 → SHIP STACK
  - Combined ≥ max(alone)        → SHIP MAX SINGLE (simpler)
  - Combined < max(alone)        → ⚠️ STACK NEGATIVE → SHIP MAX SINGLE
  - Cả 2 < +0.10                 → ACCEPT CEILING, paper writeup

---

### [2026-05-08] Phase 11 — Loss-axis Exploration (post-Phase-10A pivot)
- **Quyết định**: Pivot sang loss-axis với external supervision. Sequential evaluation strategy (1 step at a time, abort early if win).
- **Lý do:**
  - Phase 10A confirmed initial-PC axis dead (DUSt3R, MASt3R loại)
  - CRS axis exhausted ở 21.335 (Phase 9)
  - Loss-axis với external signal là direction còn lại trước khi accept ceiling
- **4 candidates xếp theo cost-effectiveness:**
  | # | Pick | Cost | Probability ≥+0.20 | Direct precedent |
  |---|------|------|---------------------|------------------|
  | Step 1 | CRS × Covisibility reweight (depth-based, KHÔNG dùng DUSt3R) | 0.5 ngày | 30-35% | CoMapGS +0.65 trên CoR-GS |
  | Step 2 | Same-view perceptual loss (DINOv2 ViT-S) | 0.5 ngày | 15-25% | LPIPS pattern |
  | Step 3 | R_feature replace R_visible (CRS signal upgrade) | 0.5-1 ngày | 20-30% | Novel |
  | Step 4 | True cross-view MPC (forward warp features) | 1-2 ngày | 40-50% | ICO-GS +0.4-0.7 |
- **Sequential strategy (KHÔNG stack-3 như đề xuất ban đầu):**
  - Lý do reject stack: Phase 7 precedent — stack-without-diagnostic gây partial cancel −0.083
  - Sequential cho phép abort sớm nếu Step X win → save partial cost
  - Best case 1 ngày (Step 1 win), worst case 4 ngày (all fail)
- **Step 1 specifics — Option C (depth-based, KHÔNG cần DUSt3R cache):**
  - Forward warp aligned DAV2 depth từ cam A → cam B → check in-bounds + depth consistency
  - Cov_A[p] = số views B covisible với pixel p của A
  - Reweight L_phot: w(p) = γ + (1−γ) · min(cov_norm, CRS_pix), γ=0.3
  - Test diagnostic 1 scene (orchids), Δ ≥ +0.20 → confirm 2 scenes → scale 8
- **Pivot plan ready (nếu Step 1-3 fail):**
  - Regularization losses (smoothness/sparsity) — 0.5-1 ngày, ~25%
  - Depth prior upgrade (DAV2 fine-tune) — 1-2 ngày, ~25%
  - Render-side tricks (anti-aliasing) — 0.5 ngày, ~15%
  - Accept ceiling, write up Phase 8 FULL ở 21.335 dB
- **Workflow rules established:**
  - Planning session draft prompt only, KHÔNG Write code production trực tiếp
  - Mọi ablation parallelize 2 GPUs với `&` + `wait`
  - Diagnostic 1 scene first → confirm 2-3 scenes → scale 8

---

### [2026-05-11] Phase 11 Step 5 (TV depth edge-preserving) — REJECTED
- **Quyết định**: Reject Step 5. Cleanup pending (Rule 13).
- **Multi-seed N=24 paired**:
  - Δ_mean = **−0.0262 dB**, SEM = 0.051, 95% CI [−0.126, +0.074]
  - Verdict: NOT significant
- **Per-seed Δ_8avg** (3 seeds):
  - seed 42: +0.093
  - seed 137: −0.053
  - seed 9999: −0.119
  - Inconsistent direction → noise dominate
- **Per-scene Δ pattern**: 5/8 negative, 3/8 positive
  - Positive: flower +0.245, leaves +0.200, room −0.045 (after complete)
  - Negative: horns −0.249, trex −0.183, fortress −0.100, fern −0.054, orchids −0.024
- **A0 mean (N=24, balanced)**: 21.185 (consistent với previous batches)
- **A1 mean (N=24)**: 21.158 → no improvement
- **Honest probability**: TV depth was anti-overfit candidate. Failure suggests Phase 8 FULL backbone near-optimal cho sparse-view 3DGS với current rasterizer.

---

### [2026-05-11] Phase 11 LOSS-AXIS EXHAUSTED — 6/6 attempts REJECTED
- **Final state**: 6 Phase 11 mechanism classes tested multi-seed N=24 paired, none significant
- **Comprehensive ablation table**:
  | Step | Mechanism | Δ paired | Verdict |
  |------|-----------|---------|---------|
  | Step 1 (Covisibility reweight) cross-batch | Per-pixel weight cov×CRS | +0.014 | MARGINAL keep OFF |
  | Step 2 (Perceptual DINO same-view) | Loss term feature distance | −0.046 | REJECTED |
  | Stack S1+S2 | Combined reweight | −0.007 | REJECTED |
  | Step 4 (Cross-view feature MPC) | Geometric + DINO warp | −0.042 | REJECTED |
  | Step 5 (TV depth edge-preserving) | Anti-overfit regularizer | −0.026 | REJECTED |
- **Highest A1 mean achieved**: Step 1 batch 2 = 21.186 (single batch lucky, Δ=+0.055 borderline)
- **No A1 mean robustly above 21.20** → ceiling ≈ 21.18 ± 0.05
- **Methodology validated**: Multi-seed N=24 paired min detectable Δ ±0.10 dB; atomicAdd variance ±1.3 dB single-scene cancelled by paired comparison
- **Anti-overfit framework validated**: REJECTED pro-overfit candidates before testing → save cost
- **Implication**: Phase 8 FULL recipe = practical ceiling cho CRSGaussian backbone LLFF 3-view

---

### [2026-05-11] Code regression confirmed — Phase 8 paper 21.335 NOT reproducible
- **Evidence**: 5 multi-seed batches N=120 consistent A0 ≈ 21.16-21.20
  - p8_rerun 21.150, p11s1 A0 21.156, p11s2 A0 21.196, p11s12 A0 21.131, p11s4 A0 21.18
- **Gap to paper**: −0.155 dB systematic (5/5 batches below paper)
- **Probability pure noise**: < 5% → real code regression
- **Root cause**: Commit `0511edd` (May 9) message literally states "mất config phase 8 full được 21.335"
- **Suspect files**: `gaussian_renderer/__init__.py` +101 lines (combined dropout B1+DropAnSH refactor, MOST suspect), `scene/gaussian_model.py` +44, `scene/__init__.py` +6 (already reverted)
- **NOT caused by xformers fiasco (May 10)**: pre-xformers 21.150 ≈ post-rollback 21.18
- **Unable to bisect**: No git snapshot of Phase 8 ablation working tree (May 4)
- **Decision**: Accept current baseline 21.18 multi-seed mean. Phase 8 paper 21.335 reproducible từ saved PLYs `output/p8/FULL_*/point_cloud/iteration_10000/`
- **Implication for Phase 11**: Paired Δ within-batch cancels common-mode regression → all 6 verdicts REMAIN VALID

---

### [2026-05-11] Decision: Continue improvement, NOT writeup yet
- **User stance**: Không writeup vội, tiếp tục cải thiện
- **Untouched directions ranked**:
  | Direction | Probability ≥+0.10 | Cost | Class |
  |-----------|---------------------|------|-------|
  | Mip-Splatting anti-aliasing | ~30-40% | 2-3 ngày | Rendering trick |
  | Iter budget 15k | ~25-30% | 0 code | Hyperparameter |
  | Visibility-based prune | ~20-25% | 0.5 ngày | Anti-overfit prune |
  | Anisotropy regularizer | ~15-20% | 0.5 ngày | Anti-overfit shape |
  | Soft scale regularizer | ~15-20% | 0.3 ngày | Anti-overfit no position |
  | CRS-pull (user idea) | ~15% | 1-2 ngày | Position-axis, risky |
  | Density-aware densify | ~15-20% | 1 ngày | Untested |
- **Track approach (parallel)**:
  - Track 1 CHEAP: Iter 15k test → 0 code, instant
  - Track 2 MEDIUM: Visibility-prune nếu Track 1 neutral
  - Track 3 HIGH: Mip-Splatting nếu cả 2 fail
- **Cleanup planned (Rule 13)** parallel với Track 1:
  - DELETE: `utils/loss/perceptual_dino.py`, `feature_mpc_crossview.py`, `tv_depth.py`, `utils/feature/dino_wrapper.py`, `utils/crs/r_feature.py`, `utils/regularizer/pseudo_*.py`
  - DELETE scripts: p11s2_*, p11s4_*, p11s5_*, p11s12_*
  - Remove flags + train.py hooks
  - KEEP: Phase 11 Step 1 covisibility (MARGINAL), Phase 7 LWEIGHT (reference), Phase 10A code (separate decision)
- **Risk**: Sau 6 rejections, probability remaining directions giảm. Anti-aliasing có precedent strong (Mip-Splatting +0.5 dB) → worth attempting.

---

### [2026-05-12] Phase 12 CRS-pull REJECTED → Pivot frequency-axis EFA-GS LFCF
- **Phase 12 CRS-pull Round 1 (seed 42, N=8 paired)**: 3/3 configs REJECT
  - A1 (full pull) Δ=−0.027, 95% CI [−0.155, +0.100], NOT SIG
  - A2 (replace Phase 4 prune) Δ=−0.033, NOT SIG
  - A3 (pull-only) Δ=−0.096, NOT SIG (gần SIG-NEG)
- **Hypothesis verdict**:
  - Scale+Opacity contribution = Δ_A1 − Δ_A3 = +0.068 → softening cần thiết
  - Replace prune = full → Phase 4 dispensable trong A2 context
- **Per-scene pattern**: CRS-pull hại thin/complex (horns −0.241, fortress −0.246, flower −0.180), giúp planar (room +0.221, fern +0.185). Pull mechanism BLUR thin structures vì K-NN target không define được "surface" trên geometry mảnh.
- **Phase 11 + 12 combined verdict**: 9/9 attempts trên CRS axis (loss-axis 5 + position-axis 3 + R_feature deferred 1) → axis EXHAUSTED.
- **Quyết định**: Skip Phase 12 multi-seed verify (pattern clear, tiết kiệm 2-3.5h). Pivot direction.
- **New direction: EFA-GS LFCF port (Phase 13)**
  - Densify-axis CHƯA THỬ trong CRSGaussian → orthogonal Phase 11/12
  - TaT regime evidence +0.17~+0.22 (similar low-PSNR forward-facing)
  - Diffscale volume-preserving isotropify → direct fix Phase 12 thin-structure failure
  - Mip-Splatting LOẠI: custom CUDA conflict + TaT regression −0.94
  - Design doc: `docs/13_efa_gs_lfcf_design.md`
- **Probability honest**: 45-50% commit-worthy (last CRS-axis attempt). Reject branch: pivot writeup "comprehensive 10-mechanism ablation methodology".

---

### [2026-05-12] Phase 13 LFCF + AbsGS Round 1 — A3 WINNER (first commit-worthy CRS-axis in 10 attempts) 🎯
- **Round 1 config**: seed 42 × 8 scenes × 5 configs (A0/A1/A2/A3/A4) = 40 runs ~4.5h
- **8-scene avg PSNR + paired Δ vs A0 (N=8)**:
  | Config | PSNR | Δ vs A0 | 95% CI | Verdict |
  |--------|------|---------|--------|---------|
  | A0 (Phase 8 FULL baseline) | 21.172 | — | — | (reference) |
  | A1 (LFCF alone) | 21.187 | +0.015 | [−0.133, +0.163] | NOT SIG |
  | A2 (LFCF no diffscale) | 21.152 | −0.020 | [−0.149, +0.110] | NOT SIG |
  | **A3 (LFCF + AbsGS)** | **21.331** | **+0.159** | **[+0.009, +0.309]** | **🎯 SIG WINNER** |
  | A4 (AbsGS alone) | 21.203 | +0.031 | [−0.107, +0.170] | NOT SIG |
- **Attribution (5 metrics)**:
  - LFCF full effect: +0.015 (neutral alone)
  - Diffscale contribution (Δ_A1 − Δ_A2): +0.035 (neutral)
  - AbsGS bonus on LFCF (Δ_A3 − Δ_A1): **+0.144 BIG**
  - AbsGS standalone: +0.031 (neutral alone)
  - **LFCF × AbsGS synergy** (Δ_A3 − (Δ_A1 + Δ_A4)): **+0.112 POSITIVE**
  - → Combo gấp 3.5× linear sum (0.046 → 0.159). Synergy REAL, không phải additive noise.
- **Per-scene complementarity**:
  - A3 wins 6/8: fern +0.110, flower +0.098, fortress +0.078, horns **+0.655 ⭐**, leaves +0.181, orchids +0.198
  - A3 loses 2/8: room −0.009, trex −0.038 (simple/planar scenes)
  - A4 wins 5/8 complementary (fern, leaves, orchids, room, trex) — A4 BIG hurt fortress −0.292 + horns −0.232 (LFCF rescues)
- **Significance**: First time trong 10/10 attempts (Phase 11 6/6 + Phase 12 3/3 + Phase 13 A0/A1/A2/A4 4/4) có 95% CI N=8 KHÔNG cross 0. CRS axis bound 21.16 broken (21.16 → 21.33 single-seed).
- **Risk factors**:
  - Single-seed Round 1 — variance band rộng, 95% CI [+0.009] gần 0
  - horns +0.655 contribute 41% của 8-avg gain → seed-42 lucky sample risk
  - A4 alone neutral nhưng A3 = A1 + A4 + synergy → cần verify synergy stable across seeds
- **Decision per design doc Section 10.1**: Δ_A3 = +0.159 ∈ WINNER band (+0.10..+0.20) → **Round 2 multi-seed verify**
- **Round 2 plan (user running)**:
  - Seeds 137 + 9999 × 3 configs (A0, A3, **PLUS A4** for complementarity) × 8 scenes = 48 runs ~3h
  - PLUS A4 = deviation từ design doc, capture complementarity data cho paper
  - Pooled N=24 paired Δ_A3 ≥ +0.10 → COMMIT-WORTHY → Phase 13.1 tolerance sweep
- **Implication for paper**: Nếu Round 2 confirm — Phase 8 FULL recipe upgrade thành Phase 13 FULL = D_cycle + R_visible + S_stability + CRS-mod SH freeze + LFCF + AbsGS. Frequency-axis contribution defendable.

---

### [2026-05-13] 🎯🎯 Phase 13 N=24 FINAL — COMMIT WORTHY (first CRS-axis breakthrough)
- **Round 2 complete**: seeds 137 + 9999 × {A0, A3, A4} × 8 scenes = 48 runs DONE
- **Pooled N=24 (3 seeds × 8 scenes paired):**
  | Config | PSNR | Δ vs A0 | SEM | 95% CI | Verdict |
  |--------|------|---------|-----|--------|---------|
  | A0 baseline | 21.166 | — | — | — | reference |
  | A1 LFCF alone (N=8) | 21.187 | +0.015 | 0.075 | [−0.133, +0.163] | ❌ NOT SIG |
  | A2 LFCF no diffscale (N=8) | 21.152 | −0.020 | 0.066 | [−0.149, +0.110] | ❌ NOT SIG |
  | **A3 LFCF + AbsGS** | **21.330** | **+0.164** | **0.032** | **[+0.101, +0.227]** | **🎯 SIG WINNER** |
  | A4 AbsGS alone | 21.244 | +0.078 | 0.035 | [+0.009, +0.147] | 🎯 SIG MARGINAL |
- **Per-seed consistency A3** (all ≥ +0.10 — robust signal, không seed-artifact):
  - seed 42: +0.159
  - seed 137: +0.195
  - seed 9999: +0.137
- **Per-scene final pattern (N=24)**:
  - horns: +0.362 ⭐⭐⭐ (thin antlers — biggest gain, design doc tiên đoán đúng)
  - orchids: +0.224 ⭐⭐ (thin stems)
  - trex: +0.196 ⭐⭐ (thin bone)
  - flower: +0.150, fern: +0.144, leaves: +0.123, fortress: +0.101 ⭐
  - room: +0.010 (NEUTRAL — Round 1 −0.084 was noise, multi-seed cancel)
  - → **7/8 wins meaningfully + 1 neutral. ZERO hại scenes** (Round 1 fear over room/trex resolved by N=24)
- **Attribution N=24**:
  - LFCF alone: +0.015 (not sig)
  - AbsGS alone: +0.078 SIG marginal
  - Diffscale: +0.035 (neutral N=8)
  - **AbsGS bonus on LFCF**: +0.149
  - **LFCF × AbsGS synergy**: +0.071 POSITIVE (74% over linear 0.093)
- **REVERSAL pattern** (paper main narrative): Phase 12 CRS-pull worst failure modes ↔ Phase 13 best wins symmetric:
  - horns: Phase12 −0.241 → Phase13 +0.362 (reversal 0.603)
  - fortress: Phase12 −0.246 → Phase13 +0.101 (reversal 0.347)
  - flower: Phase12 −0.180 → Phase13 +0.150 (reversal 0.330)
  - → Symmetric mechanism reversal: CRS-pull pull centroid HẠI thin geometry; LFCF diffscale isotropify PROTECT thin geometry.
- **4 decision criteria ALL PASS**:
  1. Multi-seed N=24 paired Δ ≥ +0.10 ✓ (+0.164)
  2. 95% CI excludes 0 strict ✓ ([+0.101, +0.227])
  3. Per-scene robustness ≥6/8 wins ✓ (7/8 + 1 neutral)
  4. Per-seed consistency all ≥ +0.10 ✓ (3 seeds: +0.159, +0.195, +0.137)
- **Comparison vs literature**:
  - Phase 8 paper 1-sample = 21.335 (lucky single-run)
  - **Phase 13 A3 N=24 = 21.330 ⭐ matches paper single-run BUT multi-seed reproducible**
  - DOC-GS 21.38 (gap −0.05 closing)
  - BinocularGS 21.44 (gap −0.11 closing)
  - ICO-GS SOTA 22.20 (gap −0.87 still open)
- **Quyết định**:
  - **COMMIT Phase 13 A3 as new FULL recipe** = Phase 8 FULL components + LFCF (scaler=1.5, interval=2, diffscale=ON, tolerance=1e-5) + AbsGS (uncomment)
  - Lock `use_lfcf=True` + `--absdensify` default trong production config
  - KEEP `--absdensify` infrastructure forever — A4 SIG marginal alone (Scenario 2 fallback nếu LFCF revert)
- **Next steps**:
  1. Optional Phase 13.1 tolerance sweep (~3.5h) — appendix sensitivity (per design doc Section 10.4)
  2. Optional Direction A λ scaler_max sweep ({1.3, 1.5, 1.8, 2.0} × 2 scenes, ~1-2h) — robustness check
  3. Continue PSNR optimization via Phase 13.2+ (sequential mech testing)

---

### [2026-05-13] Phase 13.1 LFCF intensity sweep — A3 base CONFIRMED optimal
- **Sweep design**: 3 variants × 8 scenes × seed 42 = 24 runs ~2h
  - M = scaler=1.8, interval=2 (medium stronger enlarge)
  - H = scaler=2.0, interval=1 (high)
  - X = scaler=2.5, interval=1 (extra aggressive)
  - Base = A3 reference (scaler=1.5, interval=2), reuse Round 1 data
- **Results (paired Δ vs A3 baseline, N=8 seed 42)**:
  | Variant | Test PSNR | Train PSNR | Gap | Δ_test vs A3 | Verdict |
  |---------|-----------|------------|-----|--------------|---------|
  | Base (A3) | 21.331 | 34.235 | 12.905 | 0 (ref) | optimal |
  | M | 21.318 | 34.123 | 12.805 | −0.013 | ⚪ neutral |
  | H | 19.637 | 23.279 | 3.642 | **−1.694** | 📉 catastrophic |
  | X | 19.627 | 23.282 | 3.655 | **−1.703** | 📉 catastrophic |
- **3 patterns observed**:
  1. **scaler dimension INSENSITIVE** (1.5 → 1.8 neutral, 2.0 ≈ 2.5 saturated)
  2. **interval dimension CRITICAL** (interval=2 OK, interval=1 catastrophic)
  3. **H/X = model collapse**, NOT regularization win (train −11 dB >> test −1.7 dB)
- **Root cause hypothesis (interval=1)**:
  - `densify_interval=100`, with `interval_times=1` → LFCF mỗi 100 iter = mọi densify iter
  - LFCF replaces standard `densify_and_clone` → 0 standard clones, 45 LFCF iters
  - LFCF enlarges + occasional probabilistic split, NO pure clone mechanism
  - Result: model undergrowth (insufficient Gaussian spawn) → train collapse
- **Per-scene uniform regression**: all 8 scenes regress với H/X, even thin structures (horns Δ_A3=+0.66 → Δ_H=−1.86)
- **Robustness insight**:
  - A3 (scaler=1.5, interval=2) là **stable operating point** — không phải lucky local optimum
  - Scaler dimension có ±0.3 tolerance (1.5 → 1.8 neutral)
  - Interval dimension brittle, đừng push xuống 1
- **Quyết định**: A3 base optimal trên LFCF intensity axis. Lock recipe, no further intensity sweep.
- **Anchor cho Phase 13.2+**: A3 21.330 confirmed ceiling. Beyond cần orthogonal mechanism (loss-side, optimizer-side, novel-view freq).

---

### [2026-05-13] Phase 13.2 strategy — Sequential mech testing (replace Path 1 stack)
- **Context**: Sau Phase 13.1 sweep confirm A3 optimal, evaluate next mechanism class candidates.
- **Initial proposal (Path 1 stack)**: 5 Tier 1 mechanisms combined (Gap C FALA + visibility prune + pose perturb + opacity decay tweaks + CRS prune tweaks) — 5 ngày, claimed P(+0.2) = 40-50%.
- **Reality check**:
  - Project track record: 15 mechs tested, 1 win → base rate ~7%
  - Phase 11 Stack S1+S2 historical synergy −0.063 (negative)
  - User Path 1 estimate too optimistic — honest P(+0.2) re-estimate 15-25%
  - Drop T1.4 (opacity tweak) + T1.5 (CRS prune tweak) = noise band
- **Strategy DECISION: Sequential focused mech testing** (replace stack):
  1. **Step 1: DWTGS port** (Gap A) — paper claim +0.3-0.4 PSNR sparse-view standalone (HIGHEST evidence)
  2. **Step 2: Gap C FALA** — loss-side frequency annealing (preceded by spectrum analysis)
  3. **Step 3: Visibility prune** — geometric anti-overfit different axis
  4. **Step 4 (only if Steps 1-3 insufficient)**: Tier 3 architecture change (Hierarchical Gaussians like BinocularGS)
- **Per-step stop conditions**:
  - Δ ≥ +0.10 single-seed → multi-seed verify, commit if confirmed
  - Δ +0.05~+0.10 → multi-seed verify
  - Δ < +0.05 → drop, move to next step
- **Excluded directions (evidence-based)**:
  - ❌ T1.4 + T1.5 (hyperparam tweaks, base rate <10%)
  - ❌ T2.3 Cross-view MPC (Phase 11 Step 4 already failed Δ=−0.042)
  - ❌ T3.1 Mip-Splatting (TaT regression −0.94 + custom CUDA conflict, Phase 13 Section 6 rejected)
  - ❌ T3.4 Foundation model priors (Phase 10A DUSt3R failed catastrophically)
- **User chose Gap C path first** (cheaper, faster signal). DWTGS deferred unless Gap C works.

---

### [2026-05-14 evening] Phase 13.2.1 HF-emphasis pilot — REJECTED, all loss-side freq axis exhausted
- **Run setup**: 24 runs = 4 λ (calibrated 0.019/0.063/0.189/0.631) × 2 timing (T1000/T2000) × 3 scenes (trex/horns/orchids) × seed 42
- **A3 baseline reference**: trex 23.392, horns 20.615, orchids 16.982
- **Results — Pattern INVERSE prediction**:
  | Scene | Mean Δ vs A3 (8 configs) | Pattern |
  |---|---|---|
  | trex (low deficit, low A3 gain) | +0.18 | ⭐ Win 5/8 configs |
  | orchids (high deficit, mid A3 gain) | +0.02 | ⚪ Null (Parseval/data limit) |
  | **horns (mid deficit, high A3 gain)** | **−0.31** | ❌ **8/8 NEGATIVE** |
- **Statistical significance**: horns 8/8 negative → P(random) = 0.5⁸ = 0.4%. NOT atomicAdd noise.
- **Best combo** λ=0.631 T2000 Δ_mean=+0.036: cancellation effect (trex +0.357 cancel horns −0.354), KHÔNG real win
- **Phase conflict mechanism CONFIRMED**:
  - Scene A3 spatial gain LOW (trex +0.196) → HF emphasis helps (+0.18)
  - Scene A3 spatial gain HIGH (horns +0.362) → HF emphasis catastrophic (−0.31)
  - → A3 spatial alignment **FRAGILE** to amplitude pressure
  - → Updated stacking principle: spatial + spectral axes ORTHOGONAL in concept, INTERFERE in practice
- **Memory `a3-mechanism-spatial-not-spectral` REFUTED on "stackable" claim** — updated với pilot evidence
- **Orchids null** confirms `3dgs-systematic-hf-deficit` warning: spectrum close ≠ PSNR gain (data limit 3-view sparse)
- **Inverse correlation finding (NEW)**: PSNR headroom ≠ improvement potential. Higher A3 spatial gain → harder to stack.
- **Multi-seed verify SKIPPED**: pattern clear (p=0.004), save 24 GPU-hours
- **Generalization REFUSED** for all loss-side frequency mechanisms (same class, same expected conflict):
  - ❌ FALA-reversed (sharpen GT) — same amplitude push
  - ❌ Sobel/Laplacian variants — same edge-based supervision
  - ❌ FFT-domain HF L1 — same axis (frequency amplitude)
  - ❌ DWTGS HF-sparsity — already rejected via diagnostic (wrong sign)
- **Verdict**: All loss-side frequency-axis mechanisms EXHAUSTED trên Phase 13 A3 backbone
- **Cleanup pending** (Rule 13): delete utils/loss/hf_emphasis.py + scripts/p13_2_hf_pilot.{sh,_analyze.py} + lambda_calibration.py + revert train.py gate + remove 3 args flags. KEEP logs/p13_2_hf/ (negative result record) + Section 20 design doc.
- **Pivot direction**: T13.2.3 Visibility-based prune (geometry axis, different mechanism class, ~0.5 ngày)
- **Fallback if T13.2.3 fails**: Tier 3 architecture (Hierarchical Gaussians, 2-3 tuần) hoặc accept 21.330 ceiling

---

### [2026-05-14] Phase 13.2.1 λ_HF Calibration — Evidence-based hyperparameter
- **Tool**: `scripts/p13_2_lambda_calibration.py` measure L_main + L_HF magnitudes trên A3 baseline renders
- **Method**: Compute Laplacian L1 magnitude per scene → ratio R = L_main / L_HF → derive λ at % L_main contribution targets
- **Aggregate measurements (8 scenes, A3 seed 42)**:
  - L_main = 0.0550 (= L1+SSIM photometric)
  - L_HF = 0.0873 (= mean |∇²I_render − ∇²I_GT| Laplacian L1)
  - R = 0.631
- **Calibrated λ_HF levels**: {safety 0.019 (3%), gentle 0.063 (10%), moderate 0.189 (30%), strong 0.631 (100%)} × R
- **Replace earlier guess** {0.05, 0.10, 0.20} which underestimate 1.3-3.2× per level
- **Per-scene L_HF ranking** consistent với diagnostic findings (leaves/orchids top, room bottom)
- **Defendable justification**: "λ_HF calibrated từ measured Laplacian magnitude trên A3 baseline (8 scenes × seed 42), ratio R=0.631. Pilot sweep at {3%, 10%, 30%, 100%} × R covers safety threshold to weight parity với main photometric loss."
- **Parseval ceiling estimate**: theoretical max PSNR gain ~+0.10-0.20 dB (HF carry ~12% total energy in natural images, 1/f spectrum)

---

### [2026-05-13 evening] Phase 13.2 Render-vs-GT diagnostic — VERDICT overturns initial Gap C plan
- **Script**: `scripts/p13_2_spectrum_diagnostic.py` (~400 lines). Render A0+A3 × 8 scenes ~16 min + diagnostic ~2 min.
- **Methodology**: per (config, scene) compute rel_Δ(k) = log10(P_render(k) / P_gt(k)) trên radial bins, band-mean LF/MF/HF.
- **3 critical findings**:
  1. **Universal HF deficit (8/8 scenes)**: A3 HF mean = −0.192 → render đạt chỉ **64% HF energy GT**. Range: trex −0.119 (76%) → orchids −0.255 (56%). **KHÔNG scene nào produce thừa HF**.
  2. **A3 mechanism = SPATIAL, không phải spectral**: 6/8 scenes A3 produce HF ÍT HƠN A0 (imp_HF negative). horns paradox: Δ_PSNR=+0.362 (highest) nhưng imp_HF=−0.014. → A3 PSNR gain qua spatial alignment (Gaussian đúng chỗ), KHÔNG qua HF amplitude.
  3. **Standard literature wrong sign**:
     - DWTGS HF-sparsity assume model OVER-produces HF → **REFUTED** (8/8 under-produce)
     - Standard FALA blur GT → **WORSE** (model đã quá mượt, blur GT làm thiếu HF thêm)
- **A3 pattern tally (seed 42, 8 scenes)**: 4/8 MISSING_HF (flower/horns/leaves/orchids) + 4/8 WEAK_SIGNAL (fern/fortress/room/trex), 0/8 SPURIOUS_HF, 0/8 NEAR_CEILING.
- **Mechanism direction OVERTURN**:
  - ❌ REJECT DWTGS port (wrong sign)
  - ❌ REJECT standard FALA blur GT (wrong sign)
  - ✅ PROPOSE HF-emphasis L1 loss (high-pass(GT) + L1) — correct sign, untested
  - ✅ PROPOSE FALA-reversed (sharpen GT instead of blur) — alternative correct sign
- **Risk acknowledged**:
  - Spectrum close không guarantee PSNR up (fern: A3≈A0 spectrum, +0.144 PSNR)
  - 3-view sparse fundamental limit (8/8 deficit có thể data limit, không phải mechanism)
  - Single-seed 42, pattern stability với seeds 137/9999 chưa verify
- **Pilot plan**: orchids × seed 42 × λ_HF ∈ {0.05, 0.10, 0.20} = 3 runs ~21 min. Target orchids vì HF deficit largest (−0.255) + PSNR lowest (16.97) + Δ_A3 high (+0.224).
- **Decision tree pilot**:
  - Δ_PSNR > +0.10 AND HF_rel_Δ → 0 → scale up multi-seed
  - Δ_PSNR neutral, HF closer → spectrum closed nhưng không help PSNR → abandon HF axis
  - Δ_PSNR < 0 → wrong direction → try FALA-reversed instead
- **Implication paper narrative**: A3 = "spatial alignment via frequency-aware densify gating" (specific), KHÔNG "frequency-aware learner" (vague). Spectrum-amplitude axis ORTHOGONAL with A3 spatial axis → có thể stack.

---

### [2026-05-13] Phase 13.2 pre-Gap C — FFT spectrum analysis on LLFF GT
- **Motivation**: σ_blur hyperparameter cho Gap C (Gaussian blur low-pass) chọn arbitrary rule-of-thumb (σ=3, ks=19) = không evidence-based. Risk choose σ outside meaningful LF/HF separation zone.
- **Approach**: Analyze actual frequency distribution của LLFF GT images trước khi implement Gap C.
- **Methodology**:
  - 2D FFT trên train view GT images (8 scenes × 3 train views = 24 images)
  - Radially-averaged 1D spectrum per scene
  - Cumulative energy fraction at thresholds (50/80/90/95/99%)
  - Convert cutoff frequency → σ_blur via formula `σ = 1/(2π·f_cut)`
  - Output recommendations: σ_conservative (80% energy) / σ_balanced (90%) / σ_aggressive (95%)
- **Implementation**: `scripts/p13_2_spectrum_analysis.py` (CPU-only, ~200 lines, no GPU)
- **Expected output**: 
  - 8 spectrum plots (one per scene)
  - SUMMARY.txt with per-scene + aggregate energy distribution
  - Concrete σ_ablation range recommend cho Gap C
- **Decision flow**:
  - Spectrum confirms σ=3 hợp lý → ablate σ ∈ [2, 3, 4] trong Gap C
  - Spectrum reveals σ different → shift Gap C ablation range
  - Spectrum reveals 1/f decay (no clear cutoff) → σ as design choice, document rationale
- **Cost**: ~30 phút server-side compute + analyze, gates Gap C implementation.
- **Defensive value**: σ choice có justification từ data, không phải arbitrary tune.

---

### [2026-05-14] Phase 13.2.1 HF-emphasis pilot — ❌ REJECTED
- **Setup**: Laplacian 3×3 high-pass L1 loss, λ calibrated {0.019, 0.063, 0.189, 0.631} (=R×{3,10,30,100}%), 2 timings {1000,2000}, 3 scenes {trex,horns,orchids}, seed 42 = 24 runs.
- **Result — INVERSE pattern (đúng failure mode spec cảnh báo)**:
  - trex (HF deficit ÍT −0.119): Δ_mean ≈ **+0.18** ⭐
  - orchids (HF deficit NHIỀU −0.255): Δ_mean ≈ **+0.02** ⚪ (headroom lớn nhất nhưng KHÔNG gain)
  - horns (mid, A3 win lớn nhất +0.362): Δ_mean ≈ **−0.31** ❌ (8/8 cells negative)
  - Best combo +0.036 = **cancellation** giữa trex/horns, KHÔNG phải win thật
- **Verdict**: Dose-response NGƯỢC prediction. Phase-amplitude conflict CONFIRMED — HF loss đẩy Gaussian khỏi vị trí A3 spatial alignment, phá scene A3 win mạnh nhất. Multi-seed SKIPPED (pattern quá rõ).
- **Hệ quả mở rộng**: cả CLASS loss-side frequency-axis chết — Laplacian fail + DWTGS wrong-sign + FALA-blur wrong-sign + FALA-reversed/Sobel cùng class → predict same failure. **Frequency-axis loss-domain EXHAUSTED.**

---

### [2026-05-14] Phase 13.2.3 Covisibility-weighted supervision pre-flight — ❌ REJECTED (no substrate)
- **Setup**: post-hoc pre-flight (no train) trên A3 render đã có. Test A (noise direction: var(err|mono) vs var(err|multi)) + Test B (Shannon H(covis) + Pearson(covis,freq)). 2 fix bắt buộc: sort-by-name (khớp dataset_readers.py:353), validity-mask (loại pixel no-COLMAP-support). Pairing render↔COLMAP join BY NAME (render.py:48 lưu image_name+'.png'), verified unmatched=0 mọi scene.
- **Killer finding**: `multi=0` ở **4/8 scene** (horns/orchids/room/trex) — KHÔNG pixel nào ≥2 train view cùng thấy 1 điểm 3D. covis range toàn [0.00,0.00] hoặc [0.00,1.00].
- **Nguyên nhân**: 3 train view LLFF chọn cách xa nhau (linspace[0,mid,last] của pool) → baseline rộng → COLMAP track cho điểm test-visible hầu như chỉ 0-1 train view. **Term (1-covis) ≈ hằng số → không có gì để weight.** Đảo dấu cũng vô nghĩa.
- **Test B "PASS 4/4" là FALSE PASS**: H(covis)=2.2bit cao do artifact Gaussian-splat interpolation (σ=15), KHÔNG do covis đa dạng thật. Caveat C3-style — metric đo nhầm artifact.
- **Verdict**: REJECT covisibility-weighted hoàn toàn. Structural, không phải tuning. Pre-flight 5-min CPU đủ phát hiện trước khi code 1-2 ngày.

---

### [2026-05-14] SYNTHESIS — Bệnh gốc: cross-view consistency DEAD trong 3-view wide-baseline
- **Quyết định**: Loại trước cả MỘT LỚP ý tưởng — mọi mechanism dựa trên inter-view geometric consistency giữa 3 sparse train view đều structurally dead. KHÔNG re-propose.
- **Bằng chứng hợp nhất 5 thất bại độc lập cùng 1 nguyên nhân**:
  | Hướng | Chết vì |
  |---|---|
  | Pseudo-view (Phase 5b) | warp GT qua wide-baseline parallax → rác ở vùng gap |
  | DUSt3R dense init (Phase 10A) | alignment residual error vì baseline rộng |
  | Cross-view MPC (Phase 11) | cross-view feature không nhất quán |
  | CRS-pull position (Phase 12) | position constraint cross-view sai |
  | Covisibility-weighted (13.2.3) | covis degenerate, multi=0 4/8 scene |
- **Nguyên lý**: 3 wide-baseline view KHÔNG cung cấp inter-view geometric consistency dùng được. Chỉ mechanism dùng **external-prior injection** HOẶC **within-view** HOẶC **architecture** mới sống. Đây là lý do Phase 8 A3 thắng (DAV2 monocular depth = external prior per-view; SH-freeze + LFCF = within-view; KHÔNG cross-view).
- **Loại trước**: SOTA Binocular3DGS / NexusGS / SCGaussian (cross-view stereo/flow/match) — predict fail, không port.

---

### [2026-05-14] REFUTED — "HF deficit = bottleneck / cần đổi architecture"
- **Quyết định**: Bác bỏ framing trước đó ("Gaussian primitive lowpass → HF deficit là bottleneck → chỉ Tier 3 architecture phá được"). SAI.
- **Bằng chứng phản chứng (chính evidence của ta)**:
  1. HF-emphasis pilot: đóng HF gap → PSNR phẳng (orchids Δ≈0) → HF **decoupled khỏi PSNR**
  2. **A3 train PSNR = 34.21** → Gaussian primitive THỪA SỨC tái tạo HF khi có view. Nếu primitive lowpass thì train không đạt 34. → HF deficit ở test KHÔNG phải giới hạn vật lý primitive.
- **Reframe**: HF deficit ở test = **triệu chứng của overfit/generalization**, không phải bệnh. Tín hiệu dominant đo được = **overfit gap 12.88 dB** (train 34.21 / test 21.33).
- **Hệ quả priority**: Tier 3 architecture (đổi primitive) = bet đắt SAI hướng (train=34 chứng minh primitive đủ). Co-Adapt anti-overfit bị mình mis-rank "adjacent" — thực ra nhắm đúng dominant signal.
- **Caveat chưa giải**: 12.88 trộn reducible-overfit + irreducible-3view-limit, CHƯA tách. Cần diagnostic decompose trước khi chọn direction.

---

### [2026-05-14] Next step + constraints
- **NEXT**: `scripts/p13_2_bottleneck_decompose.py` — 1 script post-hoc (no train) attribute test error per pixel vào H1 irreducible(covis=0) / H3 geometry(depth-disagree) / H5 detail(HF) / H4 appearance(SH). 8 hypotheses H1-H8 (xem bảng). Quyết direction theo % attribution, không đoán.
- **Constraint LOCKED**: **10k iter cố định** — KHÔNG đề xuất iter-budget/schedule-change. Lý do: clean paper comparison đã chốt. (T12.1 iter-15k trong task_queue cũ → VOID.)
- **SOTA survivor sau cross-view filter**: Co-Adaptation-of-3DGS (within-view dropout+opacity-noise, report +0.68 trên BinocularGS — nhưng A3 đã có DropAnSH, cần pre-flight overlap), dn-splatter (monocular **normal** prior — external-prior axis CHƯA đụng, Phase 8 chỉ dùng DAV2 depth).
- **Cleanup pending** (Quy tắc 13): HF-emphasis reject → xóa utils/loss/hf_emphasis.py + scripts/p13_2_hf_* + lambda_calibration.py + revert train.py hook + remove 3 args. Giữ logs SUMMARY + design-doc evidence. Chờ user approve.

---

### [2026-05-17] Phase 13.2.4 Bottleneck decompose — DONE (verified 8-scene)
- **Script** `scripts/p13_2_bottleneck_decompose.py` — post-hoc no-train, attribute test error/pixel → H1 irreducible / H3 geometry / H5 detail / H4 appearance. 5 bug fixed qua verify-from-code (KHÔNG đoán): GaussianModel(args) API, points3D id-preserving reader, COLMAP-vs-render resolution scale (dynamic W/Wc), depth /alpha (crs_module:705 convention), robust-median align (polyfit outlier-fooled).
- **Verdict 8-scene**: align ALIGNED/SCALE_CORRECTED 8/8 (scale 0.99-1.03, corr_in 0.95-0.998). **H3≈0.5% (geometry SOLVED), H1=21% irreducible (leaves 92.6%!), H4=63% appearance, H5≈16% (τ_hf leak), H8≈0 (exposure REFUTED), h4_ratio 0.18-0.36 (chroma/specular REFUTED — kể cả room "SH anomaly" cũ → bác bằng đo)**. Lỗi dominant = 3-view appearance ambiguity (right geometry+color, radiance per-pixel underdetermined), KHÔNG phải mechanism sửa được post-hoc.
- **⚠️ OVER-CLAIM CORRECTIONS (ghi để session sau KHÔNG lặp)**:
  1. "Accept ceiling / no fixable gap" mình kết luận nhiều lần = **OVER-CLAIM**. Bottleneck post-hoc **MÙ với training-dynamics** (chỉ soi model đã hội tụ). User push back đúng.
  2. **AbsGS (+0.164, densification-axis, cùng A3 backbone) = bằng chứng tồn tại** rằng đổi tiêu chí densify CHO gain thật ở regime ta → densification-axis KHÔNG bị bottleneck loại.
  3. Co-Adapt/DropAnSH: user nhắc — Co-Adaptation dropout family **ĐÃ test** (Track A/B + Phase 1: D1 pure DropAnSH 21.12 thắng family; D3 stack redundant **−0.48 over-regularize**). Anti-co-adaptation axis empirically exhausted.
  4. **AbsGS > LFCF**: LFCF alone Δ+0.015 (≈noise), AbsGS alone +0.078, synergy +0.071. "Phase 13 frequency win" thực chất do AbsGS (gradient-cancellation catch HF-underfit), KHÔNG do LFCF explicit.

### [2026-05-17] GDAGS (ICLR 2026, workspace) — verified mechanism + Gate-2 PASS
- **Verified GDAGS/scene/gaussian_model.py:526-527**: `consistency = grads/grads_abs`, `weight = 0.8+25·(1−c)^15`, clone dùng `grads/weight`, split `grads*weight`. **GCR = grads/grads_abs = TỈ SỐ của đúng 2 signal AbsGS đã có → KHÔNG orthogonal.** GDAGS = **policy A/B swap trên trục AbsGS proven**, KHÔNG phải +feature orthogonal. Sửa nhận định mình lượt trước (đoán từ paper-summary, code bác bỏ).
- **Quan hệ với A3**: LFCF path (`is_lfcf_iter` branch) tách biệt — GDAGS chỉ thay AbsGS-OR-rule trong standard-path. KHÔNG phải catastrophe-conflict như lo ban đầu.
- **Gate-2 pre-flight DONE** (`scripts/p13_2_gdags_gate.py`, standalone no-production-touch, confidence=1 neutral → ratio bất biến). Full 8-scene: **✅ TRACTION — 8/8 non-degenerate** (agg collapsed 19.5% « 85%, w95 12.4 « 100, heavy 9.4% « 40%; room nhẹ nhất 33% collapsed, fortress conflicted nhất — vẫn dải healthy). GDAGS coherence-weight phân biệt được Gaussian trong regime ta.
- **Caveat KHÔNG biến mất**: PROXY directional (converged L1 eval-render ≠ densify-time grad); GDAGS không orthogonal = policy-A/B không +feature, kỳ vọng modest, có thể ≈/< AbsGS; cross-paper PSNR không comparable. TRACTION = đáng pilot, KHÔNG = chắc gain.
- **Decision**: Gate-2 PASS → implement flag-gated GDAGS + A/B pilot. KHÔNG accept-ceiling (post-hoc-blindness đã nhận; densification-axis viable per AbsGS).

### [2026-05-17] GDAGS implementation contract (chờ user duyệt plan trước code)
- Flag `use_gdags` default **False** trong OptimizationParams → auto-register `--use_gdags` (mirror `--absdensify` :225). Helper `utils/densify/gdags.py` (NEW, Quy tắc 12). `gaussian_model.__init__`: `self.use_gdags=getattr(args,'use_gdags',False)` (mirror `self.absdensify` :156). Gate CHỈ standard-path `else` branch densify_and_prune. densify_and_clone/split: `if self.absdensify and not self.use_gdags`. **LFCF path KHÔNG đụng 1 ký tự. train.py KHÔNG sửa** (flag flow qua args y hệt absdensify). Verify flag-OFF = A3 byte-identical TRƯỚC pilot. Scripts: `p13_2_gdags_pilot.sh` + `_analyze.py` (A/B A3 vs A3+GDAGS, trex/horns/orchids seed42).
- **Literature filter cho user (search song song)**: hard-reject = cross-view / "produce-more-HF" / foundation / loss-only / dropout-variant. Value = training-dynamics + anti-overfit + in-train signal + ORTHOGONAL với {grad,abs-grad,scale,opacity,LFCF,CRS,dropout}. Bẫy: nhiều "new densify" = re-express grad/abs/scale → A3 đã dùng → low value.

---

### [2026-05-18] Phase 13.2.5 GDAGS A/B pilot full-8 — ❌ REJECTED (LOCK A3 21.330)
- **Implement DONE đúng contract**: `utils/densify/gdags.py` NEW (`compute_gdags_weight` EXACT mirror GDAGS:526-527, no clamp per source); `arguments/__init__.py` `self.use_gdags=False` (auto-register); `scene/gaussian_model.py` 4 chỗ gated (`__init__` getattr; densify_and_split/clone `if self.absdensify and not self.use_gdags`; densify_and_prune standard `else`: GDAGS→`clone_g=grads/w, split_g=grads*w`). **LFCF path 0 ký tự đụng. train.py 0 sửa** (grep-confirmed flag flow qua args y hệt absdensify).
- **B1 VERIFY flag-OFF = A3 byte-identical PASS**: tiêu chí PRIMARY = **N_gaussians** (KHÔNG PSNR — single-scene PSNR ±1.3 dB atomicAdd noise, project_3dgs_variance_floor; threshold 0.05 cũ SAI mâu thuẫn variance floor). VERIFYOFF (code mới, KHÔNG --use_gdags) vs logs/p13_lfcf/A3_seed42: N reldiff < 5% mọi scene → contract HOLDS, code GDAGS KHÔNG phá A3 khi flag OFF. (ΔPSNR ≤±1.3 = atomicAdd noise, KHÔNG regression — N là bằng chứng.)
- **A/B pilot full-8 seed42 (paired vs logs/p13_lfcf/A3_seed42_*, KHÔNG re-run A3)**:
  | metric | value |
  |---|---|
  | **Δtest_mean** | **+0.0149** (std 0.2753, N=8) — **≈0, trong noise floor ±0.10** |
  | ΔN_mean | **+51.3%** (capacity↑ mạnh) |
  | Δtrain_mean | **+2.081 dB** (fit↑ mạnh) |
  - Per-scene Δtest: fern −0.025, flower −0.082, **fortress +0.268**, **horns −0.605 ❌❌**, leaves +0.168, orchids +0.131, room −0.072, **trex +0.336**.
- **Verdict ❌ OVER-DENSIFY → OVERFIT confirmed**: ΔN=+51% (capacity↑) + Δtrain=+2.08 (fit↑) + Δtest≈0 (test phẳng) = chữ ký memorize kinh điển. Mean +0.015 « ±0.10 floor = noise; "kết quả không tệ" là **bẫy cherry-pick** (mean dương giả do fortress/trex/leaves bù horns/flower). **horns −0.605 = thảm hoạ**: horns là scene A3 thắng LỚN NHẤT Phase 13 (+0.362 N=24) → GDAGS phá đúng điểm mạnh nhất của A3, đúng pattern phase-conflict HF-pilot.
- **3rd INDEPENDENT CONFIRMATION — 3-view capacity ceiling**: thêm capacity ở regime 3-view = memorize, KHÔNG generalize. Hợp nhất 3 thất bại độc lập cùng cơ chế:
  | Hướng | Capacity added | Δtest | Cơ chế |
  |---|---|---|---|
  | HF-emphasis pilot (13.2.1) | HF amplitude pressure | −0.31 (horns 8/8 neg) | phase-amplitude conflict |
  | D3 dropout-stack (Phase 1) | dropout regularizer stack | −0.48 | over-regularize redundant |
  | **GDAGS (13.2.5)** | **+51% Gaussians** | **≈0, horns −0.605** | **over-densify → memorize** |
  → KHÔNG re-propose densification-axis / capacity-add mechanism. A3 (LFCF+AbsGS-OR) = stable operating point trên trục densify; policy-swap GDAGS KHÔNG gain (đúng dự đoán: GCR không orthogonal AbsGS).
- **Cost**: GDAGS ~1.5× Gaussian = ~1.5× memory/chậm (feedback_measure_compute_cost) — cost thật kể cả khi PSNR ngang.
- **Decision**: REJECT GDAGS. **LOCK A3 = test 21.330** (Phase 13 FULL recipe committed). Densification-axis EXHAUSTED. User accept verdict ("ok clear và reject", 2026-05-18).
- **Cleanup pending (Quy tắc 13)**: revert `arguments/__init__.py` (use_gdags), `scene/gaussian_model.py` (4 chỗ gated), delete `utils/densify/gdags.py` + `scripts/p13_2_gdags_*`; xóa `output/p13_2_gdags/`. GIỮ `logs/p13_2_gdags/*.log` (negative-result evidence) + design-doc Section 20. Gộp chung HF-emphasis cleanup pending. Chờ user approve list.

---

### [2026-05-18] GDAGS cleanup DONE + 2 literature candidates SCREENED ❌ (pre-implementation, verify-from-source)
- **Cleanup A+B+D executed** (user approved): deleted local `scripts/p13_2_gdags_{gate,pilot,pilot_analyze}` + `utils/densify/gdags.py` + adjacent reject `scripts/p13_2_{lambda_calibration,weighted_preflight,spectrum_analysis,spectrum_diagnostic}`; reverted `arguments/__init__.py` (use_gdags block) + `scene/gaussian_model.py` (4 chỗ → A3 baseline AbsGS-OR). grep-verify 0 residue. train.py chưa từng đụng. **A3 21.330 byte-restored.** (Rule cleanup cập nhật 2026-05-18: lần sau GIỮ script/module local, chỉ xóa server; code-edit vẫn revert cả 2 — xem feedback memory.)
- **User đề xuất 2 candidate, verify-from-source TRƯỚC khi code (kỷ luật GDAGS: paper-summary lừa, code/full-text mới thật)**:
  1. **Opacity-Gradient DC (arXiv 2510.10257)** — densify trigger = per-Gaussian `max|∂L/∂α| > τ` thay positional grad, + conservative prune (start 2000, opacity thr 0.001, hard N_max). ❌ **WRONG OBJECTIVE**: LLFF 3-view PSNR **19.55 < FSGS-baseline 19.88** (−0.33); headline "improve" = LPIPS/SSIM/**compactness 32k<57k**, KHÔNG PSNR. **Ablation tự bác**: "w/o error-driven densif" (positional+conservative-prune) PSNR 20.33 > "full" (opacity-grad trigger) PSNR 20.00 → opacity-grad trigger TỰ NÓ tốn 0.33 PSNR đổi compaction (978k→32k). Baseline FSGS 19.88 «« A3 21.330 (A3 đã compact regime OGDC kéo FSGS tới). Port → nhiều khả năng TỤT 21.330 (thay AbsGS +0.164 bằng signal anti-PSNR). KHÔNG phải capacity-add (escape 3-view ceiling) nhưng objective sai. *Lý do = evidence-specific, KHÔNG "densify exhausted" dogma.*
  2. **Improved-GS (arXiv 2508.12313)** — Edge-Aware Score `S=Σω_edge·α_render`, Long-Axis Split, Recovery-Aware Pruning (post-opacity-RESET recovery, prune bottom-20% @3300/6300). ❌ **REGIME-MISMATCH + overlap dead-axis**: eval DENSE-view ONLY (Mip360/T&T/DeepBlending 1–3M Gauss, +0.7~0.9 PSNR) — KHÔNG sparse/LLFF. Edge-Aware = densify-on-edges ≈ **HF-emphasis axis ĐÃ REJECT** (HF-pilot −0.31 horns 8/8 neg, phase-conflict). Recovery-Aware cần substrate opacity-RESET — A3 dùng opacity_decay 0.999 + CRS-prune Option C, KHÔNG reset → no-substrate (cùng class covis-reject). Adopt reset phá Phase 8 locked.
- **Giá trị**: 2 candidate bị lọc ở tầng literature/verify-from-source, 0 compute tốn (như GDAGS Gate-2 / covis pre-flight). KHÔNG re-propose 2 paper này.
- **NEXT DECISION**: survivor in-workspace duy nhất = `dn-splatter/` monocular-**NORMAL** prior (Phase 8 chỉ dùng DAV2 depth; normal = external-prior orthogonal, KHÔNG capacity-add, KHÔNG loss-freq, KHÔNG cross-view; repo CÓ trong workspace → verify-from-code được, khác C1/C2 không repo). Vs accept 21.330 (giờ defensible: bottleneck verified=appearance-ambiguity + densify×3/loss 6+/CRS 9/cross-view 5-fail + 2 lit-candidate filtered).

---

### [2026-05-18] Phase 14 — L_consist (Binocular3DGS NeurIPS24) port PLAN LOCKED (chờ Bước-0)
- **User chỉ đạo** research Binocular3DGS (in-workspace) hướng dense-init + L_consist. Verified-from-code đầy đủ.
- **L_consist verified** (Binocular train.py:123-136, scene/__init__.py:96-115, loss_utils.py:68-91): mỗi iter sinh synthetic camera dịch ngang baseline-hẹp từ 1 train view → render 2 cam → `disp=f·B/rendered_depth` → inverse-warp shifted-render về cam gốc → `L1(warped, GT_gốc) + 0.05·Godard-edge-smooth`. **Differentiable loss**, grad vào geometry+appearance. Là **self-supervised single-view stereo**, KHÔNG warp data wide-baseline → **thoát cross-view-structurally-dead**; 0 Gaussian thêm → **thoát 3-view-capacity-ceiling**.
- **Orthogonality vs d_cycle — verified [crs_module.py:558-586, d_cycle.py]**: d_cycle = `@torch.no_grad` SCORE (pixel-reproj-error cross-view-pairwise, nuôi CRS-gating); L_consist = differentiable LOSS (photometric, synthetic-view). **Khác tầng, KHÔNG bẫy GDAGS** (không phải cùng-1-signal). Chung target failure-population (floater/sai-geometry) → rủi ro = **bão hoà hiệu lực**, KHÔNG redundant cơ chế.
- **Synthesis [2026-05-14] "đừng port Binocular3DGS" = over-broad cho self-stereo** — synthesis đó nhắm cross-view wide-baseline (Binocular stereo/flow). L_consist là self-stereo single-view → exception HỢP LỆ, severity Thấp, ghi rõ ở đây.
- **Self-correct over-claim (lần 3)**: lập luận "đừng test D_cycle-off vì Phase-9 đã chốt" của tôi = OVER-CLAIM. decisions_log:933-948 ghi **"D_cycle FLIPS NEGATIVE on strong backbone"** (D1-O999: Δ=−0.015 vs weak-backbone +0.125). D_cycle-vs-D_DAV2 **CHƯA A/B lại multi-seed trên A3 cuối**. Phase-9 chốt drop-R/S/mechanism, KHÔNG chốt D_cycle. → test D_cycle-off là câu hỏi MỞ, không re-test verdict đóng. `use_d_cycle=False` = **clean toggle** (crs_module:565 if/else, fallback D_DAV2; A3 đã load DAV2 → sẵn sàng).
- **Thiết kế: 2×2 factorial** {D_cycle ON/OFF} × {L_consist OFF/ON}. A=A3 (reuse logs/p13_lfcf, KHÔNG re-run) · B=A3+Lc · C=A3−Dcyc(→D_DAV2) · D=A3−Dcyc+Lc. Đọc: A−C=giá trị biên D_cycle/A3; B−A=Lc thêm; D vs B=Lc thay được D_cycle?; B≈A=Lc bão hoà. 3 cell mới (C=0 code, chỉ tắt flag).
- **3 bug-fix port (Phase-0 verify):** (#1) CRS Camera KHÔNG có get_camera_matrix/get_focal → viết shifted-cam bằng **PseudoCamera(cameras.py:66-87)** + `W2C=wvt.T` (~10 dòng, KHÔNG copy verbatim). (#2 **silent bug Cao**) CRS rasterizer depth = **accumulated ΣαT· (verified bottleneck:203-208 + crs_module:697-708)** → hard `disp=f·B/depth` BẮT BUỘC `/(alpha+1e-6)` differentiable; Bino-side irrelevant cho fix (chưa direct-read CUDA → không assert). (#3) `trans_dist=0.4` = world-scale Bino → scale theo tỉ lệ `cameras_extent`(=nerf_norm radius) CRS-vs-Bino, ablate hẹp {0.5×,1×,2×}. (+budget: `start_iter` 20000/30k→~6700/10k, project_budget_scaling_rule).
- **Thứ tự thực thi:**
  - **Bước 0** (server, ĐANG CHẠY, chặn tất cả): re-run A3 (script gốc `p13_lfcf_multiseed.sh` CFG=A3, LOG_DIR=logs/verify_a3_restore để KHÔNG đè baseline) horns+orchids seed42 → so **N_gaussians** vs logs/p13_lfcf. reldiff <5% ✅ A3 byte-restored→baseline reuse hợp lệ; ≥20% ❌ STOP (cleanup phá → verdict GDAGS cũng phải review).
  - **Bước 1** saturation pre-check (no-train ~20ph): residual L_consist thật trên A3 ckpt. **substrate = res(B│valid_mask ∧ ~Lambertian) − res(B→0│cùng region)**. floor_A(B→0)=train-fit floor (warp→identity, triệt tiêu common trong hiệu). ~Lambertian proxy = tiny-rotate render đổi ít (KHÔNG đụng SH coeff). Known small +bias còn lại = bilinear-interp → kết luận "có đất" CHỈ khi vượt floor đủ biên (logic ±0.10); marginal=no-go. Reject sớm nếu no substrate.
  - **Bước 2** Phase-0 verify-from-code (đóng #1/#2/#3).
  - **Bước 3** implement (Quy tắc 11/12: `use_lconsist` default OFF, `utils/loss/binocular_consistency.py`, hook ≤10 dòng train.py, bg cùng-iter vì random_background). C=0 code.
  - **Bước 4** Gate-b: flag-OFF=A3 (N+loss-curve, KHÔNG PSNR) · non-degenerate (depth-corr-vs-COLMAP **KHÔNG GIẢM**, depth-var không sụp, grad/L_main in-band, test-PSNR 2-scene no-regress) · **per-scene specular guard** (1 scene regress>noise → abort, không để mean che — bài học GDAGS horns−0.605/HF-pilot horns−0.31). Note: bottleneck-verified chroma/specular minor trên LLFF (h4_ratio 0.18-0.36) → catastrophe-harm LLFF-tempered nhưng guard giữ làm insurance.
  - **Bước 5** pilot single-seed-42 full-8 (B/C/D, A reuse, 2-GPU split) → cell Δ≥+0.10 & no-catastrophe → multi-seed 137+9999.
- **Verdict gate**: A−C → có nên bỏ D_cycle; B/D → Lc thêm hay thay; mỗi nhánh verdict riêng (±0.10 floor multi-seed + per-scene guard).
- **Cleanup contract (rule 2026-05-18)**: nếu reject — script/module GIỮ local chỉ xóa server; code-edit (arguments/train.py) revert cả 2; báo re-sync list.

---

### [2026-05-18] Phase 14 RESULTS — L_consist pilot + dense-init Gate + HONEST REFRAME

**L_consist 2×2 pilot (single-seed-42 full-8, paired vs A3):**
- **B (A3+Lc) ❌ REJECT** — 3 cơ sở độc lập: (1) ΔB_PSNR=+0.041 « ±0.10 noise floor, std 0.229=5.6×mean (cherry-pick pattern: fortress+0.407/room+0.384 bù horns−0.318/leaves−0.130) = SATURATION; (2) **horns −0.318 catastrophe-guard fire** = **3rd independent confirm A3-spatial-alignment fragile** (HF-pilot horns−0.31 / GDAGS horns−0.605 / Lc horns−0.318 — 3 class cơ chế độc lập đều phá horns); (3) cost **+36.3% train-time** (508s vs 373s). ΔB_SSIM+0.003/ΔB_LPIPS−0.007 = trong noise.
- **C (A3−D_cycle) finding độc lập**: ΔC_PSNR=**−0.135**, C/horns=**−0.925** → bỏ D_cycle HẠI A3. **Phase-7 "D_cycle flips negative on strong backbone" KHÔNG replicate trên A3 cuối** → tự đính chính over-claim lần 3: **D_cycle CONFIRMED beneficial trên A3, GIỮ**. (Quyết "giữ"=status-quo, không cần multi-seed.)
- **D (A3−Dcyc+Lc) reject**: D−B=−0.045, 2 catastrophe (fern−0.258, horns−0.452). Lc KHÔNG thay được D_cycle.

**Dense-init Gate (no-train, matcher-agnostic, `scripts/p14_denseinit_gate.py`):**
- Auto-"GO" = **FALSE-ACCEPT do cherry-pick radius nhỏ nhất** (tự bắt lỗi script như L_consist MARGIN_ABS). SUBSTRATE sụp **22%(R8)→8.1%(R16)→3.5%(R24)**. Radius thực tế (densify nở ≫8px, A3 LFCF+AbsGS mạnh) = 16-24px → substrate <10%.
- **Cross-validate**: leaves NO_HELP=95.6% ↔ bottleneck H1 leaves=92.6% → Gate covis metric sound. Lỗi A3 = NO_HELP(covis<2, structural) + ALREADY(sparse-seeded) ≫ SUBSTRATE.
- **NHƯNG không tuyên bố đóng hẳn** (rút lại nén quá tay): Gate ALREADY-mask không phân biệt "sparse-near + densify-reached" vs "sparse-near nhưng densify-miss-đúng-spot". Cần refine: phân bố khoảng-cách-tới-sparse-gần-nhất cho high-error∧covisible pixel (cheap no-train) trước khi kết luận.
- Chỉ **room** giữ substrate (R16=26%,R24=15%) — 1/8 scene, không justify pipeline nặng + Phase-10A-risk + horns-frag.

**⚠️ HONEST REFRAME (user push-back đúng, ghi để session sau KHÔNG over-claim ceiling):**
- User nhắc: Phase 11 từng tuyên bố "loss-axis dead 6/6 = ceiling" → user đẩy **frequency-axis (LFCF+AbsGS) → +0.164 = chính 21.330 đang có**. Persistence của user đã đúng, "ceiling" của session-trước SAI. **Pattern: claim-exhausted của Claude có tiền sử sai khi có trục VẬT LÝ orthogonal chưa thử.**
- Phân biệt đúng: information-ceiling chỉ giải thích **xào lại CÙNG trục** saturate (L_consist/GDAGS/dense-init trên geometry-đã-solved). KHÔNG bác trục-vật-lý-MỚI rút thêm recoverable-signal — *đó chính xác là điều frequency/AbsGS đã làm* (better extraction, không phải new-info).
- Docs' own caveat **chưa giải**: "overfit gap 12.88 = reducible-overfit + irreducible-3view CHƯA tách" → H4 có mảnh reducible-overfit → anti-overfit qua trục-vật-lý-mới CÓ THỂ chạm (parallel mạnh với AbsGS chạm HF-overfit → +0.164).
- **Trục chưa đụng** (CORRECTED 2026-05-18 — verified vs decisions_log/results, KHÔNG vs MEMORY.md-index): (a) **anisotropy/shape-reg** (T12.4 liệt-kê-chưa-chạy, NHƯNG moderate-overlap với lfcf_diffscale="volume-preserving isotropify" verified lfcf.py — phải verify scope, KHÔNG clean-orthogonal); (c) **dn-splatter normal-prior** (survivor chưa đụng, nhưng nặng external+recipe-risk class). (d) dense-init Gate-refine.
  - ❌ **(b) density-aware dropout = ĐÃ REJECT Phase-2d [2026-04-21]** (decisions_log:633-648, track_a_b:1156; voxel −0.17/cov crash; CRS×density Phase-3α/3β cũng reject). Tôi đã sai khi list nó "untried" — tin MEMORY.md-index ("novelty/chưa test") + code-presence thay vì đọc FULL memory file (file ghi rõ "OUTCOME REJECTED") + grep decisions_log. **Lesson: "untried" PHẢI verify vs full-memory-file + decisions_log; index 1-dòng & code-presence KHÔNG đủ (code tồn tại TỪ thí nghiệm đã reject).**
- **KHÔNG accept 21.330 vội.** Áp đúng kỷ luật đã thắng Phase-13: chọn trục orthogonal chưa đụng → Gate rẻ → run nếu substrate. Honest expectation: không hứa thắng, nhưng các trục này thật sự chưa thử + có tiền lệ frequency.

---

### [2026-05-18] Phase 14 CLEANED + Phase 15 CONSOLIDATED (build trên Phase-13/A3 sạch)
- **Phase 14 L_consist = REJECTED → CLEANED**: revert production `arguments/__init__.py` (xoá block 5 flag use_lconsist/lconsist_*) + `train.py` (xoá hook ≤14 dòng sau L_depth) → grep 0 residue → **A3/Phase-13 byte-clean**. Module `utils/loss/binocular_consistency.py` = standalone reject → GIỮ LOCAL, xoá SERVER-only (rule 2026-05-18). Mọi hướng mới build TRÊN Phase-13/A3, KHÔNG stack lên L_consist.
- **Trạng thái HỢP NHẤT — đã đóng (định lượng, không hand-wave):**
  | Trục | Verdict | Bằng chứng |
  |---|---|---|
  | densify ×3 | exhausted | HF−0.31/D3−0.48/GDAGS≈0+horns−0.605 |
  | loss 6/6 + HF-pilot | exhausted | Phase-11 |
  | CRS 9/9 | exhausted | Phase-6→12 |
  | cross-view 5-fail | structural-dead | pseudo/DUSt3R/MPC/CRS-pull/covis |
  | **L_consist (Phase 14)** | **REJECT+CLEANED** | ΔB+0.041«±0.10 saturate + horns−0.318 (3rd-frag) + +36%cost |
  | dense-init | parked (Gate false-GO, realistic-R<10%) | leaves NO_HELP 95.6%↔H1 92.6% |
  | density-dropout | CLOSED | D3 anchor-half −0.32 net-negative (verified) |
  | **anisotropy blunt s_max/s_min** | **CLOSED** | Q4 77% high-aniso = legitimate-flat (8/8, robust) |
  | bonus | D_cycle CONFIRMED beneficial on A3 (giữ) | ΔC−0.135, horns−0.925 |
- **Phase 15 = trên A3-clean, 3-arm (user "test cả 3"):**
  - **A** blunt `s_max/s_min` — pilot (low-EV nhưng = **control falsify Q4-diagnostic**, vì diagnostic session này misfire nhiều).
  - **B** targeted `s_max-excess-vs-scene` — form data CHỈ vào (né 77%-flat), pilot.
  - **C1** dn-splatter **DSINE monocular-normal** — verified `regularization_strategy.py`/`scripts/dsine/`: external normal-estimator → **nặng-preprocess class dense-init** (tôi over-claim "nhẹ như DAV2-depth" — SAI, đã sửa); nhưng loss-integration (recipe-risk < init-replace), **duy nhất thêm info-MỚI**. Setup-then-pilot.
  - **C2** depth→normal self-consistency — DEPRIORITIZE (no-new-info → info-ceiling predicted-saturate ≈ L_consist-class; ≈ A3 pearson-depth/D_cycle overlap).
- A+B implement (Quy tắc 11/12 pattern, default OFF → A3 byte-identical) + pilot 2-GPU full-8 single-seed reuse-A3 per-scene-catastrophe-guard. C1 verify-DSINE + plan song song.

---

### [2026-05-19] Phase 15 shape-reg RESULT — REJECT+CLEANED; ROOT-CAUSE synthesis
- **Pilot full-8 seed42 (paired vs A3):**
  - **Ablunt** ΔPSNR −0.069 (5/8 neg) + **horns −0.218 catastrophe**, cost +26%. = **Q4-diagnostic VALIDATED** (Q4 dự đoán blunt hại vì 77% high-aniso=legitimate-flat → đúng hướng, empirically confirmed bằng pilot thật). Meta-value: diagnostic-discipline (bản Q4 có critique per-Gaussian của user) **dự đoán đúng** — quan trọng vì session này diagnostic misfire nhiều.
  - **Bexc3** ΔPSNR −0.007 std 0.164 (cherry-pick: fortress+0.308 bù **room −0.284 catastrophe**), cost +68%. REJECT.
  - **Bexc2** ΔPSNR −0.051 std 0.263, **horns −0.622 catastrophe**, cost +85%. REJECT.
  - → **Anisotropy A+B CLOSED empirically** (cả blunt lẫn targeted, pilot thật không chỉ diagnostic). User "test cả 3" = call đúng: validate Q4 (A) + đóng B mà diagnostic-một-mình không đóng được.
- **horns = 4th INDEPENDENT confirmation** A3-fragility: HF−0.31 / GDAGS−0.605 / L_consist−0.318 / Bexc2-horns−0.622. 4 class cơ chế độc lập đều phá horns.
- **ROOT-CAUSE synthesis (định luật bảo toàn — trả lời "sao fail mãi"):** A3 ở constrained-optimum của thông tin 3-view; tín hiệu recoverable đã rút hết (AbsGS là extraction cuối). Mọi cơ chế thêm vào (densify/loss/consistency/regularizer/dropout/shape — 6+ class) chỉ **TÁI PHÂN PHỐI** thông tin 3-view CỐ ĐỊNH → tổng zero-sum → **mean≈0 + std khổng lồ + horns-catastrophe** (chữ ký giống hệt 6+ lần = MỘT ràng buộc, không phải chuỗi xui). Phase-13/AbsGS thắng vì là **extraction** thành-phần-chưa-rút (không phải redistribution); rút xong không lặp. Lối ra cấu trúc = **THÊM thông tin ngoài 3-view**, KHÔNG reshape. Falsifiable: mọi reshape-mechanism tiếp → cùng chữ ký (đúng 4 lần session này).
- **C1 claim QUALIFIED** (user đúng — không "duy nhất tuyệt đối"): 2 hướng add-external-info còn lại = (1) C1 normal (loss-class, recipe-risk thấp HƠN nhưng — xem dưới — KHÔNG sạch), (2) dense-init/RoMa (high-recipe-risk Phase-10A).
- **Verify-(a) — gaussian_renderer KHÔNG output normal** (`__init__.py:203,235-240` = render/depth/alpha, no normal). → C1 tách: **C1a** normal-từ-rendered-depth (no-rasterizer-change, low-risk NHƯNG = depth-gradient-reg-informed-by-DSINE → redundancy-DAV2-depth = câu hỏi quyết định) vs **C1b** CUDA-rasterizer-normal (genuine-orientation-info NHƯNG heavy + đi ngược precedent "Stage-B CUDA deferred"). Tôi over-simplify "C1 loss nhẹ như DAV2" lần nữa — verify-(a) sửa.
- **RoMa honest**: dense-init park vì Gate-substrate <10% realistic-radius + recipe-calib Phase-10A — KHÔNG vì matcher kém. RoMa nâng match-quality = KHÔNG phải limiting-factor → **RoMa không giải cứu dense-init**. Trên bàn nhưng verdict không đổi bởi matcher (chỉ đáng nếu dense-init Gate-refine cho substrate khác trước).
- **NEXT (rẻ, trước mọi setup nặng):** (b) verify dn-splatter DSINE coord-system (cam vs world; mismatch=loss vô nghĩa); **(c) DECISIVE redundancy Gate** corr(∇DAV2-depth, DSINE-normal) trên 2-3 train-view no-train ~15ph → corr cao = C1a redundant → C1 chết rẻ (chỉ còn C1b-CUDA-heavy); corr thấp = C1a có info mới đáng integrate.
- **Cleanup Phase-15 (rule 2026-05-18):** revert production `arguments/__init__.py`+`train.py` → A3/Phase-13 clean (DONE, grep 0 residue); `utils/regularizer/shape_reg.py`+`scripts/p15_shape*` GIỮ-local xoá-SERVER; server `rm -rf output/p15_shape/` (logs giữ=evidence).

---

### [2026-05-19] Phase 16 p16 normal-Gate — CLOSED inconclusive; Phase 17 = pay heavy C1 pilot
- **p16 redundancy-Gate saga (5 fix-iters, stop-condition spent):** convention fix (DSINE_AXIS_M=diag([-1,1,1]) = dn-splatter verified, user-checked claims A–D) + robust-K-smooth + dual-anchor (A3-render vs DAV2-depth) + dcmp. Final: DAV2-anchor invalid-as-built (raw un-aligned → backproject-normal méo, dcmp≈84°, 7/8 UNRELIABLE); A3-anchor (only trustworthy) = **NO_WEAK lift≈1.26 < 1.5**. Meta: cheap depth-normal Gate **intrinsically can't triage C1 on sparse-view 3DGS**. Detail: memory `project_p16_gate_outcome`. p16 + `scripts/dsine_pkg/` (4-file vendor, audited verbatim) = keep-local evidence.
- **User decision (escalation):** Gate inconclusive ⇒ **trả pilot nặng định-đoạt** (KHÔNG deprioritize). → Phase 17.

### [2026-05-19] Phase 17 — C1 DSINE normal-prior IMPLEMENTED (pilot pending); PRE-REGISTERED
- **Mechanism (verify-from-code, user sửa 3 framing-error):**
  - C1a = **ADAPTATION cố ý, KHÔNG "mirror dn-splatter exactly"**. Verified `dn_model.py:590` `depth_im.detach()` → surface_normal KHÔNG differentiable; normal được supervise của dn-splatter = `normals_im` CUDA-rasterized (C1b). CRSGaussian renderer no normal → C1b heavy (loại). C1a = normal-từ-rendered-depth **differentiable** + L1 DSINE (no CUDA). **Mirror thật = CHỈ transforms** (surface pcd_to_normal @diag([1,-1,-1])→[0,1] ; DSINE raw @diag([-1,1,1])→[0,1]; cùng camera-[0,1] ⇒ né convention-hell p16) **+ L1** (AdaptiveNormal step<15k=L1; pilot 10k→L1).
- **PRE-REGISTERED PREDICTION (logged TRƯỚC pilot — predict-before-test):** dn-splatter detach depth ở surface_normal *chính xác vì* ∇(finite-diff cross-product rendered-depth) cực nhiễu (= nhiễu phá n_dep p16). C1a back-prop qua đúng operator đó trên 3DGS sparse rendered-depth → **DỰ ĐOÁN: degrade / no-gain, tập trung horns/foliage (floaty-depth)**. Pilot ra horns-catastrophe / saturate(mean≈0+var lớn) = mechanism **XÁC NHẬN, KHÔNG bất ngờ, KHÔNG re-engineer**. smooth=False (mirror) — không lén mitigation.
- **PRE-REGISTERED DECISION RULE (locked — chống multiple-comparison = audit#3 / kỷ luật τ NON-CANONICAL p16):** λ chính **=0.10** (dn-splatter default) **DUY NHẤT** quyết GO/NO, full-8×3-seed = **N=24** (chuẩn Phase-13 + project_3dgs_variance_floor). λ∈{0.05,0.20} = **SENSITIVITY-ONLY, NON-VERDICT, KHÔNG lật**. **GO ⟺** Δ_mean(C1L10 vs A3) ≥ **+0.10** ∧ 95%CI loại 0 ∧ (≥7/8 scene Δ≥0 ∧ horns Δ≥−0.05) ∧ cost-report. Khác = **NO → ĐÓNG C1** (đã trả pilot, đúng cam kết).
- **Implement (modular Quy tắc 10/11/12, build trên A3/Phase-13 clean):** NEW `utils/loss/c1_normal.py` (docstring honest "DEVIATES" + prediction). 4 flag ModelParams default OFF: `use_c1_normal`/`c1_normal_lambda`(0.10)/`c1_normal_start_iter`(0)/`c1_normal_dir`. train.py: one-shot load (key image-stem cross-process-stable) + hook sau L_depth (pattern y hệt). Default OFF ⇒ A3/Phase-13 byte-identical (verify VOFF N-criterion). NEW `scripts/p17_c1_preprocess_dsine.py` (DSINE/train-view, reuse dsine_pkg) + `p17_c1_pilot.sh` (P8_CORE+A3_FLAGS EXACT copy p15) + `p17_c1_analyze.py` (parse/CI mirror p13; verdict CHỈ C1L10).
- **Cleanup-on-reject (rule 2026-05-18):** NO → revert `arguments/__init__.py`+`train.py` → A3/Phase-13 clean + re-sync server; `utils/loss/c1_normal.py`+`scripts/p17*`+`dsine_pkg/` giữ-local xoá-server; `rm -rf output/p17_c1/` (logs+docs=evidence). GO → giữ all, update recipe.

### [2026-05-20] Phase 17 — C1a RESULT = ❌ NO (pre-registered prediction CONFIRMED); cleanup HOÃN (Gate-phẳng-C1a follow-up)
- **Pilot N=24** (C1L10 λ=0.10 vs A3, paired 3-seed×8-scene, A3 logs reuse): **Δ_mean = +0.006** · 95%CI **∋ 0** · **4/8** scene Δ≥0 · cost ≈ **2.5–3×** train (DSINE preprocess + per-iter normal-from-depth). **TẤT CẢ 4 GO-criteria FAIL** (mean<+0.10 ∧ CI∋0 ∧ non-neg<7/8 ∧ horns-catastrophe) ⇒ **VERDICT = NO** (locked, pre-registered — KHÔNG dùng λ{0.05,0.20} sensitivity để lật, đúng audit#3).
- **PRE-REGISTERED PREDICTION CONFIRMED (predict-before-test thành công):** foliage/thin + horns degrade ĐÚNG như log-trước-pilot. fern **−0.123**, leaves **−0.098**, horns **−0.203** (**5th horns-fragility confirm**: HF/GDAGS/Lc/Bexc2/**C1a**). trex per-seed loạn **+0.123 / −0.323 / −0.451** = chữ ký ∇(finite-diff rendered-depth) cực nhiễu (chính lý do dn-splatter `depth_im.detach()`). ⇒ KHÔNG bất ngờ, KHÔNG re-engineer (smooth=False, không lén mitigation — đúng cam kết).
- **6th info-conservation confirm:** C1a = thêm external-prior NHƯNG đi qua differentiable-∇depth-pathway lỗi → net reshape zero-sum (mean≈0 + horns-catastrophe), cùng signature densify×3/loss6+/CRS9/cross-view/L_consist/aniso. A3=21.330 vẫn constrained-3-view-optimum.
- **Mechanistic finding (user-driven, KHÔNG cherry-pick):** **fortress +0.473 ỔN ĐỊNH cả 3 seed** = orientation-prior giải ambiguity depth mặt-phẳng-textureless (external-info THẬT, không reshuffle). Win/hurt tách sạch theo geometry: phẳng/man-made thắng · foliage/thin/organic thua (DSINE-unreliable × ∇depth-noise cộng dồn). ⇒ **ý orientation-prior CÓ merit** trên scene structured; lỗi nằm ở **C1a differentiable-depth-normal implementation** (dn-splatter né bằng detach + CUDA-C1b).
- **Cleanup-on-reject HOÃN (KHÔNG thực thi):** user pivot → **Gate-phẳng-C1a** = **giả thuyết MỚI pre-registered** (KHÔNG lật NO của C1a-uniform; là cơ chế KHÁC, verdict N=24 TƯƠI riêng). Detect ô phẳng từ **chính normal-map DSINE @preprocess** (R̄ tile-level KxK, orientation-agnostic KHÔNG vertical-specific; τ từ **histogram-bimodal độc-lập-PSNR** = precedent tau_crs T2.7 / HF-R; mask tĩnh, 0 circularity, 0 train-cost). Gate = `valid &= planar_mask` (1 dòng, mask=None ⇒ C1a-uniform y hệt). Lằn ranh liêm chính DUY NHẤT: K/τ set từ tiêu chí độc-lập-PSNR, KHÔNG quét vs eval (= bẫy λ-grid). C1 module + `arguments`/`train.py` hook + `dsine_pkg/` + `scripts/p17*` **GIỮ NGUYÊN local làm substrate** (chưa revert, chưa rm). **C1b** (CUDA per-Gaussian normal = cơ chế thật dn-splatter, fix GỐC gradient-pathway, NHƯNG rủi-ro fork rasterizer lên A3) = lựa chọn nặng tách riêng. Caveat meta-overfit (gốc 8-scene) đã nêu thẳng; proof cuối = full-8 N=24. **Plan pre-registered Gate-phẳng-C1a: chờ user duyệt TRƯỚC khi code (Quy tắc 6).**

### [2026-05-20] Phase 17c — C1a-late + Tier 1/Tier 2 ∇depth pre-check = ❌ SUBSTRATE_ABSENT (pre-registered prediction REFUTED in OPPOSITE direction); predict-before-test 2/2 saved ~4-8h pilot
- **Background:** user chose to test C1a-late (defer start_iter from 0 → T = densify_until) BEFORE Gate-phẳng. Hypothesis: ∇depth noise tập trung trong densify-window iter<5000; defer-loss qua window đó → rescue. Cost-benefit: Tier 1+2 cheap diagnostic trước commit pilot N=24 ~4-8h.
- **Tier 1 (terminal noise vs C1a Δ per-scene, 15min):** metric `angle(n_raw, n_smooth_σ2px)` từ A3 iter 10000 ckpt, correlate với Δ_C1a per-scene paired N=24. **r_main = +0.3380** (N=8, ROBUST 0/8 LOO bin-flip). Bin = **refuted** (r ≥ −0.3) per pre-registered Q1 table. Mechanistic finding: **fortress noise CAO NHẤT (53°) NHƯNG Δ_C1a THẮNG TO NHẤT (+0.473)** = scene-noise-ranking KHÔNG phân biệt winners/losers. ⇒ Locked rule: SKIP cả Tier 2 + pilot.
- **User-exposed Tier 1 logic-gap (BEFORE seeing Tier 2):** lập luận của user — nếu noise giảm ĐỀU theo iter (multiplicative decay), ranking scene-noise stay-same → corr terminal = corr intrinsic-α-scene = +0.34 ngay cả khi C1a-late hypothesis ĐÚNG. Tier 1 không phân biệt được "noise architectural" (no drop) vs "noise uniform decay" (proportional drop). User-driven argument independent of Tier 1 outcome direction ⟹ KHÔNG phải post-hoc motivated reasoning. Escalate Tier 2 trajectory (vốn đã trong pre-registered tree cho bin ambiguous, mở rộng trigger cho logic-gap).
- **Tier 2 trajectory (re-train A3 8-scene seed42 với `--save_iterations 500 2000 5000 7000 10000`, ~4h GPU0 sequential do GPU1 bị colleague chiếm; rồi đo noise(scene, iter) ~10min):**

  | Scene | noise@500 | noise@10000 | growth × | drop_rel | class |
  |---|---|---|---|---|---|
  | fern | 22.7° | 40.6° | ×1.8 | −0.394 | NO |
  | flower | 13.9° | 44.4° | ×3.2 | −0.826 | NO |
  | fortress | 19.7° | 53.1° | ×2.7 | −0.620 | NO |
  | horns | 17.5° | 45.2° | ×2.6 | −0.679 | NO |
  | leaves | 17.1° | 50.8° | ×3.0 | −0.682 | NO |
  | orchids | 22.4° | 44.7° | ×2.0 | −0.429 | NO |
  | room | 10.3° | 20.8° | ×2.0 | −0.483 | NO |
  | trex | 22.5° | 34.0° | ×1.5 | −0.250 | NO |

  **Verdict locked: SUBSTRATE_ABSENT** (0/8 STRONG, 0/8 ANY-drop, 8/8 NO). Action per Q1 table: SKIP C1a-late → push Gate-phẳng/C1b.

- **Pre-registered prediction REFUTED in OPPOSITE direction (bất ngờ mạnh hơn dự đoán cũ):** giả thuyết nói noise drops; thực tế noise **GROWS ×1.5–3.2 mọi scene**. Cơ chế: finite-diff cross product nhạy với HF detail của depth; training tiến lên → nhiều Gaussian → depth có nhiều fine structure → finite-diff càng nhiễu. ⇒ **start_iter=0 thực ra là chế độ LOW-NOISE NHẤT** mà C1a có thể có (vì lấy luôn iter 500-2000 noise thấp). Cấu hình tốt nhất rồi vẫn fail N=24 ⟹ **C1a finite-diff pathway dead ARCHITECTURALLY**, không phải timing.
- **C1b case STRENGTHENED (empirical, không còn chỉ dựa lý luận):** Tier 2 đo cụ thể được ∇depth-pathway = nguồn lỗi kiến trúc. dn-splatter `depth_im.detach()` + CUDA per-Gaussian normal CHÍNH XÁC né được vấn đề này (gradient không chảy qua finite-diff, vào thẳng rotation/scale của Gaussian). C1b giờ có bằng chứng đo được, không phải dự đoán.
- **Predict-before-test ledger Phase 17c: 2/2 cheap-diagnostic refuted-correctly trước pilot.** Tier 1 (r=+0.34 ranking không predict) + Tier 2 (noise grows ×2-3 not drops) = cùng nhau loại C1a-late mà không tốn pilot. Tổng compute saved: ~4-8h server. Methodology + user-exposed logic-gap escalation = working as designed.
- **Cross-tab sub-finding (informational, KHÔNG verdict, FORBID over-claim per pre-reg):** r(drop_rel, Δ_C1a) = −0.30 N=8. Magnitude triage cấm, ghi nhận informational only.
- **Cleanup-on-reject HOÃN tiếp:** Phase 17c module/script giữ-local (scripts/p17c_tier1_*, p17c_tier2_*, p17c_a3_resave.sh, output/p17c_a3_resave/ ~2GB). Server-side: chỉ cần cleanup khi cả C1 axis đóng hoàn toàn (sau Gate-phẳng verdict). Logs/decisions evidence giữ.
- **Next decision (user pending):** 2 nhánh còn lại cho C1 axis:
  - **Gate-phẳng-C1a** — nhẹ (preprocess mask + 1 dòng AND, ~30min code + 4-8h pilot). Vẫn dùng finite-diff pathway, chỉ thu hẹp miền áp dụng. Plan pre-registered đã thiết kế xong từ 2026-05-19.
  - **C1b** (CUDA per-Gaussian normal) — nặng (~2-3 ngày code: fork rasterizer forward+backward, recompile, rủi-ro vỡ A3 21.330). Fix gốc gradient-pathway, Tier 2 vừa củng cố mechanistic case empirically. Conceptual design có; verify-from-code rasterizer submodule cần làm trước commit plan.
  - Đề xuất disciplined order: Gate-phẳng trước (cheap, low-risk). Pass → có rescue khỏi C1b. Fail → đã loại sạch finite-diff pathway → C1b mới defensible commit 2-3 ngày work.

### [2026-05-20] Phase 17e — Gate exploration (Step A/B/C) → gate-by-region/uncertainty KHÔNG viable; Phase 17 PAUSED; pivot RoMa + PDCNet+
- **Bối cảnh:** thay vì a-priori "planar gate", user pivot sang empirical "đo WHERE C1a thắng → derive gate". 3 step diagnostic trên A3 + C1L10 ckpts có sẵn (no train).
- **Step A — per-pixel improvement map** (`scripts/p17e_step_a_*`, render A3 vs C1L10 test-view, 3-seed): `improvement = L1_A3 − L1_C1` per pixel. Sign 7/8 match Δ_C1a N=24 (room sign-mismatch nhưng Δ=+0.032 trong noise floor). **Bất ngờ: win-rate ~50% MỌI scene** — fortress (winner lớn nhất) cũng chỉ 54% pixel C1a-thắng, KHÔNG concentrated. Orchids swing lớn nhất (p95 +0.042/p5 −0.037). ⇒ fortress +0.473 PSNR đến từ "nâng đều nhẹ khắp ảnh" + log-scale PSNR nhạy low-error scene, KHÔNG từ "vùng phẳng tập trung". Phản bác narrative gate-phẳng-by-region.
- **Step B — uncertainty-fixation hypothesis** (`scripts/p17e_step_b_*`): test "C1a thắng ở pixel A3-uncertain (L1_A3 cao)". Per-pixel `r(L1_A3, improvement)` + top-10%-L1_A3 capture rate. Locked rule: substrate ⟺ r ≥ 0.30 ∧ capture ≥ 30%. **Verdict: SUBSTRATE_ABSENT (1/8 — chỉ fortress: r=0.355, capture=31.8%, top10_win=71.8%).** 7 scene khác r yếu (0.13-0.20). **Oracle pixel-level perfect-gate headroom ≈ +0.82 dB** (informational) — headroom toán học CÓ, nhưng detector L1_A3 quá yếu để extract. top10_win >55% mọi scene = signal weak-but-consistent, KHÔNG đủ concentrated cho single-feature gate.
- **Step C — gate-by-L1_A3 simulation = ❌ INVALID by design (retracted):** script `p17e_step_c_*` project Δ_gated=+0.70 dB, NHƯNG đây là **artifact regression-to-mean / selection-bias**. Bug thiết kế (KHÔNG phải bug code): mask = top-10% L1_A3 = chọn pixel A3-tệ-nhất *by selection* → swap render C1 vào đó → C1-error ở đúng-chỗ-A3-tệ ≈ trung bình C1 (không extreme) → MSE drop giả. Bằng chứng artifact: gain ~+0.7 ĐỒNG ĐỀU cả 8 scene kể cả scene C1a HẠI (fern −0.12 → Δ_gated +0.66 vô lý); gate-10%-pixel "thắng" full-C1a-100% (bất khả). **Bài học: KHÔNG simulate training-time gate từ render tĩnh của 2 model train riêng — chỉ pilot thật trả lời được.** Step C đề xuất sai từ gốc, vứt.
- **A3 re-save sanity:** re-train A3 (Phase 17c, intermediate ckpts) iter 10000 vs p13 gốc — mean diff +0.010, mean|diff| 0.14 dB, dấu lẫn lộn 4+/4− → atomicAdd noise thuần, KHÔNG drift. Code re-save mirror P8_CORE+A3_FLAGS verbatim — confirmed consistent.
- **C1 axis status (Phase 17 PAUSED tại đây):** C1a-uniform NO (Phase 17) · C1a-late dead-architectural (17c) · gate-by-region phản-bác (17e Step A) · gate-by-uncertainty detector-yếu 1/8 (17e Step B) · simulation-projection bất khả (17e Step C). **C1b (CUDA per-Gaussian normal) = chưa thử** — deep-fix option duy nhất còn lại, defer (chưa commit 2-3 ngày fork rasterizer). Oracle headroom +0.82 dB tồn tại nhưng không gate cheap nào extract được trên finite-diff pathway.
- **Predict-before-test ledger:** 17c 2/2 (Tier 1+2). 17e: Step A/B observational hợp lệ; Step C = self-caught design flaw (retracted trước khi tốn pilot — discipline vẫn hoạt động, chỉ chậm 1 nhịp).
- **PIVOT → RoMa + PDCNet+** (dense correspondence networks). User chuyển hướng. RoMa từng note "high-recipe-risk, match-quality≠limiting" (CLAUDE.md Phase-15) — direction mới sẽ verify lại claim đó. C1 (incl C1b) để mở, chưa đóng dứt khoát; Phase 17 work giữ-local làm evidence. Chi tiết RoMa/PDCNet+ plan: Phase 18 (TBD).
- **Files Phase 17e giữ-local** (diagnostic, keep cho reference): `scripts/p17e_step_{a,b,c}_*.py`. Server cleanup hoãn tới khi C1 axis đóng hoàn toàn.

---
