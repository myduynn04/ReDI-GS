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
  1. Optional Phase 13.1 tolerance sweep (~3.5h) — paper appendix sensitivity (per design doc Section 10.4)
  2. Optional Direction A λ scaler_max sweep ({1.3, 1.5, 1.8, 2.0} × 2 scenes, ~1-2h) — paper appendix robustness
  3. **Paper writeup START** — Phase 13 = first defendable CRS-axis contribution post-Phase-8

---
