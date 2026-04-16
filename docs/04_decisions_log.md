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

---