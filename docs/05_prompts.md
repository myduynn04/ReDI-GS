# Prompts — CRSGaussian Implementation
> Copy-paste lần lượt vào Claude Code theo thứ tự task.
> Mỗi session: SYSTEM PROMPT trước → rồi mới paste prompt task.
> Tick [x] trong 03_task_queue.md sau khi xong mỗi task.

---

## SYSTEM PROMPT — Dán vào đầu MỌI session

```
Bạn là Technical Lead của dự án CRSGaussian.

Đọc các file sau theo thứ tự:
1. CLAUDE.md
2. docs/03_task_queue.md
3. docs/04_decisions_log.md

Sau khi đọc, bạn hoạt động theo 4 vai trò đồng thời:

ROLE 1 — IMPLEMENTER
  Viết code sạch, có type hints, docstring đầy đủ.
  Tuân thủ cấu trúc trong docs/02_codebase_map.md.
  Không implement quá 1 task mỗi session.

ROLE 2 — QA ENGINEER
  Sau mỗi function: viết unit test ngay trong file.
  Chạy test và báo cáo kết quả trước khi chuyển bước tiếp.
  Không tiếp tục nếu test chưa pass.

ROLE 3 — TECH REVIEWER
  Sau khi implement: tự review code vừa viết.
  Phát hiện và báo cáo: edge cases, potential bugs, memory leaks.
  Đề xuất cải thiện nhưng không tự ý implement nếu ngoài scope.

ROLE 4 — PROJECT TRACKER
  Cập nhật docs/03_task_queue.md sau mỗi task (tick [x]).
  Ghi số kết quả ngay khi có.
  Nếu phát hiện blocker → ghi vào Notes/Blockers.
  Nếu cần quyết định thiết kế mới → ghi vào docs/04_decisions_log.md.

QUY TẮC BẮT BUỘC:
  - LUÔN đọc file gốc trước khi sửa — không đoán API
  - LUÔN tóm tắt hiểu biết trước khi propose implementation
  - LUÔN chờ confirm trước khi bắt đầu code
  - KHÔNG sửa file trong FSGS/, DNGaussian/, LoopSparseGS/, DepthRegularizedGS/
  - KHÔNG implement nhiều hơn 1 task mỗi session
  - KHÔNG dùng confidence attribute của CoR-GS cho CRS
  - GFS metric (C3) — HOLD

QUY TẮC CHÚ THÍCH CODE — BẮT BUỘC mọi thay đổi:
  Khi thêm hoặc sửa bất kỳ đoạn code nào, PHẢI làm đủ 3 việc:

  (1) Header block ở đầu mỗi function/class thêm mới:
  # ============================================================
  # [CRSGaussian] Task: TXX — tên task
  # File: CoR-GS/tên_file.py  (TẠO MỚI / SỬA)
  # Mục đích: giải thích ngắn function này làm gì
  # Được gọi từ: file nào, khi nào
  # ============================================================

  (2) Inline comment tại mỗi logic quan trọng:
  Giải thích TẠI SAO làm vậy, không chỉ LÀM GÌ.
  Ví dụ:
    # Chỉ update visible Gaussians (radii > 0)
    # Lý do: Gaussian bị occlude không thể so sánh depth với prior
    vis = out['visibility_filter']

  (3) Báo cáo tóm tắt sau khi implement:
  "File đã sửa/tạo: [danh sách]
   Chú thích đã thêm: [danh sách logic được comment]"

Bắt đầu bằng cách tóm tắt:
1. Task hiện tại là gì
2. File nào sẽ đọc/sửa
3. Tiêu chí pass/fail cho task này

Chờ tôi confirm rồi mới tiếp tục.
```

---

## Thứ tự tasks

```
PHASE 0 — Baseline (bắt buộc trước khi viết bất kỳ dòng code CRS nào)
  T0.0 ✓  T0.X ✓  T0.5 [~]  T0.5b [~]  T0.6  T0.6b

PHASE 1 — Depth Alignment
  T1.1 → T1.2 → T1.3 → T1.4

PHASE 2 — CRS Module
  T2.1 → T2.2 → T2.3 → T2.4 → T2.5 → T2.6 → T2.7

PHASE 3 — Adaptive Depth Loss
  T3.1+T3.2 → T3.3

PHASE 4 — Position Constraint + Pruning
  T4.1 → T4.2 → T4.3
```

---

---

# PHASE 0 — Baseline & Setup

---

## T0.0 — Grep verify ICP/gs1 ✓ DONE

Kết quả: PASS. gaussiansN default=1, ICP chỉ chạy khi args.coprune=True.

---

## T0.X — Extract reprojection errors ✓ DONE

Kết quả: PASS. BasicPointCloud có errors field, fetchPly đọc đúng.

---

## T0.5 — CoR-GS 2-field baseline

**Tiêu chí pass:** PSNR trong 19.5–21.0.

```
Task T0.5: Chạy CoR-GS baseline chính thức.
  Dataset: LLFF fern, 3 views
  Mode: gaussiansN=2, --coreg, --coprune

BƯỚC 1 — Đọc scripts/run_llff.sh, paste command → chờ confirm → chạy.
BƯỚC 2 — Verify data: ls data/nerf_llff_data/fern/
BƯỚC 3 — Chạy MẶC ĐỊNH, KHÔNG đổi hyperparameter.
BƯỚC 4 — Ghi: PSNR=? SSIM=? LPIPS=?
BƯỚC 5 — Render depth map → output/baseline_2field/fern/

CẢNH BÁO: PSNR < 19.5 hoặc > 21.0 → DỪNG và báo cáo.

QA CHECK:
  □ Command paste trước khi chạy
  □ PSNR trong range 19.5–21.0
  □ Depth map đã lưu
  □ Ghi vào docs/03_task_queue.md T0.5
```

---

## T0.5b — Single-field baseline

**Tiêu chí pass:** Gap đã tính và interpreted.

```
Task T0.5b: Chạy single-field baseline.
  Dataset: LLFF fern, 3 views
  Mode: gaussiansN=1 (default, không cần flag)

BƯỚC 1 — Confirm: gaussiansN default=1 từ argparse (đã biết từ T0.0).
          Chạy ngay không cần thêm flag.
BƯỚC 2 — Cùng hyperparameters như T0.5.
BƯỚC 3 — Điền bảng:
  | Run             | PSNR | SSIM | LPIPS |
  | T0.5  (2-field) |      |      |       |
  | T0.5b (1-field) |      |      |       |
  | Gap             |      |      |       |
BƯỚC 4 — Ghi interpretation vào docs/04_decisions_log.md:
  Gap > 0.5 dB → co-reg quan trọng → CRS phải compensate mạnh
  Gap 0.2–0.5  → moderate
  Gap < 0.2    → single-field đủ mạnh → CRS có path rõ ràng

QA CHECK:
  □ Cùng hyperparameters với T0.5
  □ Gap đã tính và interpreted
  □ Depth map lưu output/baseline_1field/fern/
```

---

## T0.6 + T0.6b — DTU baselines

```
Task T0.6 + T0.6b: DTU scan24, 2-field và 1-field.

BƯỚC 1 — Đọc scripts/run_dtu.sh.
BƯỚC 2 — T0.6: gaussiansN=2, coreg, coprune.
BƯỚC 3 — T0.6b: gaussiansN=1 (default).
BƯỚC 4 — Điền bảng và tính gap.

CHECKPOINT Phase 0:
  LLFF gap: ? dB | DTU gap: ? dB
  GO/NO-GO cho Phase 1?
```

---

---

# PHASE 1 — Depth Alignment

---

## T1.1 — Đọc DepthRegularizedGS

```
Task T1.1: Đọc DepthRegularizedGS — hiểu weighted alignment.

BƯỚC 1:
  grep -rn "weighted\|scale.*align\|argmin\|least_square" DepthRegularizedGS/

BƯỚC 2 — Trả lời với code snippets:
  Q1: File nào? Function signature?
  Q2: Input/Output?
  Q3: w(p) tính thế nào? (paste code)
  Q4: Solve WLS closed form hay scipy?
  Q5: Edge cases nào được handle?

QA CHECK: □ 5 câu hỏi có code evidence  □ KHÔNG implement gì
```

---

## T1.2 — Tạo depth_alignment.py

**Tiêu chí pass:** Test 1: s≈2.0, t≈3.0 (±0.1).

```
Task T1.2: Tạo CoR-GS/depth_alignment.py

BƯỚC 1 — Đọc thêm:
  a. dataset_readers.py (T0.X version) — format reprojection_errors
  b. train.py dòng 1–80 — cách scene/cameras load
  Báo cáo → chờ confirm → implement.

BƯỚC 2 — Implement với header block + inline comments:

# ============================================================
# [CRSGaussian] Task: T1.2 — weighted_scale_alignment
# File: CoR-GS/depth_alignment.py  (TẠO MỚI)
# Mục đích: Align relative depth từ DepthAnything V2 về scale
#           thực của scene dùng COLMAP sparse points làm anchor.
#           Points tin cậy hơn (reproj error thấp) có weight cao hơn.
# Được gọi từ: train.py, một lần trước training loop
# ============================================================
def weighted_scale_alignment(
    D_dense: torch.Tensor,              # [H, W] relative depth từ DepthAnything V2
    D_sparse: torch.Tensor,             # [H, W] COLMAP depth, 0 = không có point
    reprojection_errors: torch.Tensor,  # [H, W] COLMAP reproj error, 0 = không có
    min_valid: int = 10,                # fallback naive LS nếu ít hơn n points
) -> tuple[torch.Tensor, float, float]:
    """
    Returns: (D_aligned [H,W], scale s, offset t)
    Algorithm:
      1. valid_mask = D_sparse > 0
      2. w = 1 / clip(reproj_errors[valid_mask], 1e-6)
         # Points có reproj error thấp = đáng tin hơn → weight cao hơn
      3. Nếu sum(valid) < min_valid → w = ones (naive LS, log warning)
         # Fallback khi scene có ít COLMAP points
      4. Closed form WLS: A=[D_dense_valid, ones]
         [s,t] = (A^T W A)^{-1} A^T W b   với b = D_sparse_valid
      5. D_aligned = s * D_dense + t
    """
    pass

BƯỚC 3 — Unit tests:
  Test 1: D_sparse=2*D_dense+3+noise → s≈2.0, t≈3.0
  Test 2: valid < min_valid → không crash
  Test 3: D_aligned.shape == D_dense.shape

QA CHECK:
  □ Header block đầy đủ
  □ Inline comments tại: valid_mask, weight calc, fallback, WLS solve
  □ Test 1 PASS: s≈2.0, t≈3.0 (±0.1)
  □ Test 2, 3 PASS
  □ Báo cáo tóm tắt file đã tạo + comments đã thêm
```

---

## T1.3 — Setup depth structure trong train.py

**Tiêu chí pass:** 10 iter không crash. Print xuất hiện.

```
Task T1.3: Setup depth_range và depth_prior_dict trong train.py.

BƯỚC 1 — Đọc train.py. Ghi line numbers:
  a. scene = Scene(...)
  b. for iteration in range(...)
  Tóm tắt → chờ confirm.

BƯỚC 2 — Thêm sau scene init (với inline comments):

  import statistics

  # ── [CRSGaussian T1.3] Tính depth_range cho toàn scene ──
  # Dùng median thay mean: robust với outlier cameras (rất gần/xa)
  # Tính một lần, dùng suốt training để normalize D_i và epsilon_depth
  train_cams = scene.getTrainCameras()
  nears = [cam.near for cam in train_cams]
  fars  = [cam.far  for cam in train_cams]
  depth_range = statistics.median(fars) - statistics.median(nears)

  # ── [CRSGaussian T1.3] depth_prior_dict: placeholder, fill ở T1.4 ──
  # Key: cam.uid, Value: Tensor[H,W] aligned depth prior
  depth_prior_dict = {}

  print(f"[CRS] depth_range={depth_range:.4f}, cameras={len(train_cams)}")

BƯỚC 3 — Chạy 10 iter, confirm không crash.

QA CHECK:
  □ Inline comments tại depth_range và depth_prior_dict
  □ depth_range dùng median
  □ 10 iter không crash
  □ Báo cáo dòng nào đã thêm trong train.py
```

---

## T1.4 — Tạo depth_model.py và fill depth_prior_dict

**Tiêu chí pass:** s>0 mọi camera, 50 iter không crash.

```
Task T1.4: Tạo depth_model.py và fill depth_prior_dict.

BƯỚC 1 — Đọc:
  a. scene/cameras.py — Camera.original_image: shape? dtype? range?
  b. train.py — images load thế nào?
  Báo cáo → chờ confirm.

BƯỚC 2 — Tạo CoR-GS/depth_model.py với header:

# ============================================================
# [CRSGaussian] Task: T1.4 — DepthModel
# File: CoR-GS/depth_model.py  (TẠO MỚI)
# Mục đích: Wrapper DepthAnything V2 để predict relative depth.
#           Precompute một lần trước training, lưu vào depth_prior_dict.
# Được gọi từ: train.py, trước training loop
# ============================================================
class DepthModel:
    def __init__(self, encoder='vitl', device='cuda'): ...
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """[3,H,W] range [0,1] → depth [H,W] relative"""
        ...

BƯỚC 3 — Fill depth_prior_dict trong train.py với inline comments:

  # ── [CRSGaussian T1.4] Precompute aligned depth prior ──
  # Chạy 1 lần trước training loop, không chạy lại trong loop
  # depth_model bị xóa sau khi xong để giải phóng VRAM
  depth_model = DepthModel('vitl', 'cuda')
  for cam in scene.getTrainCameras():
      # Predict relative depth từ DepthAnything V2
      D_dense = depth_model.predict(cam.original_image)

      # Project COLMAP 3D points lên camera để lấy sparse depth
      D_sparse, reproj_map = project_colmap_to_image(
          scene.point_cloud.points,
          scene.point_cloud.reprojection_errors, cam)

      # Align scale của D_dense về scale thực của scene
      D_aligned, s, t = weighted_scale_alignment(D_dense, D_sparse, reproj_map)
      depth_prior_dict[cam.uid] = D_aligned.cuda()
      print(f"  [CRS] cam {cam.uid}: s={s:.3f}, t={t:.3f}")

  # Xóa model ngay sau khi precompute xong — không cần trong training loop
  del depth_model; torch.cuda.empty_cache()

BƯỚC 4 — Visualize 1 depth map → output/debug_depth_prior.png
BƯỚC 5 — 50 iter không crash.

QA CHECK:
  □ Header block trong depth_model.py
  □ Inline comments tại: predict, project, align, del model
  □ s > 0 mọi cameras (nếu s < 0 → bug alignment)
  □ 50 iter không crash
  □ Báo cáo: file tạo + dòng sửa trong train.py + comments đã thêm
```

---

---

# PHASE 2 — CRS Module

---

## T2.1 — Đọc CoR-GS co-regularization logic

```
Task T2.1: Đọc CoR-GS co-reg.

BƯỚC 1:
  grep -n "coreg\|co_reg\|pseudo\|disagreement\|warp" CoR-GS/train.py

BƯỚC 2 — Trả lời với code snippets:
  Q1: Co-reg enforce consistency thế nào?
  Q2: Pseudo-view tạo ra thế nào?
  Q3: compute_prune_mask() — paste toàn bộ function
  Q4: Có warp function nào không?
  Q5: Phần nào tái dụng được cho R_i?

QA CHECK: □ 5 câu hỏi có snippets  □ KHÔNG implement gì
```

---

## T2.2 — Thêm _crs_score vào gaussian_model.py

**Tiêu chí pass:** 4 unit tests pass. KHÔNG dùng confidence.

```
Task T2.2: Thêm _crs_score attribute.

QUY TẮC: KHÔNG dùng confidence — nó weight alpha compositing trong rasterizer.

BƯỚC 1 — Đọc gaussian_model.py toàn bộ. Báo cáo:
  a. Tất cả self._* trong __init__()
  b. densification_postfix() — paste concat pattern
  c. prune_points() — paste pattern
  d. capture()/restore() — tuple order
  Chờ confirm → implement.

BƯỚC 2 — Sửa 6 chỗ với inline comments:

(1) __init__():
    # ── [CRSGaussian T2.2] CRS score buffer ──
    # Lưu ở logit space (trước sigmoid) để EMA tự nhiên hơn
    # Khởi tạo = 0 → sigmoid(0) = 0.5 (neutral)
    # Không dùng informed init vì CRS không active trước T_warmup
    self._crs_score = torch.empty(0)

(2) create_from_pcd():
    # ── [CRSGaussian T2.2] Khởi tạo CRS₀ = neutral 0 (logit) ──
    # CRS không được dùng trước T_warmup → init value không quan trọng
    # Tất cả Gaussians bắt đầu neutral, CRS sẽ update sau T_warmup
    self._crs_score = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")

(3) densification_postfix():
    # ── [CRSGaussian T2.2] Gaussian mới sinh → CRS₀ = neutral ──
    # Gaussian mới chưa có history → không biết tốt hay xấu → 0.5
    new_crs = torch.zeros((new_xyz.shape[0], 1), device="cuda")
    self._crs_score = torch.cat([self._crs_score, new_crs], dim=0)

(4) prune_points(mask):
    # ── [CRSGaussian T2.2] Prune _crs_score theo mask ──
    self._crs_score = self._crs_score[valid_points_mask]

(5) @property get_crs:
    # ── [CRSGaussian T2.2] Đọc CRS score ──
    # sigmoid chuyển logit space → probability [0,1]
    # squeeze(-1): [N,1] → [N]
    @property
    def get_crs(self) -> torch.Tensor:
        """CRS per Gaussian. Range (0,1). Shape [N]."""
        return torch.sigmoid(self._crs_score).squeeze(-1)

(6) capture()/restore(): thêm _crs_score đúng vị trí.

BƯỚC 3 — Unit tests:
  Test 1: get_crs.shape == (N,) ✓
  Test 2: get_crs range (0,1) ✓
  Test 3: densified → CRS ≈ 0.5 ✓
  Test 4: prune giảm shape đúng ✓

QA CHECK:
  □ Inline comments tại 6 chỗ sửa
  □ KHÔNG dùng confidence, KHÔNG enable use_confidence
  □ _crs_score ở logit space (zeros, không phải 0.5)
  □ 4 tests pass
  □ Báo cáo: 6 dòng/block sửa trong gaussian_model.py
```

---

## T2.3 — compute_depth_consistency (D_i)

**Tiêu chí pass:** 4 unit tests pass. Invisible → D=0.5.

```
Task T2.3: Implement compute_depth_consistency() — tạo crs_module.py

THIẾT KẾ:
  - 3D projection (không dùng rendered depth per-pixel)
  - Chỉ update visible Gaussians
  - Occluded → D=0.5 (neutral)

BƯỚC 1 — Đọc:
  a. gaussian_renderer/__init__.py — visibility_filter convention
  b. scene/cameras.py — cam.R, cam.T convention
  Báo cáo → chờ confirm.

BƯỚC 2 — Implement CoR-GS/crs_module.py với đầy đủ comments:

# ============================================================
# [CRSGaussian] Task: T2.3 — CRS Module
# File: CoR-GS/crs_module.py  (TẠO MỚI)
# Mục đích: Tính D_i (depth consistency) và R_i (reprojection
#           consistency) per-Gaussian để compute CRS score.
#           CRS = tín hiệu chất lượng geometry per-Gaussian,
#           dùng để điều phối depth loss, densify, và pruning.
# Được gọi từ: train.py, sau T_warmup, mỗi 100 iter
# ============================================================

def project_to_depth(xyz_world, cam) -> torch.Tensor:
    # ── Project world coords về depth trong camera space ──
    # Dùng camera extrinsics (R, T) để chuyển world → camera space
    # Lấy trục z (depth theo trục camera)
    R = torch.tensor(cam.R, dtype=torch.float32, device=xyz_world.device)
    T = torch.tensor(cam.T, dtype=torch.float32, device=xyz_world.device)
    xyz_cam = (xyz_world @ R.T) + T
    return xyz_cam[:, 2]   # z-depth

def project_to_pixel(xyz_world, cam) -> torch.Tensor:
    # ── Project world coords về pixel (x,y) ──
    # fx, fy tính từ FovX, FovY (không hardcode)
    import math
    R = torch.tensor(cam.R, dtype=torch.float32, device=xyz_world.device)
    T = torch.tensor(cam.T, dtype=torch.float32, device=xyz_world.device)
    xyz_cam = (xyz_world @ R.T) + T
    fx = cam.image_width  / (2*math.tan(cam.FovX/2))
    fy = cam.image_height / (2*math.tan(cam.FovY/2))
    # Clamp z để tránh division by zero (Gaussian phía sau camera)
    z = xyz_cam[:,2].clamp(1e-6)
    x = xyz_cam[:,0]/z*fx + cam.image_width/2
    y = xyz_cam[:,1]/z*fy + cam.image_height/2
    return torch.stack([x, y], dim=-1)

def compute_depth_consistency(gaussians, viewpoint_cams, depth_prior_dict,
                               render_fn, pipe, bg, depth_range) -> torch.Tensor:
    # ============================================================
    # [CRSGaussian] D_i = 1 - |d_proj - d_prior| / depth_range
    # Chỉ update visible Gaussians. Occluded → giữ D=0.5 (neutral)
    # ============================================================
    N = gaussians.get_xyz.shape[0]
    D_sum   = torch.zeros(N, device='cuda')
    D_count = torch.zeros(N, device='cuda')

    for cam in viewpoint_cams:
        if cam.uid not in depth_prior_dict: continue

        # Render để lấy visibility — dùng no_grad vì chỉ đọc thông tin
        with torch.no_grad():
            out = render_fn(cam, gaussians, pipe, bg)

        # Chỉ tính D_i cho Gaussian được rasterizer thấy (radii > 0)
        # Gaussian bị occlude bởi surface khác không nên bị penalize
        vis = out['visibility_filter']
        if vis.sum() == 0: continue

        xyz = gaussians.get_xyz[vis]

        # Dùng 3D projection, KHÔNG dùng rendered depth per-pixel
        # Lý do: rendered depth là alpha-weighted avg của nhiều Gaussians,
        #        không phải depth chính xác của Gaussian i
        d_proj  = project_to_depth(xyz, cam)

        # Lấy depth prior tại pixel tương ứng với vị trí Gaussian
        pix     = project_to_pixel(xyz, cam)
        H, W    = depth_prior_dict[cam.uid].shape
        px      = pix[:,0].clamp(0, W-1).long()
        py      = pix[:,1].clamp(0, H-1).long()
        d_prior = depth_prior_dict[cam.uid][py, px]

        # D_i = 1 - normalized_error, clamp về [0,1]
        d_i = (1.0 - (d_proj - d_prior).abs() / (depth_range + 1e-8)).clamp(0, 1)

        D_sum[vis]   += d_i
        D_count[vis] += 1

    # Average D_i qua các cameras
    D = torch.full((N,), 0.5, device='cuda')  # default neutral
    valid = D_count > 0
    D[valid] = D_sum[valid] / D_count[valid]

    # Gaussian không visible từ bất kỳ camera nào → giữ neutral 0.5
    # Không update vì không có thông tin để đánh giá geometry
    return D

BƯỚC 3 — Unit tests:
  Test 1: shape [N] ✓  Test 2: range [0,1] ✓
  Test 3: invisible → D=0.5 ✓  Test 4: depth match → D≈1.0 ✓

QA CHECK:
  □ Header block cho crs_module.py và compute_depth_consistency
  □ Inline comments tại: render no_grad, visibility filter, 3D projection,
    d_prior lookup, neutral default
  □ 4 tests pass
  □ Báo cáo: file tạo + số comments đã thêm
```

---

## T2.4 — compute_reprojection_consistency (R_i)

**Tiêu chí pass:** Renders cached. 3 tests pass.

```
Task T2.4: Implement compute_reprojection_consistency() — thêm vào crs_module.py

BƯỚC 1 — Từ T2.1: có warp function có sẵn không?

BƯỚC 2 — Implement với comments:

def warp_pixels(pixels_src, depth_src, cam_src, cam_dst):
    # ── Warp pixel coords từ cam_src → cam_dst dùng depth ──
    # Bước 1: unproject pixels + depth → 3D world
    # Bước 2: re-project 3D world → cam_dst pixels
    # Mục đích: giả lập nhìn cùng điểm từ góc khác
    import math
    fx = cam_src.image_width/(2*math.tan(cam_src.FovX/2))
    fy = cam_src.image_height/(2*math.tan(cam_src.FovY/2))
    # Unproject: pixel → camera space dùng depth
    x_cam = (pixels_src[:,0] - cam_src.image_width/2)  / fx * depth_src
    y_cam = (pixels_src[:,1] - cam_src.image_height/2) / fy * depth_src
    xyz_cam = torch.stack([x_cam, y_cam, depth_src], -1)
    # Camera → world (inverse của world→camera transform)
    R_src   = torch.tensor(cam_src.R, dtype=torch.float32, device=pixels_src.device)
    T_src   = torch.tensor(cam_src.T, dtype=torch.float32, device=pixels_src.device)
    xyz_world = (xyz_cam - T_src) @ R_src
    # Re-project world → cam_dst pixels
    return project_to_pixel(xyz_world, cam_dst)

def compute_reprojection_consistency(gaussians, viewpoint_cams, render_fn, pipe, bg):
    # ============================================================
    # [CRSGaussian] R_i = 1 - mean_j(|color_i - warp(color_j)|) / 255
    # Cache renders để tránh re-render nhiều lần
    # Invisible → R=0.5 (neutral)
    # ============================================================
    N = gaussians.get_xyz.shape[0]

    # Cache tất cả renders trước — tránh re-render O(N²)
    renders, visibilities = {}, {}
    with torch.no_grad():
        for cam in viewpoint_cams:
            out = render_fn(cam, gaussians, pipe, bg)
            renders[cam.uid]      = {'color': out['render'], 'depth': out['depth']}
            visibilities[cam.uid] = out['visibility_filter']

    R_sum, R_count = torch.zeros(N,'cuda'), torch.zeros(N,'cuda')

    for cam_a in viewpoint_cams:
        vis_a = visibilities[cam_a.uid]
        if vis_a.sum() == 0: continue

        xyz_a   = gaussians.get_xyz[vis_a]
        pix_a   = project_to_pixel(xyz_a, cam_a)
        H, W    = renders[cam_a.uid]['color'].shape[1:]
        px_a    = pix_a[:,0].clamp(0,W-1).long()
        py_a    = pix_a[:,1].clamp(0,H-1).long()
        # Color của Gaussian nhìn từ cam_a (scale 0-255)
        color_a = renders[cam_a.uid]['color'][:, py_a, px_a].T * 255

        diffs = []
        for cam_b in viewpoint_cams:
            if cam_b.uid == cam_a.uid: continue

            # Warp pixel từ cam_a sang cam_b dùng depth của cam_a
            # → lấy color tại vị trí tương ứng trong render của cam_b
            d_a     = renders[cam_a.uid]['depth'][0, py_a, px_a]
            pix_b   = warp_pixels(pix_a, d_a, cam_a, cam_b)
            pb_x    = pix_b[:,0].clamp(0,W-1).long()
            pb_y    = pix_b[:,1].clamp(0,H-1).long()
            color_b = renders[cam_b.uid]['color'][:, pb_y, pb_x].T * 255

            # Chênh lệch màu (mean over RGB channels)
            diffs.append((color_a - color_b).abs().mean(-1))

        if diffs:
            # R_i: 1 = màu nhất quán, 0 = màu hoàn toàn khác nhau
            r_i = 1.0 - torch.stack(diffs).mean(0) / 255.0
            R_sum[vis_a]   += r_i.clamp(0, 1)
            R_count[vis_a] += 1

    # Gaussian không visible → giữ neutral 0.5
    R = torch.full((N,), 0.5, device='cuda')
    valid = R_count > 0
    R[valid] = R_sum[valid] / R_count[valid]
    return R

QA CHECK:
  □ Header block cho warp_pixels và compute_reprojection_consistency
  □ Inline comments tại: cache renders, warp logic, color scale, neutral default
  □ Renders cached (không re-render trong loop)
  □ 3 tests pass
  □ Báo cáo: functions thêm vào crs_module.py + comments
```

---

## T2.5 — update_crs() với EMA

**Tiêu chí pass:** 3 tests pass. CRS range [0.08, 0.92].

```
Task T2.5: Implement update_crs() — thêm vào crs_module.py

BƯỚC 1 — Implement với comments:

def update_crs(gaussians, D, R, w1=0.5, w2=0.5, ema=0.9) -> None:
    # ============================================================
    # [CRSGaussian] Update _crs_score dùng EMA trên logit space
    # CRS_new_logit = 5.0 * (w1*D + w2*R - 0.5)
    #   → scale × 5.0 để CRS ∈ [0.08, 0.92] thay vì [0.5, 0.73]
    #   → -0.5 để center logit về 0 khi D=R=0.5
    # EMA trên logit space (không phải sigmoid output):
    #   → Tránh squeeze về mean khi EMA trên bounded [0,1]
    #   → Thay đổi từ từ → training ổn định hơn
    # ============================================================
    with torch.no_grad():
        # Scale và center logit để tận dụng full range sigmoid
        # D, R ∈ [0,1] → (w1*D+w2*R) ∈ [0,1] → (... - 0.5) ∈ [-0.5, 0.5]
        # × 5.0 → logit ∈ [-2.5, 2.5] → sigmoid ∈ [0.08, 0.92]
        crs_new_logit = 5.0 * (w1 * D + w2 * R - 0.5)

        # EMA: smoothing để tránh oscillation khi D/R noisy
        # ema=0.9: 90% giữ history, 10% cập nhật mới
        gaussians._crs_score.data = (
            ema * gaussians._crs_score.data +
            (1 - ema) * crs_new_logit.unsqueeze(-1)
        )

BƯỚC 2 — Unit tests:
  Test 1: Convergence — 50 updates D=R=0.8
          → CRS → sigmoid(5*(0.8-0.5)) = sigmoid(1.5) ≈ 0.82 ✓
  Test 2: EMA rate — 1 update ema=0.9 → score thay đổi 10% ✓
  Test 3: Gradient intact ✓
  Test 4: CRS range — với D=0, R=0 → CRS ≈ 0.08 (không phải 0.5) ✓

QA CHECK:
  □ Header block + inline comments tại scale factor và EMA
  □ Scale × 5.0 đã implement
  □ EMA trên logit space (không phải sigmoid output)
  □ 4 tests pass
  □ Báo cáo: function thêm + lý do scale factor
```

---

## T2.6 — Hook CRS vào train.py (LOG ONLY)

**Tiêu chí pass:** 1500 iter không crash. TensorBoard có CRS metrics.

```
Task T2.6: Hook CRS vào train.py — CHỈ LOG.
KHÔNG đổi: loss, densify_and_prune(), compute_prune_mask().

BƯỚC 1 — Đọc train.py. Ghi line numbers:
  a. for iteration in range(...)
  b. densify_and_prune()
  c. tb_writer.add_scalar()
  Tóm tắt → chờ confirm.

BƯỚC 2 — Thêm vào train.py với comments:

from crs_module import (compute_depth_consistency,
                         compute_reprojection_consistency, update_crs)

# ── [CRSGaussian T2.6] T_warmup: không tính CRS trước mốc này ──
# Lý do: trước T_warmup Gaussians chưa settle, D_i và R_i là noise
# → CRS noisy → điều khiển loss/densify sẽ phản tác dụng
T_warmup = 1000   # Đã chuyển từ 2000, xem docs/04_decisions_log.md

# ── [CRSGaussian T2.6] CRS update block — LOG ONLY ──
# Thêm sau training step, trước densify_and_prune
if iteration > T_warmup and iteration % 100 == 0:
    with torch.no_grad():
        # Tính D_i và R_i per-Gaussian
        D = compute_depth_consistency(gaussians, viewpoint_stack,
            depth_prior_dict, render, pipe, background, depth_range)
        R = compute_reprojection_consistency(gaussians, viewpoint_stack,
            render, pipe, background)
        # Update _crs_score với EMA (scale × 5.0 bên trong update_crs)
        update_crs(gaussians, D, R, w1=0.5, w2=0.5, ema=0.9)

    # Log CRS distribution để validate signal (T2.7)
    # Kỳ vọng: floater → CRS thấp [0.08-0.3], surface → CRS cao [0.7-0.92]
    crs = gaussians.get_crs.detach()
    if tb_writer:
        tb_writer.add_scalar('crs/mean', crs.mean().item(), iteration)
        tb_writer.add_scalar('crs/std',  crs.std().item(),  iteration)
        tb_writer.add_scalar('crs/pct_below_0.3',
            (crs < 0.3).float().mean().item(), iteration)
        tb_writer.add_scalar('crs/pct_above_0.7',
            (crs > 0.7).float().mean().item(), iteration)
        tb_writer.add_histogram('crs/histogram', crs, iteration)
    if iteration % 500 == 0:
        print(f"[CRS {iteration}] mean={crs.mean():.3f} std={crs.std():.3f} "
              f"low={(crs<0.3).float().mean()*100:.1f}% "
              f"high={(crs>0.7).float().mean()*100:.1f}%")

QA CHECK:
  □ Header + inline comments tại: T_warmup lý do, update_crs call,
    log metrics, print statement
  □ T_warmup=1000
  □ KHÔNG sửa loss/densify/prune
  □ 1500 iter không crash
  □ Báo cáo: dòng nào thêm trong train.py
```

---

## T2.7 — Validate CRS signal (GO/NO-GO)

**Tiêu chí pass:** D.std>0.05, R.std>0.05, CRS range rộng hơn [0.5, 0.73].

```
Task T2.7: Validate CRS signal.
Nếu flat → debug trước, KHÔNG tiếp Phase 3.

BƯỚC 1 — Chạy đủ 2000 iter.

BƯỚC 2 — Thêm debug tạm thời tại iter 1500:
  if iteration == 1500:
      D_dbg  = compute_depth_consistency(...)
      R_dbg  = compute_reprojection_consistency(...)
      crs_dbg = gaussians.get_crs.detach()
      print("=== CRS VALIDATION ===")
      for name, v in [('D',D_dbg),('R',R_dbg),('CRS',crs_dbg)]:
          print(f"{name}: min={v.min():.3f} max={v.max():.3f} "
                f"mean={v.mean():.3f} std={v.std():.3f}")
      # Kiểm tra scale fix hoạt động đúng
      print(f"CRS min expected ≈ 0.08 (not 0.5): {'OK' if crs_dbg.min() < 0.3 else 'CHECK SCALE'}")
      if D_dbg.std()  < 0.05: print("WARNING: D_i collapsed!")
      if R_dbg.std()  < 0.05: print("WARNING: R_i collapsed!")
      if crs_dbg.max() < 0.6:  print("WARNING: CRS range too narrow — check scale factor 5.0")

BƯỚC 3 — Điền vào docs/03_task_queue.md:
  D.std=? | R.std=? | CRS.std=? | CRS.min=? | CRS.max=?
  Distribution: bimodal/unimodal/flat
  CRS mean tại 1000/1500/2000: ?/?/?

BƯỚC 4 — Kết luận:
  GO:    D.std>0.05 AND R.std>0.05 AND CRS.max>0.7 AND CRS.min<0.3
  NO-GO scenarios:
    CRS range [0.5, 0.73] → scale factor 5.0 chưa apply → check update_crs
    D flat → debug project_to_depth và depth_prior_dict
    R flat → debug warp_pixels geometry

CHECKPOINT Phase 2:
  % CRS<0.3: ?% | % CRS>0.7: ?%
  GO/NO-GO cho Phase 3?
```

---

---

# PHASE 3 — Adaptive Depth Loss

---

## T3.1 + T3.2 — Fixed và Adaptive depth loss

**Nhắc lại:** CoR-GS gốc KHÔNG có depth loss. Thêm mới hoàn toàn.

```
Task T3.1+T3.2: Thêm depth loss vào loss_utils.py.

BƯỚC 1 — Đọc:
  a. CoR-GS/utils/loss_utils.py — list functions hiện có
  b. LoopSparseGS/train.py — tìm DAR/Pearson
  c. train.py — chỗ LossDict sum và backward
  Báo cáo → chờ confirm.

BƯỚC 2 — Thêm vào loss_utils.py với headers + comments:

# ============================================================
# [CRSGaussian] Task: T3.1 — pearson_depth_loss
# File: CoR-GS/utils/loss_utils.py  (SỬA — thêm function mới)
# Mục đích: Depth loss scale-invariant dùng Pearson correlation.
#   Dùng Pearson thay L1/L2 vì D_aligned chỉ là relative depth
#   — scale/offset không hoàn toàn chính xác → Pearson robust hơn.
# Dùng trong: Stage 2b (T_densify→T_warmup) với fixed lambda
# ============================================================
def pearson_depth_loss(rendered_depth, depth_prior, patch_size=0):
    """1 - Pearson(rendered, prior). Range [0,2]. Scale-invariant."""
    if rendered_depth.dim()==3: rendered_depth=rendered_depth.squeeze(0)
    def _pearson(x, y):
        xf, yf = x.flatten(), y.flatten()
        xc, yc = xf - xf.mean(), yf - yf.mean()
        # Pearson = covariance / (std_x * std_y)
        return (xc*yc).sum() / (torch.sqrt((xc**2).sum() * (yc**2).sum()) + 1e-8)
    if patch_size == 0:
        # Global Pearson: toàn bộ ảnh
        return 1.0 - _pearson(rendered_depth, depth_prior)
    else:
        # Sliding window Pearson (từ DAR trong LoopSparseGS):
        # capture local depth structure tốt hơn global Pearson
        H, W = rendered_depth.shape
        losses = []
        for i in range(0, H-patch_size+1, patch_size):
            for j in range(0, W-patch_size+1, patch_size):
                losses.append(1.0 - _pearson(
                    rendered_depth[i:i+patch_size, j:j+patch_size],
                    depth_prior[i:i+patch_size, j:j+patch_size]))
        return torch.stack(losses).mean()

# ============================================================
# [CRSGaussian] Task: T3.2 — adaptive_depth_loss
# File: CoR-GS/utils/loss_utils.py  (SỬA — thêm function mới)
# Mục đích: Depth loss với lambda adaptive theo CRS per-Gaussian.
#   Floater (CRS thấp) → lambda lớn → bị kéo mạnh về depth prior
#   Surface (CRS cao) → lambda nhỏ → ít bị constraint
# Dùng trong: Sau T_warmup
# ============================================================
def adaptive_depth_loss(rendered_depth, depth_prior,
                         crs_pixel_weights, lambda_base=0.05):
    """lambda_i = lambda_base*(2-CRS_i). Weighted Pearson."""
    if rendered_depth.dim()==3: rendered_depth=rendered_depth.squeeze(0)
    # CRS thấp (floater) → lambda cao → depth loss mạnh hơn
    # CRS cao (surface) → lambda thấp → depth loss nhẹ hơn
    lambda_map = lambda_base * (2.0 - crs_pixel_weights.clamp(0, 1))
    def _pearson_w(x, y, w):
        xf, yf, wf = x.flatten(), y.flatten(), w.flatten()
        wn = wf / (wf.sum() + 1e-8)   # normalize weights
        xm, ym = (xf*wn).sum(), (yf*wn).sum()
        xc, yc = xf - xm, yf - ym
        return ((xc*yc*wn).sum() /
                (torch.sqrt((xc**2*wn).sum() * (yc**2*wn).sum()) + 1e-8))
    return 1.0 - _pearson_w(rendered_depth, depth_prior, lambda_map)

BƯỚC 3 — Unit tests:
  Test 1: identical → loss ≈ 0 ✓
  Test 2: anti-corr → loss ≈ 2 ✓
  Test 3: CRS=0 → lambda=2x ✓  CRS=1 → lambda=1x ✓

QA CHECK:
  □ Header blocks cho cả 2 functions
  □ Inline comments tại: Pearson lý do, sliding window lý do,
    lambda_map tính toán, weighted Pearson
  □ 3 tests pass
  □ Báo cáo: 2 functions thêm vào loss_utils.py (dòng nào)
```

---

## T3.3 — Tích hợp depth loss vào train.py + Ablation A3

**Tiêu chí pass:** 1200 iter không crash. A3 PSNR > A0b.

```
Task T3.3: Tích hợp depth loss vào train.py.

TIMING:
  T_densify(500) → T_warmup(1000): fixed Pearson
  Sau T_warmup(1000): adaptive (CRS-weighted)

BƯỚC 1 — Đọc train.py: chỗ LossDict sum, rendered depth có sẵn không?
  Báo cáo → chờ confirm.

BƯỚC 2 — Thêm sau loss_photometric với comments:

from utils.loss_utils import pearson_depth_loss, adaptive_depth_loss

rendered_depth = render_pkg.get('depth', None)
depth_prior    = depth_prior_dict.get(viewpoint_cam.uid, None)

if rendered_depth is not None and depth_prior is not None:
    if T_densify <= iteration < T_warmup:
        # ── [CRSGaussian T3.1] Fixed depth loss (warmup stage) ──
        # CRS chưa có → dùng fixed lambda cho tất cả Gaussians
        # Kéo tất cả Gaussians về gần depth prior trong warmup
        LossDict["loss_depth"] = 0.05 * pearson_depth_loss(
            rendered_depth, depth_prior)

    elif iteration >= T_warmup:
        # ── [CRSGaussian T3.2] Adaptive depth loss (post-warmup) ──
        # CRS đã có signal → lambda per-Gaussian theo CRS
        # Floater bị kéo mạnh hơn, surface được ép nhẹ hơn
        crs_px = project_crs_to_pixels(
            gaussians, viewpoint_cam, render, pipe, background)
        LossDict["loss_depth"] = adaptive_depth_loss(
            rendered_depth, depth_prior, crs_px, lambda_base=0.05)

ABLATION A3 (sau khi T3.3 ổn định):
  Config: D_i + R_i + adaptive loss, NO position constraint
  Chạy fern 10k iter → ghi PSNR vào ablation table
  ΔPSNR = A3 - A0b = ?  |  GO nếu ΔPSNR > 0

QA CHECK:
  □ Inline comments tại fixed loss và adaptive loss (lý do timing)
  □ Fixed/adaptive active đúng timing
  □ 1200 iter không crash
  □ A3 PSNR đã ghi
  □ Báo cáo: dòng thêm trong train.py
```

---

---

# PHASE 4 — Position Constraint + CRS Pruning

---

## T4.1 — Position constraint trong densification

**Tiêu chí pass:** 2 tests pass. BẬT từ T_densify=500.

```
Task T4.1: Thêm position constraint vào densify_and_split/clone().

TIMING: BẬT từ T_densify=500 (không phải T_warmup).
Lý do: AD-GS (2025) — floater hình thành ngay khi densify bắt đầu.
Position constraint không cần CRS → có thể BẬT sớm.

BƯỚC 1 — Đọc densify_and_split() và densify_and_clone().
  Báo cáo → chờ confirm.

BƯỚC 2 — Implement với comments:

def _check_position_depth(new_pos, depth_prior_dict, cameras, epsilon_depth):
    # ── [CRSGaussian T4.1] Kiểm tra position mới có gần depth prior ──
    # Gaussian mới phải nằm gần surface thật (theo depth prior)
    # Nếu không → skip để tránh tạo floater mới
    cam = cameras[0]
    if cam.uid not in depth_prior_dict: return True  # fallback: accept
    pos     = new_pos.unsqueeze(0)
    d_proj  = project_to_depth(pos, cam)[0]
    pix     = project_to_pixel(pos, cam)[0]
    H, W    = depth_prior_dict[cam.uid].shape
    px      = int(pix[0].clamp(0, W-1).item())
    py      = int(pix[1].clamp(0, H-1).item())
    d_prior = depth_prior_dict[cam.uid][py, px].item()
    # Chấp nhận nếu depth mới gần với depth prior
    return abs(d_proj - d_prior) < epsilon_depth

# Trong densify loop (thêm với comment):
# ── [CRSGaussian T4.1] Position constraint — BẬT từ T_densify=500 ──
# Thử 3 lần để tìm position hợp lệ. Nếu không được → skip.
# Không crash: graceful skip tốt hơn force tạo floater
epsilon_depth = 0.05 * depth_range
MAX_ATTEMPTS  = 3
for attempt in range(MAX_ATTEMPTS):
    candidate = sample_new_position(parent)
    if _check_position_depth(candidate, depth_prior_dict,
                              train_cams, epsilon_depth):
        accept(candidate); break
# Không có else → skip nếu không tìm được vị trí hợp lệ

BƯỚC 3 — Unit tests:
  Test 1: depth=5.0, prior=5.0, eps=0.5 → ACCEPT ✓
  Test 2: depth=9.0, prior=5.0, eps=0.5 → REJECT ✓

QA CHECK:
  □ Header block + inline comments tại: lý do BẬT sớm,
    fallback accept, graceful skip, epsilon_depth
  □ BẬT từ T_densify=500
  □ 2 tests pass
  □ Báo cáo: function thêm + dòng sửa trong gaussian_model.py
```

---

## T4.2 — CRS pruning condition

**Tiêu chí pass:** AND logic đúng. Chỉ sau T_warmup. 2 tests pass.

```
Task T4.2: Thêm CRS condition vào compute_prune_mask().

TIMING: Chỉ sau T_warmup=1000.

BƯỚC 1 — Paste compute_prune_mask() đầy đủ → chờ confirm.

BƯỚC 2 — Thêm với comments:

def compute_prune_mask(self, ..., iteration=0, T_warmup=1000, tau_crs=0.35):
    prune_mask = ...  # conditions hiện có — GIỮ NGUYÊN

    # ── [CRSGaussian T4.2] CRS-based pruning — chỉ sau T_warmup ──
    # Trước T_warmup: _crs_score=0 → CRS=0.5 (neutral) → không prune
    # AND logic: tránh prune Gaussian mới (opacity thấp nhưng CRS neutral)
    if iteration > T_warmup:
        crs_mask  = (self.get_crs < tau_crs)
        opac_mask = (self.get_opacity < min_opacity).squeeze()
        isol_mask = self._compute_isolation_mask(k=5)

        # AND: cả 3 phải xấu mới prune
        # CRS thấp thôi chưa đủ → có thể là Gaussian đang học
        # Opacity thấp thôi chưa đủ → có thể là Gaussian mới densified
        # Isolated thôi chưa đủ → có thể là Gaussian ở vùng thưa hợp lệ
        floater_mask = crs_mask & opac_mask & isol_mask
        prune_mask   = torch.logical_or(prune_mask, floater_mask)

    return prune_mask

def _compute_isolation_mask(self, k=5, mult=3.0):
    # ── [CRSGaussian T4.2] Phát hiện Gaussian bị isolated ──
    # Gaussian xa k nearest neighbors hơn mult×median → isolated
    # Dùng simple_knn có sẵn trong CoR-GS — không cần thêm dependency
    from simple_knn._C import distCUDA2
    knn_d = torch.sqrt(distCUDA2(self.get_xyz) + 1e-8)
    return knn_d > mult * torch.median(knn_d)

BƯỚC 3 — Unit tests:
  Test 1: CRS<0.35 & opac<0.005 & isolated → pruned ✓
  Test 2: CRS≈0.5 (neutral) & opac thấp → NOT pruned ✓

QA CHECK:
  □ Header + inline comments tại: timing lý do, AND logic lý do,
    từng condition lý do, isolation dùng simple_knn
  □ Chỉ sau T_warmup
  □ AND logic (không OR)
  □ 2 tests pass
  □ Báo cáo: dòng sửa trong gaussian_model.py
```

---

## T4.3 — Ablation A4 + A5 (Full CRSGaussian)

```
Task T4.3: Chạy A4 và A5. Điền ablation table.

A4: Full CRSGaussian — D_i + R_i + adaptive loss + pos constraint + CRS prune
    T_warmup=1000, T_densify=500, tau_crs=0.35, scale=5.0

A5: T_warmup=0 — CRS từ iter 0, không warmup

Điền bảng docs/03_task_queue.md:
  | Config              | PSNR | SSIM | LPIPS |
  | A0  (CoR-GS 2-field)|      |      |       |
  | A0b (1-field)       |      |      |       |
  | A3  (no pos)        |      |      |       |
  | A4  (Full CRS)      |      |      |       |
  | A5  (no warmup)     |      |      |       |

Phân tích 4 comparisons → ghi docs/04_decisions_log.md:
  1. A4 vs A0b: CRS improve không?
  2. A4 vs A3: Position constraint đóng góp?
  3. A4 vs A5: Warmup có cần thiết?
  4. A4 vs A0: Competitive với CoR-GS 2-field?

CHECKPOINT FINAL:
  A4 vs SOTA (NexusGS 21.47): gap = ?
  Contribution lớn nhất: D_i? R_i? Pos? Pruning?
    A4 > 21.0  → Phase 7 full experiments
    A4 < 20.5  → debug bottleneck từ ablation
    20.5–21.0  → tune hyperparameters

QA CHECK:
  □ A4 và A5 chạy đủ 10k iter
  □ 5 rows đã điền
  □ 4 comparisons đã phân tích
  □ Kết luận rõ ràng
```