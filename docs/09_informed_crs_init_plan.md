# Informed CRS₀ Initialization — Implementation Plan

> **Task:** T5.1-T5.8
> **Prerequisite:** Phase 2 DONE (CRS module), Phase 3 DONE (depth loss), T4.2 DONE (CRS pruning)
> **Dependency:** aligned_depth_dict + depth_range đã có sẵn từ T1.3
> **Triết lý:** CRS₀ có ý nghĩa hình học ngay từ đầu → densification iter 500-1000 không blind

---

## Tổng quan thay đổi

| File | Hành động | Mô tả |
|------|-----------|-------|
| `arguments/__init__.py` | SỬA | Thêm 11 arguments vào ModelParams |
| `utils/crs/crs_init.py` | TẠO MỚI | 4 functions tính CRS₀ |
| `scene/gaussian_model.py` | SỬA | create_from_pcd() + densification_postfix() |
| `train.py` | SỬA | Hook informed CRS₀ trước training loop |
| `tests/test_informed_crs_init.py` | TẠO MỚI | 6 unit tests |

---

## Bước 1 — Arguments (T5.1)

**File:** `arguments/__init__.py` → class `ModelParams`

Thêm sau block `self.dav2_encoder = "vitl"`:

```python
# ── [CRSGaussian T5] Informed CRS₀ Initialization ──
# Triết lý: dùng geometry info có sẵn sau alignment để set CRS₀
# có ý nghĩa ngay từ đầu, thay vì neutral 0.5.
# Master switch: False → giữ behavior cũ (CRS₀=0.5 tất cả)
self.informed_crs_init = False

# Component switches (chỉ active khi informed_crs_init=True)
# Mỗi component bật/tắt độc lập → ablation isolate contribution
self.crs_init_use_reproj = True   # q_reproj: SfM reprojection quality
self.crs_init_use_depth  = True   # q_depth: DAV2 depth agreement
self.crs_init_use_view   = True   # q_view: multi-view stereo support

# Weights — tự normalize khi component bị tắt
# Default equal weight 1/3 mỗi cái. Ablate sau khi có C1-C7.
self.crs_init_w_reproj = 0.333
self.crs_init_w_depth  = 0.333
self.crs_init_w_view   = 0.334

# Hyperparameters
self.crs_init_tau_r = 2.5    # reproj normalization. Ablate: {2.0, 2.5, 3.0}
self.crs_init_gamma = 5.0    # logit scale. Ablate: {3.0, 5.0, 8.0}
self.crs_init_eta   = 0.7    # densify inherit factor. Ablate: {0.5, 0.7, 0.9}

# Densify inherit switch (riêng biệt với informed_crs_init)
self.crs_densify_inherit = False  # True: child CRS = clip(η*parent, 0, 0.5)
```

**Lưu ý ParamGroup:** `bool` type dùng `action="store_true"` → `--informed_crs_init` = True khi có mặt.
Float type dùng `type=float` → `--crs_init_gamma 8.0` khi muốn đổi.

---

## Bước 2 — Core module (T5.2)

**File:** `utils/crs/crs_init.py` (TẠO MỚI)

### Function 1: `compute_reproj_quality(reproj_errors, tau_r=2.5)`
- Input: `(N,)` numpy — COLMAP reprojection errors
- Output: `(N,)` numpy — range [0, 1]
- Logic: `q = 1 - clip(error/tau_r, 0, 1)`
- Edge: error=0 → q=1.0 (perfect), error>=tau_r → q=0.0

### Function 2: `compute_depth_agreement(points_xyz, cameras, aligned_depth_dict, depth_range)`
- Input: COLMAP 3D points + cameras + aligned depth maps
- Output: `(N,)` numpy — range [0, 1]
- Logic: Project mỗi point lên từng camera → lấy d_COLMAP vs d_DAV2 tại pixel đó → average qua cameras visible
- Edge: Invisible point → q=0.5 (neutral). Point ngoài image → skip camera đó.
- **Implementation:**
  1. Với mỗi camera: build world2pixel matrix (W2C @ K)
  2. Project N points → (N, 2) pixel coords
  3. Filter in-frame AND depth > 0
  4. Lookup aligned_depth_dict[cam.uid] tại pixel coords
  5. d_COLMAP = z component sau W2C transform
  6. q_per_cam = 1 - clip(|d_DAV2 - d_COLMAP| / depth_range, 0, 1)
  7. Average qua cameras visible, invisible → 0.5

### Function 3: `compute_view_support(points_xyz, cameras, n_train)`
- Input: 3D points + cameras + total training view count
- Output: `(N,)` numpy — range [0, 1]
- Logic: Count bao nhiêu cameras thấy point (in-frame + depth > 0)
- Formula: `q = (n_obs - 1) / max(N_train - 1, 1)`
- Edge: n_obs=1 → q=0.0 (1 view = no stereo). n_obs=N_train → q=1.0.
- **Note:** Với LLFF 3-view: n_obs ∈ {1,2,3} → q ∈ {0.0, 0.5, 1.0}

### Function 4: `compute_informed_crs0(points_xyz, reproj_errors, cameras, aligned_depth_dict, depth_range, ...switches, weights, hyperparams...)`
- Orchestrator: gọi 3 functions trên, weighted average, scale logit
- Output: `(N,)` torch.Tensor — **logit space** (để lưu trực tiếp vào _crs_score)
- Logic:
  ```python
  components, weights_active = [], []
  if use_reproj: components.append(q_reproj); weights_active.append(w_reproj)
  if use_depth:  components.append(q_depth);  weights_active.append(w_depth)
  if use_view:   components.append(q_view);   weights_active.append(w_view)

  # Auto-normalize weights
  w_sum = sum(weights_active)
  weights_norm = [w / w_sum for w in weights_active]

  Q = sum(w * c for w, c in zip(weights_norm, components))
  logit = gamma * (Q - 0.5)
  return torch.from_numpy(logit).float()
  ```
- Edge: Tất cả components tắt → Q=0.5 → logit=0 → CRS=0.5 (neutral fallback)

---

## Bước 3 — gaussian_model.py (T5.3 + T5.4)

### T5.3: create_from_pcd() nhận informed_crs0

Thêm optional param `informed_crs0: torch.Tensor = None`:

```python
def create_from_pcd(self, pcd, spatial_lr_scale, informed_crs0=None):
    ...
    # ── [CRSGaussian T5.3] Informed CRS₀ ──
    if informed_crs0 is not None:
        # informed_crs0: (N,) logit từ compute_informed_crs0()
        self._crs_score = informed_crs0.unsqueeze(-1).to("cuda")
    else:
        # Behavior cũ: neutral CRS = 0.5
        self._crs_score = torch.zeros((N, 1), device="cuda")
```

### T5.4: densification_postfix() conservative inherit

Thêm optional param `eta: float = 0.0`:

```python
def densification_postfix(self, ..., eta=0.0):
    ...
    # ── [CRSGaussian T5.4] CRS cho Gaussians mới ──
    if eta > 0:
        # Conservative inherit: child CRS capped tại 0.5
        # Lý do: child ở vị trí khác parent → chưa proven quality
        parent_crs = torch.sigmoid(self._crs_score).squeeze(-1)  # (N_old,)
        # Chỉ lấy CRS của parents tương ứng với new Gaussians
        # new_xyz.shape[0] = số children mới
        # parent indices cần được truyền vào hoặc lấy từ cuối self._crs_score
        inherited = (parent_crs[-new_xyz.shape[0]:] * eta).clamp(1e-6, 0.5)
        child_logit = torch.log(inherited / (1 - inherited))  # inverse sigmoid
        new_crs = child_logit.unsqueeze(-1)
    else:
        new_crs = torch.zeros((new_xyz.shape[0], 1), device="cuda")

    self._crs_score = torch.cat([self._crs_score, new_crs], dim=0)
```

**Lưu ý quan trọng:** `densification_postfix()` được gọi từ `densify_and_clone()` và `densify_and_split()`.
Cần verify thứ tự: parent CRS nằm ở đâu trong self._crs_score tại thời điểm gọi.
Cách an toàn: truyền parent_crs_scores explicitly thay vì dùng index cuối.

---

## Bước 4 — train.py (T5.5)

Thêm sau block `align_depth_to_colmap(...)`, **trước** `Scene()` init:

```python
# ── [CRSGaussian T5.5] Informed CRS₀ initialization ──
informed_crs0 = None
if dataset.informed_crs_init and dataset.use_depth_prior:
    from utils.crs.crs_init import compute_informed_crs0
    # Cần: scene.point_cloud — nhưng Scene chưa init!
    # → Đọc COLMAP data trực tiếp, hoặc di chuyển block này
    #   sau Scene() init nhưng trước training loop.
    # OPTION: Gọi sau scene = Scene(...), lấy scene.init_point_cloud
    pass
```

**ISSUE:** `scene.init_point_cloud` chỉ có sau `Scene()` init. Nhưng `create_from_pcd()` được gọi
bên trong `Scene.__init__()`. Cần kiểm tra flow:

```
Scene.__init__()
  → readColmapSceneInfo() → lấy pcd
  → gaussians.create_from_pcd(pcd, ...)  ← CRS₀ cần ở đây!
```

**Giải pháp:** 2 options:
1. **Option A:** Sửa `Scene.__init__()` truyền `informed_crs0` xuống `create_from_pcd()`
   → Cần tính informed_crs0 TRƯỚC Scene(), tức là đọc COLMAP data 2 lần.
2. **Option B:** Gọi `compute_informed_crs0()` bên trong `Scene.__init__()` sau khi có pcd
   → Truyền dataset args vào Scene.
3. **Option C (đơn giản nhất):** Sau `scene = Scene(...)`, ghi đè `gaussians._crs_score`
   bằng informed logit. create_from_pcd() vẫn init neutral, rồi overwrite ngay sau.
   → Không cần sửa Scene hay đọc COLMAP 2 lần.

**Chọn Option C** — đơn giản nhất, không sửa Scene, backward-compatible:

```python
scene = Scene(args, gaussians, shuffle=False)
# ... (aligned_depth_dict, depth_range đã có ở trên)

# ── [CRSGaussian T5.5] Informed CRS₀ ──
if dataset.informed_crs_init and dataset.use_depth_prior:
    from utils.crs.crs_init import compute_informed_crs0
    informed_crs0 = compute_informed_crs0(
        points_xyz    = scene.init_point_cloud.points,
        reproj_errors = ...,  # cần verify source — scene hay COLMAP bin
        cameras       = allCameras,
        aligned_depth_dict = aligned_depth_dict,
        depth_range   = depth_range,
        use_reproj    = dataset.crs_init_use_reproj,
        use_depth     = dataset.crs_init_use_depth,
        use_view      = dataset.crs_init_use_view,
        w_reproj      = dataset.crs_init_w_reproj,
        w_depth       = dataset.crs_init_w_depth,
        w_view        = dataset.crs_init_w_view,
        tau_r         = dataset.crs_init_tau_r,
        gamma         = dataset.crs_init_gamma,
    )
    # Overwrite neutral CRS₀ với informed logit
    gaussians._crs_score = informed_crs0.unsqueeze(-1).cuda()

    # Log distribution để validate γ
    Q = torch.sigmoid(informed_crs0)
    print(f"[CRS] CRS₀ distribution: min={Q.min():.3f} max={Q.max():.3f} "
          f"mean={Q.mean():.3f} std={Q.std():.3f}")
    print(f"[CRS] CRS₀ < 0.35: {(Q<0.35).sum()} ({(Q<0.35).float().mean()*100:.1f}%)")
    print(f"[CRS] CRS₀ > 0.65: {(Q>0.65).sum()} ({(Q>0.65).float().mean()*100:.1f}%)")
```

**Cần verify:** `scene.init_point_cloud` có chứa reproj_errors không?
Nếu không → đọc từ COLMAP bin như depth_alignment.py đã làm.

---

## Bước 5 — train.py densification (T5.5 cont.)

Truyền `eta` vào densification call khi `--crs_densify_inherit`:

```python
_eta = dataset.crs_init_eta if dataset.crs_densify_inherit else 0.0

for i in range(args.gaussiansN):
    GsDict[f"gs{i}"].densify_and_prune(
        ...,
        eta=_eta,   # ← truyền xuống densify_and_split/clone → densification_postfix
    )
```

**Cần verify:** `densify_and_prune()` → `densify_and_split()` → `densification_postfix()`.
Eta cần chain qua tất cả.

---

## Bước 6 — Unit tests (T5.6)

**File:** `tests/test_informed_crs_init.py`

| Test | Input | Expected |
|------|-------|----------|
| 1. q_reproj | error={0, 1.25, 2.5, 5.0} | q={1.0, 0.5, 0.0, 0.0} |
| 2. q_view | n_obs={1,2,3}, N_train=3 | q={0.0, 0.5, 1.0} |
| 3. full pipeline | Synthetic points + cameras | Shape (N,), dtype float32, range ~ [-2.5, 2.5] logit |
| 4. component switches | use_reproj=F, use_depth=F, use_view=T | Chỉ q_view, weights normalize to 1.0 |
| 5. baseline behavior | informed_crs_init=False | _crs_score = zeros → CRS=0.5 |
| 6. densify inherit | η=0.7, parent={0.8, 0.4} | child={0.5, 0.28} (capped) |

---

## Bước 7 — Verify + Log (T5.7 + T5.8)

**Step 1 — Unit test trên server:**
```bash
conda activate corgs && cd CRSGaussian && \
python tests/test_informed_crs_init.py
```

**Step 2 — Verify baseline không break:**
```bash
python train.py \
  --source_path data/nerf_llff_data/fern \
  -m output/test_baseline_check \
  --eval -r 8 --n_views 3 --iterations 100 \
  --gaussiansN 1 --use_depth_prior \
  --dav2_path ../Depth-Anything-V2
  # KHÔNG có --informed_crs_init → behavior cũ
```

**Step 3 — Log Q distribution:**
```bash
python train.py \
  --source_path data/nerf_llff_data/fern \
  -m output/test_informed_crs0 \
  --eval -r 8 --n_views 3 --iterations 100 \
  --gaussiansN 1 --use_depth_prior \
  --dav2_path ../Depth-Anything-V2 \
  --informed_crs_init \
  2>&1 | grep "\[CRS\]"
```

→ Paste output `[CRS]` lines → validate γ=5.0 → confirm → chạy ablation C0-C8.

---

## Ablation configs

| Config | Command-line flags |
|--------|--------------------|
| C0 | (default — no `--informed_crs_init`) |
| C1 | `--informed_crs_init --crs_init_use_depth False --crs_init_use_view False` |
| C2 | `--informed_crs_init --crs_init_use_reproj False --crs_init_use_view False` |
| C3 | `--informed_crs_init --crs_init_use_reproj False --crs_init_use_depth False` |
| C4 | `--informed_crs_init --crs_init_use_view False` |
| C5 | `--informed_crs_init --crs_init_use_depth False` |
| C6 | `--informed_crs_init --crs_init_use_reproj False` |
| C7 | `--informed_crs_init` |
| C8 | `--informed_crs_init --crs_densify_inherit` |

**Lưu ý:** `bool` args dùng `action="store_true"` trong ParamGroup.
Để TẮT component (default=True), cần truyền `False` explicit:
`--crs_init_use_depth False` → ParamGroup sẽ parse là string "False".

**ISSUE:** ParamGroup dùng `action="store_true"` cho bool → không thể truyền False!
**FIX cần thiết:** Với các bool default=True, KHÔNG dùng store_true.
Thay bằng: dùng int (0/1) hoặc custom bool parser. Xem Bước 1 implementation.

---

## Thứ tự implement (quan trọng)

```
T5.1  arguments/__init__.py     ← trước, để các file khác reference
T5.2  utils/crs/crs_init.py     ← core logic, test independently
T5.6  tests/...                  ← viết test ngay sau core module
T5.3  gaussian_model.py          ← create_from_pcd sửa nhỏ
T5.4  gaussian_model.py          ← densification_postfix sửa
T5.5  train.py                   ← hook tất cả lại
T5.7  Server: verify baseline
T5.8  Server: log Q distribution
```
