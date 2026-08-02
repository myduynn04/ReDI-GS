# ============================================================
# [CRSGaussian] Task: T2.3 — compute_depth_consistency
# File: CRSGaussian/utils/crs/crs_module.py  (TẠO MỚI)
# Mục đích: Tính D_i per-Gaussian — đo mỗi Gaussian có đứng
#           đúng độ sâu so với aligned depth prior không.
#           D_i = 1 - |d_proj_i - d_prior_i| / depth_range
# Được gọi từ: update_crs() (T2.5), mỗi 100 iter sau T_warmup
# ============================================================

import torch
from utils.graphics_utils import fov2focal


@torch.no_grad()
def compute_depth_consistency(
    xyz: torch.Tensor,
    cameras: list,
    aligned_depth_dict: dict,
    depth_range: float,
) -> torch.Tensor:
    """Tính depth consistency D_i cho mỗi Gaussian trên tất cả cameras.

    Với mỗi camera, project 3D position của Gaussian xuống image plane,
    so depth projected với aligned depth prior tại pixel đó.
    D_i cuối = trung bình D_i trên các cameras mà Gaussian visible.

    Args:
        xyz: (N, 3) Gaussian positions trên GPU.
             Lấy từ gaussians.get_xyz (nn.Parameter).
        cameras: list of Camera objects (có .uid, .world_view_transform,
                 .FoVx, .FoVy, .image_width, .image_height).
        aligned_depth_dict: {cam.uid: Tensor (H, W) float32 trên CPU}.
             Aligned metric depth từ Phase 1 (DAV2 + WLS alignment).
        depth_range: float — median(far) - median(near) across cameras.
             Dùng để normalize depth error về [0, 1].

    Returns:
        D: (N, 1) tensor float32 trên GPU, range [0, 1].
           D_i ≈ 1.0: Gaussian nằm đúng vị trí depth prior (surface).
           D_i ≈ 0.0: Gaussian cách xa depth prior (floater).
           D_i = 0.5: Gaussian không visible từ bất kỳ camera nào (neutral).
    """
    N = xyz.shape[0]
    device = xyz.device

    # Tích lũy D_i qua các cameras rồi average
    D_sum = torch.zeros(N, 1, device=device)
    count = torch.zeros(N, 1, device=device)

    # Homogeneous coords (N, 4) — tính một lần, reuse cho mọi camera.
    # .detach() vì D_i là diagnostic signal, không cần gradient ngược
    # về positions.
    ones = torch.ones(N, 1, device=device, dtype=xyz.dtype)
    xyz_hom = torch.cat([xyz.detach(), ones], dim=1)  # (N, 4)

    # Clamp depth_range tối thiểu để tránh chia cho ~0
    depth_range_safe = max(depth_range, 1e-6)

    for cam in cameras:
        if cam.uid not in aligned_depth_dict:
            continue

        H = cam.image_height
        W = cam.image_width

        # ── 3D Projection: World → Camera → Pixel ──
        # world_view_transform lưu column-major (transposed) trên cuda.
        # .T chuyển về row-major: W2C (4x4) chuẩn.
        # KHÔNG dùng R.T @ P + T trực tiếp — vì getWorld2View2 có
        # translate+scale adjustment mà R/T riêng lẻ không capture.
        W2C = cam.world_view_transform.T  # (4, 4) GPU, row-major

        # World → Camera coords: (4, 4) @ (4, N) → (4, N) → (N, 4)
        pts_cam = (W2C @ xyz_hom.T).T  # (N, 4)
        depth = pts_cam[:, 2]          # (N,) — depth in camera space

        # Camera intrinsics từ Field of View
        fx = fov2focal(cam.FoVx, W)
        fy = fov2focal(cam.FoVy, H)
        cx, cy = W / 2.0, H / 2.0

        # Project → pixel coordinates
        pixel_x = pts_cam[:, 0] / depth * fx + cx
        pixel_y = pts_cam[:, 1] / depth * fy + cy

        # ── Visibility filter (projection-based) ──
        # Gaussian "visible" = depth > 0 (trước camera) VÀ project vào
        # trong image bounds. Tương đương frustum culling của rasterizer,
        # trừ zero-radius check (degenerate Gaussians — hiếm).
        # Gaussian không qua filter → giữ neutral, không bị penalize
        # vì depth so sánh không có ý nghĩa khi bị occlude/ngoài frame.
        valid = (
            (depth > 0)
            & (pixel_x >= 0) & (pixel_x < W)
            & (pixel_y >= 0) & (pixel_y < H)
        )

        n_valid = valid.sum().item()
        if n_valid == 0:
            continue

        # Pixel indices để lookup depth prior (clamp cho safety dù
        # valid mask đã đảm bảo in-bounds)
        px = pixel_x[valid].long().clamp(0, W - 1)
        py = pixel_y[valid].long().clamp(0, H - 1)

        # ── Depth prior lookup ──
        # aligned_depth_dict lưu GPU tensors (moved tại depth_alignment.py).
        depth_prior_map = aligned_depth_dict[cam.uid]  # (H, W) GPU
        d_prior = depth_prior_map[py, px]  # (M,)

        # Depth projected của Gaussians visible
        d_proj = depth[valid]  # (M,)

        # ── D_i = 1 - |d_proj - d_prior| / depth_range ──
        # Clamp [0, 1]: error > depth_range → D_i = 0 (hoàn toàn sai).
        # Lý do dùng depth_range normalize: đưa error về scale-invariant,
        # cho phép cùng ngưỡng tau_crs hoạt động trên mọi scene.
        D_i = 1.0 - torch.abs(d_proj - d_prior) / depth_range_safe
        D_i = D_i.clamp(0.0, 1.0)  # (M,)

        # Tích lũy vào buffer
        D_sum[valid, 0] += D_i
        count[valid, 0] += 1.0

    # ── Average D_i trên các cameras visible ──
    visible_any = (count > 0).squeeze(1)  # (N,)

    # Gaussian không visible từ bất kỳ camera nào → neutral 0.5.
    # Không update vì không có thông tin để đánh giá — CRS sẽ không
    # phạt cũng không thưởng Gaussian này.
    D = torch.full((N, 1), 0.5, device=device)
    D[visible_any] = D_sum[visible_any] / count[visible_any]

    return D


# ============================================================
# [CRSGaussian] Task: T2.4 — compute_reprojection_consistency
# File: CRSGaussian/utils/crs/crs_module.py  (THÊM VÀO)
# Mục đích: Tính R_i per-Gaussian — đo color consistency khi
#           project Gaussian xuống nhiều cameras.
#           Surface → cùng scene point → colors giống → R cao.
#           Floater → khác scene point → colors khác → R thấp.
# Hướng 1/3 options — xem docs/08_R_i_options.md
# Được gọi từ: update_crs() (T2.5), mỗi 100 iter sau T_warmup
# ============================================================


@torch.no_grad()
def compute_reprojection_consistency(
    xyz: torch.Tensor,
    cameras: list,
    min_visible_views: int = 2,
) -> torch.Tensor:
    """Tính reprojection consistency R_i cho mỗi Gaussian.

    Project Gaussian 3D position xuống từng camera, lookup GT image color
    tại pixel đó, so sánh pairwise L1 giữa các cameras.
    Surface Gaussian → project ra cùng scene point → GT colors giống → R ≈ 1.
    Floater → project ra vị trí scene khác nhau → GT colors khác → R thấp.

    Dùng GT image color (cam.original_image), KHÔNG dùng render()["color"]
    (SH per-Gaussian). Lý do: SH đã optimize fit training views → floater
    cũng có SH color ổn định → không phân biệt được floater.
    GT color phản ánh scene thật tại pixel projected.


    Args:
        xyz: (N, 3) Gaussian positions trên GPU.
        cameras: list of Camera objects. Cần có .original_image (3,H,W)
                 float [0,1] trên cuda. PseudoCamera không có → bị skip.
        min_visible_views: tối thiểu N views Gaussian phải visible để compute R.
            < min → R = 0.5 (neutral, không đủ data). Default 2.

    Returns:
        R: (N, 1) tensor float32 trên GPU, range [0, 1].
           R ≈ 1.0: color consistent cross-view (surface).
           R ≈ 0.0: color inconsistent (floater).
           R = 0.5: visible < min_visible_views cameras (neutral).
    """
    N = xyz.shape[0]
    device = xyz.device

    # Lọc cameras có GT image (skip PseudoCamera)
    valid_cams = [c for c in cameras if hasattr(c, 'original_image')]
    K = len(valid_cams)

    if K < min_visible_views:
        # Không đủ camera để có ≥ min_visible_views → tất cả neutral
        return torch.full((N, 1), 0.5, device=device)

    # ── Collect GT colors tại projected pixel cho mỗi camera ──
    # Sentinel -1 đánh dấu invalid (Gaussian không visible từ camera đó).
    colors = torch.full((N, K, 3), -1.0, device=device)

    # Homogeneous coords — tính 1 lần, reuse cho mọi camera.
    # .detach() vì R_i là diagnostic signal, không cần gradient.
    ones = torch.ones(N, 1, device=device, dtype=xyz.dtype)
    xyz_hom = torch.cat([xyz.detach(), ones], dim=1)  # (N, 4)

    for k, cam in enumerate(valid_cams):
        H = cam.image_height
        W = cam.image_width

        # ── Projection logic (cùng pattern với D_i) ──
        # world_view_transform lưu column-major → .T về row-major W2C.
        W2C = cam.world_view_transform.T  # (4, 4) GPU
        pts_cam = (W2C @ xyz_hom.T).T     # (N, 4)
        depth = pts_cam[:, 2]              # (N,) — gauss_z trong camera space

        fx = fov2focal(cam.FoVx, W)
        fy = fov2focal(cam.FoVy, H)
        cx, cy = W / 2.0, H / 2.0

        pixel_x = pts_cam[:, 0] / depth * fx + cx
        pixel_y = pts_cam[:, 1] / depth * fy + cy

        # Visibility: depth > 0 và trong image bounds
        valid = (
            (depth > 0)
            & (pixel_x >= 0) & (pixel_x < W)
            & (pixel_y >= 0) & (pixel_y < H)
        )

        if valid.sum() == 0:
            continue


        px = pixel_x[valid].long().clamp(0, W - 1)
        py = pixel_y[valid].long().clamp(0, H - 1)

        # GT image: (3, H, W) float [0,1] on cuda
        # [:, py, px] → (3, M), .T → (M, 3)
        gt_colors = cam.original_image[:, py, px].T  # (M, 3)
        colors[valid, k, :] = gt_colors

    # ── Pairwise L1 diff giữa tất cả cặp cameras ──
    # Với K=3 (LLFF sparse-view) → C(3,2) = 3 pairs.
    # valid_mask: camera k có color hợp lệ cho Gaussian n.
    valid_mask = (colors[:, :, 0] >= 0)  # (N, K)

    R_sum = torch.zeros(N, 1, device=device)
    pair_count = torch.zeros(N, 1, device=device)
    visible_count = valid_mask.sum(dim=1)  # (N,) — số camera visible per Gaussian

    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            # Cả hai cameras phải visible cho Gaussian này
            both_valid = valid_mask[:, k1] & valid_mask[:, k2]  # (N,)
            if both_valid.sum() == 0:
                continue

            # L1 diff trung bình qua 3 kênh RGB → scalar per-Gaussian.
            # Image [0,1] → max diff = 1.0, tương đương /255 khi [0,255].
            diff = torch.abs(
                colors[both_valid, k1] - colors[both_valid, k2]
            ).mean(dim=1)  # (M,)

            R_sum[both_valid, 0] += diff
            pair_count[both_valid, 0] += 1.0

    # ── R_i = 1 - mean_pairwise_diff ──
    # [Phase 8a] Enforce min_visible_views: Gaussian phải visible từ ≥ min views.
    enough_views = visible_count >= min_visible_views  # (N,) bool
    has_pairs = (pair_count > 0).squeeze(1) & enough_views

    # Gaussian không đủ visible views → neutral 0.5 (insufficient data).
    R = torch.full((N, 1), 0.5, device=device)
    R[has_pairs] = 1.0 - R_sum[has_pairs] / pair_count[has_pairs]
    R = R.clamp(0.0, 1.0)

    return R


# ============================================================
# [CRSGaussian] Task: T2.5 — update_crs
# File: CRSGaussian/utils/crs/crs_module.py  (THÊM VÀO)
# Mục đích: Gom D_i + R_i → CRS logit, ghi vào gaussians._crs_score
#           bằng EMA. Hàm entry-point duy nhất gọi từ train.py.
# Được gọi từ: train.py, mỗi 100 iter sau T_warmup
# ============================================================


# ============================================================
# [CRSGaussian Hướng D MVP] Task: compute_render_contribution
# File: utils/crs/crs_module.py
# Mục đích: RC_i = render contribution per-Gaussian (proxy version).
#   RC = opacity × max(screen-space radii cross views) × view coverage
# Lưu ý proxy: KHÔNG capture true α·T accumulation (cần CUDA fork).
# Phase A MVP dùng proxy này để test core hypothesis (L2 EMA + L3 α-couple).
# Được gọi từ: train.py qua compute_crs_rnrc() khi use_rnrc=True
# ============================================================


# ============================================================
# [CRSGaussian Hướng D MVP] Task: compute_crs_rnrc
# File: utils/crs/crs_module.py
# Mục đích: Layer 2 (continuous EMA) + warmup logic cho RNRC.
#   spawn_iter: tensor [N] — iter mỗi Gaussian được tạo (init/clone/split).
#   - Iter < rnrc_warmup → CRS = 1 (bypass, scene chưa converge)
#   - Per-Gaussian age < rnrc_per_gauss_warmup → CRS floor = rnrc_floor
# Update gaussians._rc_smooth (running EMA của RC) như instance attr.
# Được gọi từ: train.py mỗi iter khi use_rnrc=True
# ============================================================


@torch.no_grad()
def compute_crs_rnrc(gaussians, rc_thisstep, opt, iter, spawn_iter=None):
    """Compute CRS từ RC với continuous EMA + warmup.

    Args:
        gaussians: GaussianModel (cần ._rc_smooth attr — auto-init nếu chưa có).
        rc_thisstep: (N,) tensor RC tính từ render hiện tại.
        opt: OptimizationParams — cần rnrc_beta, rnrc_warmup, rnrc_floor,
             rnrc_per_gauss_warmup, rnrc_norm_mode.
        iter: int — iteration hiện tại.
        spawn_iter: (N,) tensor int — iter mỗi Gaussian được tạo. None → no per-gauss warmup.

    Returns:
        crs: (N,) tensor float32 GPU, range (0, 1).
    """
    device = rc_thisstep.device
    N = rc_thisstep.shape[0]

    # ── Layer 2: Continuous EMA ──
    # Auto-init / re-init khi N thay đổi (densify/prune giữa updates).
    if (not hasattr(gaussians, "_rc_smooth")) or gaussians._rc_smooth.shape[0] != N:
        gaussians._rc_smooth = rc_thisstep.clone()
    else:
        gaussians._rc_smooth = (opt.rnrc_beta * gaussians._rc_smooth
                                + (1.0 - opt.rnrc_beta) * rc_thisstep)

    rc_smooth = gaussians._rc_smooth
    rc_pos = rc_smooth[rc_smooth > 0]
    if rc_pos.numel() == 0:
        return torch.full((N,), 0.5, device=device)

    # Normalize
    if opt.rnrc_norm_mode == "p75":
        denom = torch.quantile(rc_pos, 0.75) + 1e-6
    else:  # "median"
        denom = rc_pos.median() + 1e-6
    rc_norm = (rc_smooth / denom).clamp(0, 2) / 2.0   # ∈ [0, 1]

    # Sigmoid với scale=5 (giống CRS gốc) → range [0.08, 0.92]
    crs = torch.sigmoid(5.0 * (rc_norm - 0.5))

    # ── Global warmup: first rnrc_warmup iters bypass ──
    if iter < opt.rnrc_warmup:
        return torch.ones_like(crs)

    # ── Per-Gaussian warmup: young Gaussians có CRS floor ──
    if spawn_iter is not None and spawn_iter.shape[0] == N:
        age = iter - spawn_iter.float()
        young_mask = age < opt.rnrc_per_gauss_warmup
        floor_t = torch.full_like(crs, opt.rnrc_floor)
        crs = torch.where(young_mask, torch.maximum(crs, floor_t), crs)

    return crs


@torch.no_grad()
def compute_render_contribution(gaussians, cameras, render_func, render_args):
    """RC_i = opacity × normalized(max_radii) × (n_visible / n_views).

    Args:
        gaussians: GaussianModel (cần .get_xyz, .get_opacity).
        cameras: list training Camera objects (có .original_image attribute).
        render_func: hàm render đã import (gaussian_renderer.render).
        render_args: tuple positional args ngoài (cam, gaussians) — thường (pipe, background).

    Returns:
        RC: (N,) tensor float32 GPU. Range [0, +∞).
            Invisible-everywhere (radii=0 ở mọi view) → 0.
            Surface visible nhiều view, opacity cao → giá trị lớn.
    """
    N = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device
    max_radii = torch.zeros(N, device=device)
    visible_count = torch.zeros(N, device=device)

    # Lọc training cams (skip pseudo cams)
    train_cams = [c for c in cameras if hasattr(c, 'original_image')]
    n_views = max(len(train_cams), 1)

    for cam in train_cams:
        pkg = render_func(cam, gaussians, *render_args)
        # radii: (N,) int32 từ rasterizer, 0 nếu không visible
        radii = pkg["radii"].float()
        # visibility_filter: (N,) bool — Gaussian passed culling
        vis = pkg.get("visibility_filter")
        if vis is None:
            vis = radii > 0
        # Pad nếu shape mismatch (xảy ra khi N tăng giữa renders, hiếm)
        if radii.shape[0] < N:
            pad = N - radii.shape[0]
            radii = torch.cat([radii, torch.zeros(pad, device=device)])
            vis = torch.cat([vis, torch.zeros(pad, dtype=torch.bool, device=device)])
        elif radii.shape[0] > N:
            radii = radii[:N]
            vis = vis[:N]
        # Take max screen radii cross views (Gaussian "important" view nhất)
        max_radii = torch.maximum(max_radii, radii)
        visible_count += vis.float()

    opacity = gaussians.get_opacity.squeeze()  # (N,)
    coverage = visible_count / n_views          # (N,) [0, 1]
    RC = opacity * max_radii * coverage         # (N,) [0, +∞)
    return RC


@torch.no_grad()
def update_crs(
    gaussians,
    cameras: list,
    aligned_depth_dict: dict,
    depth_range: float,
    w1: float = 0.5,
    w2: float = 0.5,
    scale: float = 5.0,
    ema: float = 0.9,
    # ── [CRSGaussian Tier 2-min] D_cycle support ──
    use_d_cycle: bool = False,
    iter: int = 0,
    d_cycle_warmup: int = 1000,
    d_cycle_sigma: float = 5.0,
    d_cycle_update_freq: int = 100,
    render_func=None,
    pipe=None,
    bg=None,
    # ── [CRSGaussian Phase 8b] S_stability support ──
    use_sh_reliability: bool = False,
    sh_stability_warmup: int = 1000,
    sh_stability_ema_beta: float = 0.95,
    crs_w_s: float = 0.33,
    # ── [CRSGaussian Phase 9] D-only formula support ──
    disable_r_signal: bool = False,
) -> None:
    """Tính D_i, R_i rồi update gaussians._crs_score in-place bằng EMA.

    Công thức:
        score      = w1 * D_i + w2 * R_i
        crs_logit  = scale * (score - 0.5)
        _crs_score = ema * old_logit + (1 - ema) * crs_logit

    _crs_score lưu ở logit space. gaussians.get_crs property đã có
    sigmoid() → CRS ∈ [0, 1].

    EMA trên logit space (không phải sigmoid output):
    - Logit space tuyến tính → EMA ổn định, không bị nén ở 2 đầu.
    - _crs_score init = 0 (logit) → sigmoid = 0.5 (neutral).
    - Sau vài lần update, EMA hội tụ về giá trị thực tế.

    Args:
        gaussians: GaussianModel — cần .get_xyz (N,3) và ._crs_score (N,1).
        cameras: list Camera objects.
        aligned_depth_dict: {cam.uid: Tensor (H,W)} CPU — từ Phase 1.
        depth_range: float — normalization cho D_i.
        w1: weight cho D_i. Default 0.5.
        w2: weight cho R_i. Default 0.5.
        scale: scale factor trước sigmoid. Default 5.0.
            Mở rộng CRS range: logit ∈ [-scale/2, scale/2].
        ema: EMA decay. Default 0.9.
            Cao hơn → smooth hơn, phản ứng chậm hơn.
        use_d_cycle: [Tier 2-min] Replace D_DAV2 với D_cycle (cycle-depth
            consistency, no DAV2 dependency). Default False (backward compat).
        iter: current training iteration. Cần cho d_cycle_warmup gating
            + cache invalidation theo d_cycle_update_freq.
        d_cycle_warmup: trước iter này dùng D_DAV2 (scene chưa converge,
            rendered depth chưa stable). Default 1000.
        d_cycle_sigma: cycle error normalization (pixels) cho exp(-err/σ).
        d_cycle_update_freq: mỗi N iter mới recompute D_cycle (cache giữa).
            Lý do: D_cycle render N_cams depth maps → cost cao, cache giảm
            overhead xuống ~1/N của brute-force compute mỗi iter.
        render_func, pipe, bg: cần khi use_d_cycle=True để render depth maps.
            None khi flag OFF — không ảnh hưởng baseline.
    """
    xyz = gaussians.get_xyz  # (N, 3) GPU

    # ── [CRSGaussian Phase 9] R compute gating ──
    # Khi disable_r_signal=True → skip R entirely (formula D-only hoặc D+S).
    # Tiết kiệm compute (no R aggregation).
    need_r_compute = not disable_r_signal

    # ── [CRSGaussian Phase 8] Pre-render depth maps (D_cycle only sau Phase 24 cleanup) ──
    shared_depth_maps = None
    need_depth_render = (
        use_d_cycle and iter >= d_cycle_warmup
    ) and render_func is not None
    if need_depth_render:
        shared_depth_maps = {}
        for cam in cameras:
            if not hasattr(cam, 'original_image'):
                continue
            try:
                pkg = render_func(cam, gaussians, pipe, bg, disable_dropout=True)
            except TypeError:
                pkg = render_func(cam, gaussians, pipe, bg)
            shared_depth_maps[cam.uid] = pkg["depth"].detach()  # (1, H, W)

    # ── [CRSGaussian Tier 2-min] D signal: D_DAV2 vs D_cycle ──
    # Switch logic:
    #   use_d_cycle=False (default) → D_DAV2 (behavior cũ, backward compat).
    #   use_d_cycle=True  AND iter < d_cycle_warmup → D_DAV2 (scene chưa converge,
    #                                                  D_cycle nhiễu).
    #   use_d_cycle=True  AND iter ≥ d_cycle_warmup → D_cycle (cached mỗi
    #                                                  d_cycle_update_freq iter).
    if use_d_cycle and iter >= d_cycle_warmup and render_func is not None:
        # Cache D_cycle qua attr gaussians._d_cycle_cache để giảm cost.
        # Re-compute khi: (a) chưa có cache, (b) shape mismatch (densify/prune),
        # (c) update_freq period (vd mỗi 100 iter trùng với CRS update interval).
        N = xyz.shape[0]
        need_recompute = (
            not hasattr(gaussians, "_d_cycle_cache")
            or gaussians._d_cycle_cache is None
            or gaussians._d_cycle_cache.shape[0] != N
            or (iter % d_cycle_update_freq == 0)
        )
        if need_recompute:
            from utils.crs.d_cycle import compute_D_cycle
            D = compute_D_cycle(
                gaussians, cameras, render_func, pipe, bg, sigma=d_cycle_sigma,
                depth_maps=shared_depth_maps,  # reuse pre-rendered
            )
            gaussians._d_cycle_cache = D.detach()
        else:
            D = gaussians._d_cycle_cache
    else:
        # Behavior cũ — D_DAV2 (depth consistency với DepthAnything V2 prior).
        D = compute_depth_consistency(xyz, cameras, aligned_depth_dict, depth_range)

    # ── [CRSGaussian Phase 9] R signal: enabled / disabled ──
    # disable_r_signal=True (Phase 9) → R=None, formula bỏ qua R hoàn toàn.
    if need_r_compute:
        R = compute_reprojection_consistency(xyz, cameras)
    else:
        R = None

    # ── [CRSGaussian Phase 8b] S signal: SH stability (optional 3rd dim) ──
    # Chỉ kích hoạt sau sh_stability_warmup (cần đủ EMA samples).
    # update_sh_stability gọi từ train.py mỗi crs_update_interval; ở đây chỉ
    # READ (compute_S từ EMA đã update) — không trigger thêm computation.
    S = None
    if use_sh_reliability and iter >= sh_stability_warmup:
        from utils.crs.sh_stability import compute_S_stability
        S = compute_S_stability(gaussians, scale=1.0)  # (N, 1)

    # ── [CRSGaussian Phase 8 + 9] Multi-component CRS formula ──
    # Auto-normalize weights theo components active:
    #   D + R + S   (Phase 8 FULL) : w_dr=(1-w_s)/2 each, w_s = crs_w_s
    #   D + R       (Phase 5/7)    : w1·D + w2·R                 (legacy 2-component)
    #   D + S       (Phase 9 NoR+S): w_d=(1-w_s),     w_s = crs_w_s
    #   D only      (Phase 9 NoR)  : score = D directly
    if R is not None and S is not None:
        w_s = crs_w_s
        w_dr = (1.0 - w_s) / 2.0
        score = w_dr * D + w_dr * R + w_s * S
    elif R is not None and S is None:
        score = w1 * D + w2 * R
    elif R is None and S is not None:
        w_s = crs_w_s
        w_d = 1.0 - w_s
        score = w_d * D + w_s * S
    else:
        # D-only formula — Phase 9 D_ONLY_* configs.
        score = D

    # ── Score → scale → logit ──
    # Trừ 0.5 để center quanh 0: score=0.5 (neutral) → logit=0 → sigmoid=0.5.
    # Scale mở rộng range: floater (score≈0.1) → logit≈-2.0 → CRS≈0.12
    #                       surface (score≈0.9)  → logit≈2.0  → CRS≈0.88
    crs_logit = scale * (score - 0.5)  # (N, 1)

    # ── EMA update trên logit space ──
    # Lần đầu gọi: _crs_score = 0 (init từ T2.2), EMA sẽ kéo về
    # crs_logit thực tế. Sau ~5 lần update (500 iter), EMA ổn định.
    gaussians._crs_score = ema * gaussians._crs_score + (1.0 - ema) * crs_logit

    # ── [CRSGaussian Phase 9] Diagnostic return ──
    # Khi R=None (Phase 9 D-only), trả neutral tensor thay vì None để giữ
    # backward compat với diagnostic loggers (log_crs_stability, etc.).
    R_for_diag = R if R is not None else torch.full_like(D, 0.5)
    return D, R_for_diag  # [CRSGaussian] Return cho diagnostics (crs_diagnostics.py)


# ============================================================
# [CRSGaussian Tier 2-min] render_crs_map — alpha-composite CRS per pixel
# File: utils/crs/crs_module.py
# Mục đích: Render CRS_pix(p) = Σ Tᵢ(p) · αᵢ(p) · CRSᵢ thông qua
#           override_color path (verified support tại
#           gaussian_renderer/__init__.py:172/182). Bypass SH eval
#           sạch (không bias 0.282 hay offset 0.5).
#           Normalize bằng alpha (=Σαᵢ Tᵢ) để có weighted-AVERAGE
#           thay vì sum (tránh pixel ít cover bị giả thấp).
# Được gọi từ: train.py loss reweighter block khi use_loss_reweight=True.
# ============================================================


@torch.no_grad()
def render_crs_map(gaussians, camera, render_func, pipe, bg):
    """Render alpha-composited CRS map per pixel — proper weighted average.

    Args:
        gaussians: GaussianModel — cần .get_crs (N,1) ∈ [0,1].
        camera: Camera object hiện tại (training viewpoint).
        render_func: gaussian_renderer.render.
        pipe: PipelineParams.
        bg: background tensor (3,) GPU.

    Returns:
        crs_map: (H, W) tensor float [0, 1] — CRS_pix per pixel.
                 Pixel coverage thấp (alpha < eps) → fallback CRS = 1.0
                 (không reweight, hành vi giống pixel reliable).
    """
    crs = gaussians.get_crs.detach().clamp(0.0, 1.0)  # (N, 1)
    N = crs.shape[0]
    # Broadcast CRS sang 3 channels để feed vào override_color path.
    # crs_rgb shape (N, 3) — rasterizer trả 3-channel image, ta chỉ cần 1.
    crs_rgb = crs.expand(N, 3).contiguous()

    # Render qua override_color → bypass SH eval (verified
    # gaussian_renderer/__init__.py:172/182). disable_dropout=True để
    # render đầy đủ — không skip Gaussian random như training.
    pkg = render_func(
        camera, gaussians, pipe, bg,
        override_color=crs_rgb, disable_dropout=True,
    )

    # pkg["render"] shape (3, H, W) — 3 channels equal vì broadcast đầu vào.
    # Lấy mean để giảm về 1 channel (3 channels giống nhau, mean ổn định).
    crs_pix_raw = pkg["render"].mean(dim=0)  # (H, W) — Σ αᵢ Tᵢ CRSᵢ
    alpha = pkg["alpha"].squeeze(0)          # (H, W) — Σ αᵢ Tᵢ (pixel coverage)

    # Normalize: weighted-average thay vì sum.
    # Pixel cover thấp (alpha → 0): tránh chia 0 → fallback về 1.0 (no reweight).
    # Lý do fallback 1.0 (không 0): alpha thấp = vùng chưa rendered, không phải
    # vùng floater. Loss reweighter dùng giá trị này → w(p) = γ + (1-γ)·1 = 1.
    crs_map = torch.where(
        alpha > 1e-3,
        crs_pix_raw / (alpha + 1e-6),
        torch.ones_like(crs_pix_raw),
    ).clamp(0.0, 1.0)
    return crs_map
