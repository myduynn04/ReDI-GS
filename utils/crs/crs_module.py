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
        # aligned_depth_dict lưu CPU tensors → chuyển GPU khi cần.
        # Với 3 cameras sparse-view, overhead transfer không đáng kể
        # (~0.7 MB/image ở 504x378).
        depth_prior_map = aligned_depth_dict[cam.uid].to(device)  # (H, W)
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

    Returns:
        R: (N, 1) tensor float32 trên GPU, range [0, 1].
           R ≈ 1.0: color consistent cross-view (surface).
           R ≈ 0.0: color inconsistent (floater).
           R = 0.5: visible < 2 cameras, không đủ pair (neutral).
    """
    N = xyz.shape[0]
    device = xyz.device

    # Lọc cameras có GT image (skip PseudoCamera)
    valid_cams = [c for c in cameras if hasattr(c, 'original_image')]
    K = len(valid_cams)

    if K < 2:
        # Không đủ camera để tạo pair → tất cả neutral
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
        depth = pts_cam[:, 2]              # (N,)

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
    has_pairs = (pair_count > 0).squeeze(1)  # (N,)

    # Gaussian visible < 2 cameras → không có pair → neutral 0.5.
    # Không đủ thông tin cross-view để đánh giá consistency.
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
) -> None:
    """Tính D_i, R_i rồi update gaussians._crs_score in-place bằng EMA.

    Công thức:
        crs_logit = scale * (w1 * D_i + w2 * R_i - 0.5)
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
            Ablate: {0.5/0.5, 0.7/0.3, 0.3/0.7, 0.6/0.4, 0.4/0.6, 0.8/0.2, 0.2/0.8}
        scale: scale factor trước sigmoid. Default 5.0.
            Mở rộng CRS range: logit ∈ [-scale/2, scale/2].
            Ablate: {3.0, 5.0, 8.0}
        ema: EMA decay. Default 0.9.
            Cao hơn → smooth hơn, phản ứng chậm hơn.
            Ablate: {0.8, 0.9, 0.95}
    """
    xyz = gaussians.get_xyz  # (N, 3) GPU

    # ── Tính 2 tín hiệu ──
    D = compute_depth_consistency(xyz, cameras, aligned_depth_dict, depth_range)
    R = compute_reprojection_consistency(xyz, cameras)

    # ── Weighted average → scale → logit ──
    # Trừ 0.5 để center quanh 0: D=R=0.5 (neutral) → logit=0 → sigmoid=0.5.
    # Scale mở rộng range: floater (D≈0, R≈0.33) → logit≈-1.67 → CRS≈0.16
    #                       surface (D≈1, R≈1.0)  → logit≈2.5  → CRS≈0.92
    crs_logit = scale * (w1 * D + w2 * R - 0.5)  # (N, 1)

    # ── EMA update trên logit space ──
    # Lần đầu gọi: _crs_score = 0 (init từ T2.2), EMA sẽ kéo về
    # crs_logit thực tế. Sau ~5 lần update (500 iter), EMA ổn định.
    gaussians._crs_score = ema * gaussians._crs_score + (1.0 - ema) * crs_logit
