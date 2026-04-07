# ============================================================
# [CRSGaussian] Task: T5.2 — Informed CRS₀ Initialization
# File: CRSGaussian/utils/crs/crs_init.py  (TẠO MỚI)
# Mục đích: Tính CRS₀ có ý nghĩa hình học cho COLMAP Gaussians
#           thay vì neutral 0.5. Dùng 3 signals có sẵn sau
#           alignment: reproj quality, depth agreement, view support.
# Được gọi từ: train.py, sau Scene() init + depth alignment,
#              trước training loop. Overwrite _crs_score trực tiếp.
# ============================================================

import torch
import numpy as np
from utils.graphics_utils import fov2focal
from utils.depth.depth_alignment import _load_colmap_points


def compute_reproj_quality(reproj_errors, tau_r=2.5):
    """[CRSGaussian T5.2] Tính q_reproj: chất lượng SfM reprojection per-point.

    q_reproj = 1 - clip(error / tau_r, 0, 1)
    - error=0  → q=1.0 (perfect SfM triangulation)
    - error≥τ_r → q=0.0 (unreliable point)
    τ_r=2.5: nhất quán với depth_alignment.py max_reproj_err filter.

    Args:
        reproj_errors: (N,) numpy array — COLMAP reprojection errors (pixels)
        tau_r: float — normalization threshold. Ablate: {2.0, 2.5, 3.0}

    Returns:
        (N,) numpy array — range [0, 1]
    """
    # max(tau_r, 1e-6): tránh chia cho 0 nếu tau_r bị set sai
    # np.clip: đảm bảo output [0, 1] — error âm (không nên xảy ra) → q=1
    q = 1.0 - np.clip(reproj_errors / max(tau_r, 1e-6), 0.0, 1.0)
    return q


@torch.no_grad()
def compute_depth_agreement(points_xyz, cameras, aligned_depth_dict, depth_range):
    """[CRSGaussian T5.2] Tính q_depth: DAV2-COLMAP depth agreement per-point.

    Project mỗi COLMAP 3D point lên từng camera → lấy d_COLMAP (z sau W2C)
    vs d_DAV2 (lookup aligned depth map) → average qua cameras visible.

    Invisible point (không nằm trong frame bất kỳ camera nào) → q=0.5 (neutral).
    Lý do: không có thông tin để đánh giá → không penalize cũng không reward.

    Args:
        points_xyz: (N, 3) torch.Tensor float32 GPU — COLMAP 3D points
        cameras: list of Camera objects
        aligned_depth_dict: {cam.uid: Tensor (H,W) GPU} — aligned metric depth
        depth_range: float — median(far) - median(near) across cameras

    Returns:
        (N,) numpy array — range [0, 1]
    """
    N = points_xyz.shape[0]
    device = points_xyz.device
    # Tránh chia cho 0 nếu tất cả cameras cùng near/far
    depth_range_safe = max(depth_range, 1e-6)

    # Homogeneous coords (N, 4) — tính 1 lần, reuse cho mọi camera
    ones = torch.ones(N, 1, device=device, dtype=points_xyz.dtype)
    xyz_hom = torch.cat([points_xyz, ones], dim=1)  # (N, 4)

    # Tích lũy q_depth per-camera rồi average — giống pattern crs_module.py
    q_sum = torch.zeros(N, device=device)
    count = torch.zeros(N, device=device)

    for cam in cameras:
        if cam.uid not in aligned_depth_dict:
            continue

        H = cam.image_height
        W = cam.image_width

        # ── World → Camera transform ──
        # world_view_transform lưu column-major (transposed) trên cuda.
        # .T → row-major W2C (4x4). Cùng convention với crs_module.py và
        # depth_alignment.py — KHÔNG dùng R.T @ P + T vì getWorld2View2
        # có translate+scale adjustment mà R/T riêng lẻ không capture.
        W2C = cam.world_view_transform.T  # (4, 4) GPU

        pts_cam = (W2C @ xyz_hom.T).T  # (N, 4)
        # depth = z component trong camera space
        # d_COLMAP: depth "thật" từ COLMAP 3D point → camera
        depth = pts_cam[:, 2]           # (N,)

        # Camera intrinsics từ Field of View
        fx = fov2focal(cam.FoVx, W)
        fy = fov2focal(cam.FoVy, H)
        cx, cy = W / 2.0, H / 2.0

        pixel_x = pts_cam[:, 0] / depth * fx + cx
        pixel_y = pts_cam[:, 1] / depth * fy + cy

        # ── Visibility filter ──
        # depth > 0: point trước camera (không phải phía sau)
        # in frame: pixel nằm trong image bounds
        # Point không qua filter → skip, không penalize
        valid = (
            (depth > 0)
            & (pixel_x >= 0) & (pixel_x < W)
            & (pixel_y >= 0) & (pixel_y < H)
        )

        if valid.sum().item() == 0:
            continue

        # clamp cho safety dù valid mask đã đảm bảo in-bounds
        px = pixel_x[valid].long().clamp(0, W - 1)
        py = pixel_y[valid].long().clamp(0, H - 1)

        # ── So sánh depth ──
        # d_dav2: aligned depth từ DAV2 tại pixel tương ứng
        # d_colmap: depth từ project COLMAP 3D point qua camera matrix
        # Chênh lệch lớn → point nằm sai depth → q thấp
        d_dav2 = aligned_depth_dict[cam.uid][py, px]  # (M,)
        d_colmap = depth[valid]                        # (M,)

        # q = 1 - clip(|d_DAV2 - d_COLMAP| / depth_range, 0, 1)
        # depth_range normalize: cho phép cùng ngưỡng hoạt động trên mọi scene
        q_cam = 1.0 - torch.abs(d_dav2 - d_colmap) / depth_range_safe
        q_cam = q_cam.clamp(0.0, 1.0)

        q_sum[valid] += q_cam
        count[valid] += 1.0

    # ── Average qua cameras visible ──
    # Invisible point (count=0) → 0.5 (neutral)
    # Lý do: không có thông tin depth agreement → không penalize cũng
    # không reward. CRS update sau T_warmup sẽ refine từ D_i/R_i thực tế.
    result = torch.full((N,), 0.5, device=device)
    visible = count > 0
    result[visible] = q_sum[visible] / count[visible]

    return result.cpu().numpy()


@torch.no_grad()
def compute_view_support(points_xyz, cameras, n_train):
    """[CRSGaussian T5.2] Tính q_view: multi-view stereo support per-point.

    Count bao nhiêu training cameras thấy point (in-frame + depth > 0).
    q_view = (n_obs - 1) / max(N_train - 1, 1)

    Dùng (n-1)/(N-1) thay vì n/N vì:
    - Point thấy từ 1 view duy nhất = không có stereo constraint = q=0
    - Point thấy từ tất cả views = maximum stereo support = q=1
    LLFF 3-view: n_obs ∈ {1,2,3} → q ∈ {0.0, 0.5, 1.0}

    Args:
        points_xyz: (N, 3) torch.Tensor float32 GPU — COLMAP 3D points
        cameras: list of Camera objects
        n_train: int — tổng số training views (3 cho LLFF)

    Returns:
        (N,) numpy array — range [0, 1]
    """
    N = points_xyz.shape[0]
    device = points_xyz.device

    ones = torch.ones(N, 1, device=device, dtype=points_xyz.dtype)
    xyz_hom = torch.cat([points_xyz, ones], dim=1)  # (N, 4)

    # Đếm số cameras mà mỗi point visible (in-frame + depth > 0)
    n_obs = torch.zeros(N, device=device)

    for cam in cameras:
        H = cam.image_height
        W = cam.image_width

        W2C = cam.world_view_transform.T  # (4, 4) GPU
        pts_cam = (W2C @ xyz_hom.T).T     # (N, 4)
        depth = pts_cam[:, 2]

        fx = fov2focal(cam.FoVx, W)
        fy = fov2focal(cam.FoVy, H)
        cx, cy = W / 2.0, H / 2.0

        pixel_x = pts_cam[:, 0] / depth * fx + cx
        pixel_y = pts_cam[:, 1] / depth * fy + cy

        valid = (
            (depth > 0)
            & (pixel_x >= 0) & (pixel_x < W)
            & (pixel_y >= 0) & (pixel_y < H)
        )

        n_obs[valid] += 1.0

    # q = (n_obs - 1) / max(N_train - 1, 1)
    # Dùng (n-1)/(N-1): 1 view = 0 (không có stereo constraint)
    # max(n_train-1, 1): tránh chia 0 khi chỉ có 1 training view
    # .clamp(min=0): n_obs=0 (point ngoài tất cả cameras) → q=0
    denom = max(n_train - 1, 1)
    q = (n_obs - 1).clamp(min=0.0) / denom
    q = q.clamp(0.0, 1.0)

    return q.cpu().numpy()


def compute_informed_crs0(
    points_xyz,
    cameras,
    aligned_depth_dict,
    depth_range,
    source_path,
    n_views,
    # Component switches
    use_reproj=True,
    use_depth=True,
    use_view=True,
    # Weights
    w_reproj=0.333,
    w_depth=0.333,
    w_view=0.334,
    # Hyperparameters
    tau_r=2.5,
    gamma=5.0,
):
    """[CRSGaussian T5.2] Tính informed CRS₀ logit cho COLMAP Gaussians.

    Orchestrator: gọi 3 component functions → weighted average → scale logit.
    Mỗi component bật/tắt độc lập; weights tự normalize khi component tắt.

    Công thức:
        Q_i = sum(w_k * q_k) / sum(w_k)   cho active components k
        ℓᵢ⁽⁰⁾ = γ * (Q_i - 0.5)
    Returns logit để lưu trực tiếp vào _crs_score.

    Args:
        points_xyz: numpy (N, 3) — COLMAP 3D points (từ scene.init_point_cloud.points)
        cameras: list of Camera objects (training cameras)
        aligned_depth_dict: {cam.uid: Tensor (H,W) GPU}
        depth_range: float
        source_path: str — dataset path, dùng để đọc reproj errors từ bin
        n_views: int — số training views
        use_reproj/use_depth/use_view: bool — component switches
        w_reproj/w_depth/w_view: float — component weights
        tau_r: float — reproj error normalization threshold
        gamma: float — logit scale factor

    Returns:
        (N,) torch.Tensor float32 — logit space
        CRS₀ = sigmoid(logit). Range: sigmoid(±γ/2) ≈ [0.08, 0.92] khi γ=5.
    """
    N = len(points_xyz)

    # Đưa points lên GPU cho projection (compute_depth_agreement, compute_view_support
    # dùng cam.world_view_transform trên cuda → points cũng phải GPU)
    pts_gpu = torch.from_numpy(np.asarray(points_xyz)).float().cuda()

    # ── Tính từng component ──
    # Mỗi component bật/tắt độc lập qua use_* switches.
    # Khi tắt: component không tính, weight không tham gia normalize.
    # Cho phép ablation isolate contribution từng signal.
    components = []
    weights = []

    if use_reproj:
        # Đọc reproj errors từ COLMAP bin (KHÔNG dùng scene.point_cloud.errors
        # vì BasicPointCloud không lưu errors — chỉ có points, colors, normals).
        # _load_colmap_points trả (xyz (N',3), errors (N',1)).
        # N' có thể ≠ N nếu dataset_readers filter points →
        # _match_reproj_errors handle bằng nearest-neighbor matching.
        colmap_xyz, colmap_errors = _load_colmap_points(source_path, n_views)
        reproj_errors = _match_reproj_errors(
            points_xyz, colmap_xyz.numpy(), colmap_errors.squeeze(-1).numpy()
        )
        q_reproj = compute_reproj_quality(reproj_errors, tau_r=tau_r)
        components.append(q_reproj)
        weights.append(w_reproj)

    if use_depth:
        # q_depth: so sánh COLMAP depth vs DAV2 aligned depth
        # Point nằm đúng depth → q cao. Floater lệch depth → q thấp.
        q_depth = compute_depth_agreement(
            pts_gpu, cameras, aligned_depth_dict, depth_range
        )
        components.append(q_depth)
        weights.append(w_depth)

    if use_view:
        # q_view: point thấy từ nhiều cameras → có stereo constraint → reliable
        # n_train = len(cameras): số training views thực tế
        q_view = compute_view_support(pts_gpu, cameras, n_train=len(cameras))
        components.append(q_view)
        weights.append(w_view)

    # ── Fallback: tất cả components tắt → neutral ──
    # logit=0 → sigmoid=0.5 → như behavior cũ
    if len(components) == 0:
        return torch.zeros(N, dtype=torch.float32)

    # ── Weighted average với auto-normalize ──
    # Khi 1 component bị tắt: weights của 2 còn lại tự normalize về sum=1.
    # Ví dụ: tắt reproj (w=0.333) → depth (0.333) + view (0.334) = 0.667
    #         → normalize: depth=0.499, view=0.501 ≈ equal weight.
    # Cho phép grid search: w ∈ {0.0, 0.2, 0.33, 0.5, 0.8, 1.0}
    w_sum = sum(weights)
    if w_sum <= 0:
        return torch.zeros(N, dtype=torch.float32)

    Q = np.zeros(N, dtype=np.float32)
    for c, w in zip(components, weights):
        Q += (w / w_sum) * c

    # ── Scale logit: ℓ = γ * (Q - 0.5) ──
    # Q ∈ [0, 1] → (Q-0.5) ∈ [-0.5, 0.5] → logit ∈ [-γ/2, γ/2]
    # Với γ=5: logit ∈ [-2.5, 2.5] → CRS₀ ∈ [0.08, 0.92]
    # γ cần validate từ Q distribution thực tế (T5.8).
    # Nếu Q tập trung quanh 0.5 → γ nhỏ đủ. Nếu spread rộng → γ lớn hơn.
    logit = gamma * (Q - 0.5)

    return torch.from_numpy(logit).float()


def _match_reproj_errors(pcd_points, colmap_xyz, colmap_errors):
    """[CRSGaussian T5.2] Match reproj errors từ COLMAP bin với points trong BasicPointCloud.

    BasicPointCloud.points đến từ cùng points3D.bin (qua .ply conversion).
    Thứ tự có thể giống nhau, nhưng để an toàn ta match bằng nearest neighbor.

    Nếu N_pcd == N_colmap và points gần như trùng nhau → dùng trực tiếp
    (tránh overhead O(N²) khi N lớn).

    Args:
        pcd_points: numpy (N, 3) — points từ scene.init_point_cloud
        colmap_xyz: numpy (N', 3) — points từ _load_colmap_points
        colmap_errors: numpy (N',) — reproj errors tương ứng

    Returns:
        numpy (N,) — reproj errors matched cho mỗi pcd point
    """
    N_pcd = len(pcd_points)
    N_col = len(colmap_xyz)

    pcd_pts = np.asarray(pcd_points, dtype=np.float32)

    # Fast path: cùng size VÀ cùng order → dùng trực tiếp
    # Kiểm tra bằng max distance giữa corresponding points
    if N_pcd == N_col:
        max_dist = np.max(np.abs(pcd_pts - colmap_xyz.astype(np.float32)))
        if max_dist < 1e-4:
            return colmap_errors.copy()

    # Slow path: nearest-neighbor matching
    # Dùng khi points bị filter khác nhau (hiếm, nhưng an toàn)
    # O(N * N') — chấp nhận được vì chỉ chạy 1 lần
    from scipy.spatial import cKDTree
    tree = cKDTree(colmap_xyz.astype(np.float64))
    _, indices = tree.query(pcd_pts.astype(np.float64), k=1)

    return colmap_errors[indices]
