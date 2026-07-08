# ============================================================
# [CRSGaussian Phase 13] LFCF — Low-Frequency Component First
# File: utils/densify/lfcf.py (NEW)
# Source: EFA-GS/3DGS/scene/gaussian_model.py:165-225 (helpers) + 589-672 (densify_lff)
#
# Mục đích: Port EFA-GS densification mechanism vào CRSGaussian.
#   1. Tolerance-based decision: enlarge (grad không giảm = noise) vs split (grad giảm = signal)
#   2. Diffscale volume-preserving isotropify: shrink major axis, enlarge minor
#   3. Depth-aware probabilistic split: deeper Gaussian → higher split prob
#
# Pure Python/PyTorch — KHÔNG động rasterizer CUDA.
# Functions return masks + scale changes; caller (gaussian_model.densify_and_prune)
# dispatch physical changes (set_attributes, densification_postfix, prune_points).
# ============================================================

import math

import torch


# ============================================================
# [CRSGaussian Phase 13] LFCF decay schedule
# Source: EFA-GS/3DGS/train.py:32-38
# ============================================================
def calculate_training_percent_powered(iter, densify_from, densify_until,
                                        pow=1.0, lower_bound=0.0):
    """Decay 1.0 → lower_bound over [densify_from, densify_until]. pow=1.0 = linear.

    Args:
        iter: current iteration
        densify_from: iter bắt đầu densify
        densify_until: iter dừng densify
        pow: decay rate (1.0 linear, >1 slower decay, <1 faster decay)
        lower_bound: floor cuối schedule

    Returns: scalar float ∈ [lower_bound, 1.0]
    """
    di = max(0.0, min(1.0, (densify_until - iter) / max(densify_until - densify_from, 1)))
    pow_di = di ** pow * (1 - lower_bound) + lower_bound
    return pow_di


# ============================================================
# [CRSGaussian Phase 13] Compute 3D interval (per-Gaussian projected depth)
# Source: EFA-GS/3DGS/scene/gaussian_model.py:165-212
# Adapted cho CoR-GS-base Camera class (dùng FoVx/FoVy thay focal_x/focal_y).
# ============================================================
@torch.no_grad()
def compute_3D_interval(xyz, cameras):
    """Per-Gaussian min projected depth across training cameras.

    Deeper Gaussian (larger z khi projected) → larger interval → smaller multiplier
    sau normalize (depth-aware: deep Gaussian được enlarge ÍT hơn, split prob CAO hơn).

    Args:
        xyz: (N, 3) tensor — Gaussian positions, world space
        cameras: list of Camera objects với attrs:
                 R (np.ndarray 3x3), T (np.ndarray 3,),
                 FoVx, FoVy (radians), image_width, image_height
                 (focal_x/focal_y computed via fov2focal — CoR-GS Camera class)

    Returns: interval (N, 1) tensor — projected depth / focal_length
    """
    from utils.graphics_utils import fov2focal

    distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
    valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
    focal_length = 0.0
    for cam in cameras:
        # Compute focal từ FoV + image dimensions (CoR-GS Camera convention)
        focal_x = fov2focal(cam.FoVx, cam.image_width)
        focal_y = fov2focal(cam.FoVy, cam.image_height)
        R = torch.tensor(cam.R, device=xyz.device, dtype=torch.float32)
        T = torch.tensor(cam.T, device=xyz.device, dtype=torch.float32)
        # R stored transposed (GLM convention) — no transpose here
        xyz_cam = xyz @ R + T[None, :]
        valid_depth = xyz_cam[:, 2] > 0.2
        z = torch.clamp(xyz_cam[:, 2], min=0.001)
        x = xyz_cam[:, 0] / z * focal_x + cam.image_width / 2.0
        y = xyz_cam[:, 1] / z * focal_y + cam.image_height / 2.0
        # Tangent-space filtering (extend bounds 15% per side per EFA-GS)
        in_screen = (
            (x >= -0.15 * cam.image_width) & (x <= 1.15 * cam.image_width)
            & (y >= -0.15 * cam.image_height) & (y <= 1.15 * cam.image_height)
        )
        valid = valid_depth & in_screen
        distance[valid] = torch.min(distance[valid], z[valid])
        valid_points = valid_points | valid
        if focal_length < focal_x:
            focal_length = focal_x
    # Gaussians không visible từ bất kỳ cam → assign max depth của visible set
    if valid_points.any():
        distance[~valid_points] = distance[valid_points].max()
    else:
        distance[:] = 1.0
    interval = distance / max(focal_length, 1e-6)
    return interval[..., None]


# ============================================================
# [CRSGaussian Phase 13] Normalize interval to [0, 1]
# Source: EFA-GS/3DGS/scene/gaussian_model.py:215-225
# ============================================================
@torch.no_grad()
def normalize_interval(interval, opt='log', norm='minmax'):
    """Normalize depth interval → [0, 1] range.

    Args:
        interval: (N, 1) tensor
        opt: 'log' (default — amplify depth differences) | 'exp' | identity
        norm: 'minmax' (default) | 'one' (divide by sum)

    Returns: (N, 1) tensor normalized
    """
    if opt == 'exp':
        interval = torch.exp(interval)
    elif opt == 'log':
        interval = torch.log(interval)
    if norm == 'minmax':
        mn, mx = torch.min(interval), torch.max(interval)
        return (interval - mn) / (mx - mn + 1e-9)
    elif norm == 'one':
        return interval / (interval.sum() + 1e-9)
    return interval


# ============================================================
# [CRSGaussian Phase 13] Core LFCF decision logic
# Source: EFA-GS/3DGS/scene/gaussian_model.py:589-655 (densify_lff body)
# ============================================================
@torch.no_grad()
def compute_lfcf_decisions(
    grads, prev_lff_xyz_grad, prev_selected_pts_mask_bool,
    xyz, scaling, cameras, grad_threshold,
    scaling_multiplier_max, scaling_multiplier_min,
    training_percent_powered,
    splitting_ub, splitting_lb,
    tolerance, diffscale,
):
    """Compute enlarge/split masks + log-space scale deltas per LFCF logic.

    Tolerance-based decision (Section 4.1 design doc):
        - new_selected (chưa pick lần trước, pick lần này) → ENLARGE (chưa biết signal/noise)
        - intersection (pick liên tục 2 kỳ):
            * grad ĐANG GIẢM (decent) → SPLIT (modeling real signal)
            * grad KHÔNG giảm → ENLARGE (noise/aliasing stuck)

    Diffscale (Section 4.2 design doc):
        Volume-preserving isotropify khi enlarge:
            - shrink major axis (×mult^(-2/3))
            - shrink middle axis (×mult^(-1/3))
            - enlarge minor axis (×mult^(+1))
            → mult^(1 - 1/3 - 2/3) = mult^0 → volume preserved

    Args:
        grads: (N, 1) tensor — current LFCF gradients
        prev_lff_xyz_grad: (N, 1) tensor — gradients từ kỳ LFCF trước
        prev_selected_pts_mask_bool: (N,) bool tensor — selection mask kỳ trước
        xyz: (N, 3) — Gaussian positions
        scaling: (N, 3) — log-space scaling (raw param, not activated)
        cameras: list of Camera (for compute_3D_interval)
        grad_threshold: float — selection threshold
        scaling_multiplier_max: float ≥ 1.0 — enlarge ceiling
        scaling_multiplier_min: float ≥ 1.0 — enlarge floor (deep Gaussians)
        training_percent_powered: float ∈ [0, 1] — decay factor
        splitting_ub: float — pass-through (caller compute lottery)
        splitting_lb: float — pass-through (caller compute lottery)
        tolerance: float — FP-stability tolerance for grad comparison
        diffscale: bool — apply volume-preserving isotropify

    Returns dict:
        'enlarged_mask': (N,) bool — Gaussians cần enlarge
        'splitted_mask': (N,) bool — Gaussians cần split (UN-FILTERED bởi lottery)
        'enlarged_scaling_changes': (M_enlarge, 3) — log-space delta cho enlarge
        'splitted_scaling_changes': (M_split, 3) — log-space delta (shrink) cho parents
        'log_scaling_multiplier': (N, 1) — full per-Gaussian log mult
        'interval_coef': (N, 1) — normalized depth coef
        'selected_pts_mask_bool': (N,) — current selection (save for next iter)
    """
    assert scaling_multiplier_max >= 1.0, f"{scaling_multiplier_max=} < 1.0"
    assert scaling_multiplier_min >= 1.0, f"{scaling_multiplier_min=} < 1.0"
    assert 0.0 <= training_percent_powered <= 1.0, f"{training_percent_powered=} not in [0,1]"

    device = scaling.device
    N = xyz.shape[0]

    # ── Step 1: Chọn Gaussian gradient cao (giống tiêu chí gốc: ‖grad‖ ≥ ngưỡng) ──
    selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

    # ── Step 2: [THESIS Eq 3.15 — "selected in two consecutive cycles"] ──
    # Phân biệt Gaussian được chọn LẦN ĐẦU vs LIÊN TỤC 2 kỳ.
    # LFCF cần lịch sử gradient → phải được chọn 2 kỳ liên tiếp mới so được g_cur vs g_prev.
    prev_mask = prev_selected_pts_mask_bool
    # Defensive: N đổi giữa 2 kỳ LFCF (densify/prune) → reset prev mask
    if prev_mask.shape[0] != N:
        prev_mask = torch.zeros(N, dtype=torch.bool, device=device)

    intersect_mask = prev_mask & selected_pts_mask       # chọn LIÊN TỤC 2 kỳ → có lịch sử để so
    enlarged_mask = (~prev_mask) & selected_pts_mask      # chọn LẦN ĐẦU → chưa biết signal/noise → ENLARGE
    splitted_mask = torch.zeros_like(enlarged_mask, dtype=torch.bool)

    # ── Step 3: [THESIS Eq 3.15 — quyết định split vs enlarge theo lịch sử gradient] ──
    # action = SPLIT nếu (chọn 2 kỳ liên tiếp) VÀ (g_cur ≤ g_prev − ε_tol);  ENLARGE nếu ngược lại.
    # - g_cur ≤ g_prev − tol (gradient ĐANG GIẢM) = Gaussian hội tụ về tín hiệu thật → SPLIT (thêm chi tiết)
    # - g_cur không giảm (dao động/kẹt) = nhiễu/aliasing → chỉ ENLARGE (phủ vùng, không thêm chi tiết)
    if intersect_mask.any():
        whether_decent = (
            torch.norm(grads[intersect_mask], dim=-1)
            <= torch.norm(prev_lff_xyz_grad[intersect_mask] - tolerance, dim=-1)  # g_cur ≤ g_prev − ε_tol
        )
        enlarged_mask[intersect_mask] = ~whether_decent   # không giảm → enlarge
        splitted_mask[intersect_mask] = whether_decent     # giảm → split

    # ── Step 4: [THESIS Eq 3.16 — hệ số enlarge phụ thuộc độ sâu] ──
    # m_k = δ_k·m_min + (1−δ_k)·m_max, với δ_k = độ sâu chiếu chuẩn hóa (gần 1 = xa camera).
    interval = compute_3D_interval(xyz, cameras)                       # độ sâu chiếu per-Gaussian
    interval_coef = normalize_interval(interval, 'log', 'minmax').to(device)  # δ_k ∈ [0,1]
    # δ cao (xa) → hệ số về m_min (enlarge ÍT, thiên về split);  δ thấp (gần) → về m_max (enlarge NHIỀU)
    scaling_multiplier_coef = (
        interval_coef * scaling_multiplier_min          # δ_k · m_min
        + (1 - interval_coef) * scaling_multiplier_max  # (1−δ_k) · m_max
    )
    # Nhân với training_percent_powered = decay theo iter (cuối training enlarge nhẹ dần)
    log_scaling_multiplier = torch.log(scaling_multiplier_coef) * training_percent_powered

    # ── Step 5: [THESIS "volume-preserving transformation"] — diffscale isotropify khi enlarge ──
    # 3 trục sắp TĂNG dần [s_min, s_mid, s_max], nhân lần lượt m^(+1), m^(−1/3), m^(−2/3).
    # Tích số mũ = 1 − 1/3 − 2/3 = 0 → m^0 = 1 → THỂ TÍCH KHÔNG ĐỔI, chỉ đổi hình dạng.
    # Mục đích: kéo Gaussian dài-nhọn (needle) về tròn hơn (isotropic) → giảm streaking artifact.
    enlarged_scaling_changes = torch.zeros_like(scaling[enlarged_mask], dtype=torch.float32, device=device)
    if diffscale and enlarged_mask.any():
        coef_of_enlarge = torch.ones_like(enlarged_scaling_changes)   # mặc định trục nhỏ nhất = +1 (enlarge)
        enlarged_sorted_indices = torch.sort(scaling[enlarged_mask], dim=1, descending=False)[1]  # [s_min,s_mid,s_max]
        coef_of_enlarge.scatter_(1, enlarged_sorted_indices[:, 1].unsqueeze(1), -1.0 / 3.0)  # s_mid → m^(−1/3)
        coef_of_enlarge.scatter_(1, enlarged_sorted_indices[:, 2].unsqueeze(1), -2.0 / 3.0)  # s_max → m^(−2/3) (co mạnh nhất)
        enlarged_scaling_changes += log_scaling_multiplier[enlarged_mask] * coef_of_enlarge
    elif enlarged_mask.any():
        # Không diffscale: enlarge đều 3 trục (thể tích tăng m^3) — bản đơn giản
        enlarged_scaling_changes += log_scaling_multiplier[enlarged_mask]

    # ── Step 6: Compute split shrink changes (per EFA-GS:640-647) ──
    # Split children inherit parent scaling × (1 / split_multiplier) — parent's
    # scale gets shrunk via splitted_scaling_changes (negative log delta).
    SPLIT_MULTIPLIER = 2.0  # constant per EFA-GS pattern
    splitted_scaling_changes = torch.zeros_like(scaling[splitted_mask], dtype=torch.float32, device=device)
    if splitted_mask.any():
        log_split_mult = torch.log(torch.tensor(SPLIT_MULTIPLIER, device=device))
        if diffscale:
            # Sort DESC: [s_max, s_mid, s_min]
            # coef: s_max → 1.0 (shrink most), s_mid → 0.5 (shrink moderately), s_min → 0 (unchanged)
            coef_of_split = torch.ones_like(splitted_scaling_changes)
            splitted_sorted_indices = torch.sort(scaling[splitted_mask], dim=1, descending=True)[1]
            coef_of_split.scatter_(1, splitted_sorted_indices[:, 1].unsqueeze(1), 0.5)
            coef_of_split.scatter_(1, splitted_sorted_indices[:, 2].unsqueeze(1), 0.0)
            splitted_scaling_changes -= 0.5 * (
                log_scaling_multiplier[splitted_mask] + log_split_mult
            ) * coef_of_split
        else:
            splitted_scaling_changes -= 0.5 * (log_scaling_multiplier[splitted_mask] + log_split_mult)

    return {
        'enlarged_mask': enlarged_mask,
        'splitted_mask': splitted_mask,
        'enlarged_scaling_changes': enlarged_scaling_changes,
        'splitted_scaling_changes': splitted_scaling_changes,
        'log_scaling_multiplier': log_scaling_multiplier,
        'interval_coef': interval_coef,
        'selected_pts_mask_bool': selected_pts_mask,
    }


# ============================================================
# [CRSGaussian Phase 13] Compute split stds với diffscale option
# Source: EFA-GS/3DGS/scene/gaussian_model.py:657 (split children stds)
# ============================================================
def compute_split_stds_diffscale(scaling, log_scaling_multiplier,
                                  diffscale, split_multiplier=2.0):
    """Compute stds cho split children sampling (gaussian noise around parent xyz).

    Khi diffscale=True, split children get inherited shape (scaling unchanged for std).
    Khi diffscale=False, split applied uniform shrink.

    EFA-GS pattern: stds = get_scaling[splitted_mask] (use activated scaling directly).
    Children xyz = parent xyz + normal(0, stds). Children scaling inherit parent scaling.

    Args:
        scaling: (M, 3) — activated scaling của parents-to-split (i.e. torch.exp(_scaling))
        log_scaling_multiplier: (M, 1) — passed-through (not used in current EFA-GS pattern,
                                kept for API completeness)
        diffscale: bool — current EFA-GS keeps stds = get_scaling regardless;
                           kept as arg cho potential variation
        split_multiplier: float — pass-through (not used internally)

    Returns: stds (M, 3) tensor
    """
    # Per EFA-GS line 657: stds = self.get_scaling[splitted_mask].repeat(N, 1)
    # với N=1 cho LFCF → just return scaling clone
    stds = scaling.clone()
    return stds
