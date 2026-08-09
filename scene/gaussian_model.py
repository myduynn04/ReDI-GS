#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import matplotlib.pyplot as plt
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, fov2focal
from utils.general_utils import strip_symmetric, build_scaling_rotation, chamfer_dist
import open3d as o3d
from torch.optim.lr_scheduler import MultiStepLR


# ============================================================
# [CRSGaussian] Task: T4.1 — _depth_constraint_mask
# File: CRSGaussian/scene/gaussian_model.py  (THÊM VÀO)
# Mục đích: Kiểm tra Gaussian mới có nằm gần depth prior không.
#           Reject Gaussians quá xa surface → ngăn floater sinh ra.
# Lý do (AD-GS 2025): floater hình thành ngay khi densification
#   bắt đầu — Gaussian mới sample tự do → vòng lặp tự khuếch đại.
# Được gọi từ: densify_and_split(), densify_and_clone()
# ============================================================


@torch.no_grad()
def _depth_constraint_mask(new_xyz, cameras, aligned_depth_dict, epsilon_depth):
    """Kiểm tra new_xyz có nằm gần depth prior không.

    Project new_xyz xuống tất cả cameras, so depth projected với
    depth prior. Accept nếu BẤT KỲ camera nào cho |diff| < epsilon.
    Reject nếu TẤT CẢ cameras đều cho diff lớn hoặc invisible.

    Args:
        new_xyz: (M, 3) positions của Gaussians mới, GPU.
        cameras: list of Camera objects.
        aligned_depth_dict: {cam.uid: Tensor (H,W)} GPU.
        epsilon_depth: float — max depth error cho phép.

    Returns:
        keep: (M,) boolean tensor — True = accept, False = reject.
    """
    M = new_xyz.shape[0]
    device = new_xyz.device
    # Track: Gaussian đã pass ít nhất 1 camera chưa
    accepted = torch.zeros(M, dtype=torch.bool, device=device)

    ones = torch.ones(M, 1, device=device, dtype=new_xyz.dtype)
    xyz_hom = torch.cat([new_xyz, ones], dim=1)  # (M, 4)

    for cam in cameras:
        if cam.uid not in aligned_depth_dict:
            continue

        H = cam.image_height
        W = cam.image_width

        W2C = cam.world_view_transform.T  # (4, 4) GPU
        pts_cam = (W2C @ xyz_hom.T).T     # (M, 4)
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

        if valid.sum() == 0:
            continue

        px = pixel_x[valid].long().clamp(0, W - 1)
        py = pixel_y[valid].long().clamp(0, H - 1)

        d_prior = aligned_depth_dict[cam.uid][py, px]  # (K,)
        d_proj = depth[valid]                           # (K,)

        # Accept nếu depth error < epsilon
        # epsilon_depth = 0.05 * depth_range ≈ 1.53 units cho fern
        close_enough = torch.abs(d_proj - d_prior) < epsilon_depth

        # Cập nhật: Gaussian pass camera này → accepted
        valid_indices = valid.nonzero(as_tuple=True)[0]
        accepted[valid_indices[close_enough]] = True

    return accepted


class GaussianModel:
    """
    GaussianModel — data structure trung tâm giữ TOÀN BỘ các Gaussian của scene.

    Một Gaussian được mô tả bởi 6 thuộc tính học được (N = số Gaussian):
        _xyz          (N, 3)     — tâm Gaussian trong không gian 3D
        _features_dc  (N, 1, 3)  — màu cơ bản (SH bậc 0, DC)
        _features_rest(N, 15, 3) — hệ số SH bậc 1-3 → màu đổi theo góc nhìn
        _scaling      (N, 3)     — kích thước ellipsoid theo 3 trục
        _rotation     (N, 4)     — hướng xoay (quaternion)
        _opacity      (N, 1)     — độ đặc/trong suốt

    Dấu `_` = giá trị THÔ (raw, trước activation). Muốn giá trị thật phải qua
    getter (get_scaling = exp(_scaling), get_opacity = sigmoid(_opacity)...).

    Lúc __init__ tất cả tensor đều rỗng (N=0); được nạp thật khi create_from_pcd().
    """

    def setup_functions(self):
        # Đăng ký các "activation function" — hàm chuyển giá trị thô (_) → giá trị thật.
        # Lý do phải có activation: optimizer cần param tự do trong (-∞, +∞), nhưng
        # giá trị vật lý có ràng buộc (scale > 0, opacity ∈ [0,1]). Activation ép về miền hợp lệ.

        # Ghép scaling + rotation → ma trận covariance 3D của Gaussian (dùng khi render).
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)  # L = R·S (ma trận biến đổi)
            actual_covariance = L @ L.transpose(1, 2)   # Σ = L·Lᵀ (đảm bảo đối xứng, semi-definite dương)
            symm = strip_symmetric(actual_covariance)   # lấy 6 phần tử tam giác trên (Σ đối xứng nên đủ)
            return symm

        # scale: raw ∈ (-∞,+∞) → exp → (0,+∞). Đảm bảo kích thước luôn dương.
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log      # nghịch đảo: giá trị thật → raw (dùng khi khởi tạo)

        self.covariance_activation = build_covariance_from_scaling_rotation

        # opacity: raw → sigmoid → (0,1). Đảm bảo độ trong suốt hợp lệ.
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid  # nghịch đảo

        # rotation: normalize quaternion về độ dài 1 (quaternion đơn vị mới biểu diễn phép xoay đúng).
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, args):
        # ── Phần A — Metadata cơ bản ──
        self.args = args                      # lưu toàn bộ args (ModelParams) để các method sau đọc flag
        self.active_sh_degree = 0             # bậc SH đang dùng, bắt đầu = 0 (chỉ màu cơ bản, warm-up)
        self.max_sh_degree = args.sh_degree   # bậc SH tối đa (thường = 3); oneupSHdegree() tăng dần tới đây

        # ── Phần B — 6 thuộc tính Gaussian + buffer densify (tất cả rỗng, N=0) ──
        self.init_point = torch.empty(0)      # lưu point cloud init gốc (để phân tích sau)
        self._xyz = torch.empty(0)            # (N,3) vị trí tâm Gaussian
        self._features_dc = torch.empty(0)    # (N,1,3) màu cơ bản — SH bậc 0
        self._features_rest = torch.empty(0)  # (N,15,3) màu view-dependent — SH bậc 1-3
        self._scaling = torch.empty(0)        # (N,3) kích thước ellipsoid (raw, qua exp mới ra thật)
        self._rotation = torch.empty(0)       # (N,4) quaternion hướng xoay
        self._opacity = torch.empty(0)        # (N,1) độ đặc (raw, qua sigmoid mới ra thật)
        self.max_radii2D = torch.empty(0)     # (N,) bán kính lớn nhất khi chiếu 2D — dùng để prune Gaussian quá to
        self.xyz_gradient_accum = torch.empty(0)  # (N,1) tích lũy gradient vị trí — quyết định densify chỗ nào
        self.denom = torch.empty(0)               # (N,1) đếm số lần accum → lấy trung bình gradient
        self.optimizer = None                 # Adam optimizer, gắn sau ở training_setup()
        self.percent_dense = 0                # ngưỡng % kích thước scene để phân biệt clone vs split
        self.spatial_lr_scale = 0             # scale learning rate của xyz theo kích thước scene
        self.setup_functions()                # đăng ký các activation function (xem ở trên)
        self.bg_color = torch.empty(0)        # màu nền (nếu train background)
        # ── [CRSGaussian DIAG E1] Freeze SH flag ──
        # Khi True → f_dc/f_rest lr đã bị set=0, gradient vẫn flow nhưng
        # param không update. Dùng để test SH overfit hypothesis.
        self._sh_frozen = False
        # ── [CRSGaussian DIAG A2] Freeze DC-only flag ──
        # Khi True → chỉ f_dc lr=0, f_rest vẫn tự do. Dùng để isolate
        # đóng góp của DC drift vs higher-order SH trong overfit.
        self._dc_frozen = False
        self.confidence = torch.empty(0)
        # ── [Phase 13] Activate AbsGS dormant code ──
        # Defensive: args may not have absdensify attr if CLI flag not parsed yet
        self.absdensify = getattr(args, 'absdensify', False)

        # ── [CRSGaussian Phase 13] LFCF tracking attrs ──
        # Init empty; populated in training_setup (zero tensors size N).
        # prev_selected_pts_mask stored as BOOL (not sparse int) — simpler slicing.
        # Guards via `numel() > 0` in prune_points / add_densification_stats /
        # densification_postfix — no-op khi LFCF chưa init (use_lfcf=False).
        self.lff_xyz_grad_accum = torch.empty(0)
        self.lff_denom = torch.empty(0)
        self.prev_lff_xyz_grad = torch.empty(0)
        self.prev_selected_pts_mask_bool = torch.empty(0, dtype=torch.bool)
        self.split_multiplier = 2.0  # constant per EFA-GS pattern

    def capture(self):
        # Gói toàn bộ trạng thái model thành 1 tuple để lưu checkpoint (.pth).
        # Bao gồm 6 thuộc tính Gaussian + buffer densify + optimizer state + CRS score.
        return (
            self.active_sh_degree,          # bậc SH đang active
            self._xyz,                      # vị trí
            self._features_dc,              # màu DC
            self._features_rest,            # màu SH bậc cao
            self._scaling,                  # kích thước
            self._rotation,                 # xoay
            self._opacity,                  # độ đặc
            self.max_radii2D,               # buffer prune
            self.xyz_gradient_accum,        # tích lũy gradient
            self.denom,                     # mẫu số accum
            self.optimizer.state_dict(),    # trạng thái Adam (momentum, variance)
            self.spatial_lr_scale,          # scale LR
            self._crs_score,  # [CRSGaussian T2.2] Save CRS to checkpoint
        )

    # [NOT USE] restore chỉ chạy khi resume từ checkpoint — production luôn train mới.
    def restore(self, model_args, training_args):
        # [CRSGaussian T2.2] Backward-compatible: old checkpoints have 12 fields, new have 13
        if len(model_args) == 13:
            (self.active_sh_degree,
             self._xyz,
             self._features_dc,
             self._features_rest,
             self._scaling,
             self._rotation,
             self._opacity,
             self.max_radii2D,
             xyz_gradient_accum,
             denom,
             opt_dict,
             self.spatial_lr_scale,
             self._crs_score) = model_args
        else:
            (self.active_sh_degree,
             self._xyz,
             self._features_dc,
             self._features_rest,
             self._scaling,
             self._rotation,
             self._opacity,
             self.max_radii2D,
             xyz_gradient_accum,
             denom,
             opt_dict,
             self.spatial_lr_scale) = model_args
            # Old checkpoint → init neutral CRS
            self._crs_score = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        # self.optimizer.load_state_dict(opt_dict)

    # ── Getters — đọc giá trị THẬT của Gaussian (raw _ → activation) ──
    # @property = gọi như thuộc tính (gm.get_scaling) chứ không phải hàm (gm.get_scaling()).

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)   # exp(_scaling) → kích thước dương

    @property
    def get_rotation(self):
        w = self.rotation_activation(self._rotation)    # [NOT USE] dòng thừa, w không được dùng
        return self.rotation_activation(self._rotation) # normalize quaternion về độ dài 1

    @property
    def get_xyz(self):
        return self._xyz                                # vị trí không cần activation (raw = thật)

    @property
    def get_features(self):
        # Ghép màu cơ bản (DC) + màu view-dependent (rest) thành 1 tensor SH đầy đủ (N,16,3)
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)   # sigmoid(_opacity) → độ đặc ∈ (0,1)

    # ============================================================
    # [CRSGaussian Phase 2c] Opacity decay (inspired by Binocular3DGS)
    # Mục đích: continuous opacity pressure mỗi iter → zombie Gaussian
    #           giảm opacity dần → bị CRS pruning hoặc legacy opacity prune loại
    # Được gọi từ: train.py sau optimizer.step(), mỗi iter sau densify_from_iter
    # ============================================================
    def opacity_decay(self, factor: float = 0.995):
        """[CRSGaussian Phase 2c] Multiply activated opacity by factor → logit."""
        with torch.no_grad():
            # Activate → multiply → clamp min để tránh inverse_sigmoid(0) = -inf
            opacity = self.get_opacity * factor
            opacity = opacity.clamp(min=1e-6)
            self._opacity.data = self.inverse_opacity_activation(opacity)

    # ── [CRSGaussian T2.2] CRS property ──
    @property
    def get_crs(self):
        """CRS in [0, 1]. Logit stored in self._crs_score."""
        return torch.sigmoid(self._crs_score)

    def get_covariance(self, scaling_modifier=1):
        # Ghép scale + rotation → ma trận covariance Σ (dùng khi rasterizer render Gaussian)
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    # ── [CRSGaussian Phase 13] In-place scaling delta helper ──
    # Per EFA-GS pattern (gaussian_model.py:133-146): direct .data modify,
    # KHÔNG sync optimizer Adam state. Trade-off acknowledged: Adam momentum/
    # variance cho enlarged Gaussians out-of-sync. EFA-GS authors validated
    # — keep behavior identical, không thêm sync logic.
    @torch.no_grad()
    def set_attributes(self, attribute, mask, changes):
        """Add changes to attribute[mask] in-place (.data direct modify)."""
        assert attribute in ["xyz", "scaling", "opacity", "rotation"], \
            f"attribute {attribute} not allowed for set_attributes"
        real_attr = "_" + attribute
        getattr(self, real_attr).data[mask] += changes.to(self._scaling.device)

    # Tăng bậc SH active lên 1 (warm-up màu). train.py gọi định kỳ: học màu thô trước,
    # chi tiết view-dependent (bậc cao) sau, tới khi đạt max_sh_degree.
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float,
                         informed_crs0=None, q_init=None):
        """Khởi tạo Gaussians từ COLMAP point cloud.

        Args:
            pcd: BasicPointCloud — points, colors, normals
            spatial_lr_scale: float — scale cho learning rate
            informed_crs0: (N,) torch.Tensor logit hoặc None
                [CRSGaussian T5.3] Informed CRS₀ từ compute_informed_crs0().
                None → neutral CRS₀=0.5 (behavior cũ).
        """
        self.spatial_lr_scale = spatial_lr_scale   # lưu scale LR (= cameras_extent, radius scene)

        # ── Bước 1: nạp vị trí + màu từ point cloud lên GPU ──
        # pcd.points (N,3) numpy → tensor float CUDA. Đây là tọa độ 3D của mỗi điểm.
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).cuda().float()
        # pcd.colors (N,3) RGB ∈ [0,1] → RGB2SH đổi sang hệ số SH bậc 0 (DC).
        # Vì 3DGS lưu màu dưới dạng SH chứ không phải RGB trực tiếp.
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        # ── Bước 2: dựng tensor SH features (N, 3, 16) — 3 kênh màu × 16 hệ số SH ──
        # (max_sh_degree+1)² = (3+1)² = 16 hệ số cho SH bậc 3.
        features = torch.zeros((fused_point_cloud.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        if self.args.use_color:
            features[:, :3, 0] =  fused_color   # đặt màu DC (hệ số SH[0]) = màu điểm; bậc cao để = 0
        features[:, 3:, 1:] = 0.0               # (an toàn) các hệ số bậc cao khởi tạo 0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        self.init_point = fused_point_cloud     # lưu point cloud gốc để phân tích sau

        # ── Bước 3: khởi tạo SCALE dựa trên khoảng cách tới láng giềng ──
        # distCUDA2 = khoảng cách bình phương tới điểm gần nhất (k-nearest). Điểm càng thưa
        # → Gaussian càng to để lấp khoảng trống. clamp_min tránh chia 0 / log(0).
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud)[0], 0.0000001)
        # scale thật = sqrt(dist2); lưu ở raw space nên phải log (vì get_scaling = exp).
        # repeat(1,3): cùng 1 scale cho cả 3 trục → Gaussian khởi tạo hình cầu (isotropic).
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        # ── Bước 4: khởi tạo ROTATION = quaternion đơn vị (không xoay) ──
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1   # quaternion (1,0,0,0) = phép xoay identity (Gaussian không nghiêng)

        # ── Bước 5: khởi tạo OPACITY = 0.1 cho mọi Gaussian ──
        # 0.1 là giá trị thật; inverse_sigmoid đưa về raw space (vì get_opacity = sigmoid).
        # Khởi tạo mờ (0.1) để training tự tăng opacity cho Gaussian cần thiết.
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # ── Bước 6: bọc mọi thứ thành nn.Parameter (requires_grad → optimizer học được) ──
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # tách DC (SH[0]) và rest (SH[1:]) thành 2 param riêng; transpose để shape (N,1,3)/(N,15,3)
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")  # buffer prune, init 0
        self.confidence = torch.ones_like(opacities, device="cuda")   # độ tin cậy (confidence rasterizer), init 1
        # ── [CRSGaussian T5.3] CRS score per Gaussian — informed hoặc neutral ──
        # Logit space: sigmoid(0) = 0.5 (neutral).
        # Khi informed_crs0 được truyền vào: dùng geometry prior thay neutral.
        # KHÔNG dùng self.confidence (đã dành cho rasterizer).
        if informed_crs0 is not None:
            self._crs_score = informed_crs0.unsqueeze(-1).to("cuda")
        else:
            self._crs_score = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")
        # ── [CRSGaussian P37 H1] Q_init — độ tin cậy vị trí 3D từ triangulation ──
        # Buffer TĨNH, KHÔNG phải nn.Parameter, KHÔNG vào optimizer.
        # Cùng khuôn với spawn_iter: đi qua prune/densify nhưng không được học.
        # None khi use_roma_qinit=False → mọi chỗ dùng đều có guard → behavior
        # giống hệt trước khi có Phase 37.
        # Lưu _q_init_mean để điền cho Gaussian con (xem densification_postfix).
        if q_init is not None:
            if q_init.shape[0] != fused_point_cloud.shape[0]:
                raise ValueError(
                    f"[P37] q_init lệch chiều dài: {q_init.shape[0]} vs "
                    f"point cloud {fused_point_cloud.shape[0]}"
                )
            self._q_init = q_init.reshape(-1, 1).float().to("cuda")
            self._q_init_mean = float(self._q_init.mean().item())
            print(f"[P37] Q_init loaded: N={self._q_init.shape[0]} "
                  f"mean={self._q_init_mean:.4f} std={self._q_init.std().item():.4f}")
        else:
            self._q_init = None
            self._q_init_mean = 0.5
        # ── [CRSGaussian Hướng D MVP] Spawn iter tracking ──
        # Per-Gaussian "creation iter" để compute age trong rnrc warmup logic.
        # Init Gaussians (từ COLMAP point cloud) → spawn_iter=0.
        # Densified Gaussians sẽ được set spawn_iter=current_iter trong densification_postfix.
        self.spawn_iter = torch.zeros(fused_point_cloud.shape[0], device="cuda", dtype=torch.int32)
        if self.args.train_bg:
            self.bg_color = nn.Parameter((torch.zeros(3, 1, 1) + 0.).cuda().requires_grad_(True))




    # Gắn Adam optimizer + khởi tạo các buffer tích lũy gradient. Gọi 1 lần trước training loop.
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense   # ngưỡng % scene phân biệt clone vs split
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")       # accum gradient thường
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")   # accum gradient tuyệt đối (AbsGS)
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # max gradient tuyệt đối
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")                     # mẫu số đếm accum

        # ── [Phase 13] LFCF init — safe always (guarded by numel() elsewhere) ──
        # Init zeros tensors size N. Cost = negligible (~few KB total).
        # Even if use_lfcf=False, these stay zero — add_densification_stats
        # parallel accum will run (no-op effect on Phase 8 FULL behavior).
        N = self.get_xyz.shape[0]
        self.lff_xyz_grad_accum = torch.zeros((N, 1), device="cuda")
        self.lff_denom = torch.zeros((N, 1), device="cuda")
        self.prev_lff_xyz_grad = torch.zeros((N, 1), device="cuda")
        self.prev_selected_pts_mask_bool = torch.zeros(N, dtype=torch.bool, device="cuda")

        # Mỗi thuộc tính Gaussian là 1 param group với learning rate RIÊNG.
        # Lý do LR khác nhau: vị trí (xyz) cần LR nhỏ scale theo scene; màu bậc cao (f_rest)
        # học chậm hơn màu DC 20 lần để tránh overfit view-dependent.
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},       # màu DC
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},  # màu SH bậc cao, LR/20
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]
        if self.args.train_bg:   # [NOT USE] train background color — production không bật
            l.append({'params': [self.bg_color], 'lr': 0.001, "name": "bg_color"})

        # Adam optimizer với lr=0 mặc định (mỗi group tự set lr riêng ở trên)
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # Scheduler riêng cho xyz — LR giảm dần theo hàm mũ từ init→final trong quá trình train
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)


    def update_learning_rate(self, iteration):
        ''' Cập nhật LR của xyz theo scheduler mỗi iteration (chỉ xyz có scheduler, còn lại cố định) '''
        xyz_lr = self.xyz_scheduler_args(iteration)   # tính LR mới cho iter này
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group['lr'] = xyz_lr
                return xyz_lr

    # ============================================================
    # [CRSGaussian DIAG E1] freeze_sh
    # File: scene/gaussian_model.py
    # Mục đích: Set lr=0 cho f_dc và f_rest param groups → SH coeffs
    #           không update nữa, trong khi xyz/opacity/scaling/rotation
    #           vẫn tự do optimize. Dùng để test hypothesis "SH overfit
    #           memorize training views là nguyên nhân train-test gap".
    # Cách dùng: gọi 1 lần khi iteration == freeze_sh_after trong train.py.
    # Lý do không remove param groups: densification code (prune/densify/
    # clone) assume 6 groups cố định — remove sẽ crash.
    # ============================================================
    # [NOT USE] freeze_sh = GLOBAL freeze (mọi Gaussian). Production dùng SELECTIVE freeze
    # theo CRS (apply_crs_modulated_sh_freeze) — per-Gaussian, khác hẳn. Đây chỉ là DIAG tool.
    def freeze_sh(self):
        """[CRSGaussian DIAG E1] Freeze SH bằng cách set lr=0 cho f_dc, f_rest."""
        if self._sh_frozen:
            return  # đã frozen, không log lại
        n_frozen = 0
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in ("f_dc", "f_rest"):
                param_group['lr'] = 0.0
                n_frozen += 1
        self._sh_frozen = True
        print(f"[DIAG E1] SH frozen ({n_frozen} param groups). "
              f"Only xyz/opacity/scaling/rotation will update.")

    # ============================================================
    # [CRSGaussian DIAG A2] freeze_dc
    # Mục đích: Freeze CHỈ f_dc (DC color component), f_rest tự do.
    #           Cùng pattern với freeze_sh nhưng skip "f_rest" → isolate
    #           DC contribution trong SH overfit.
    # Cách dùng: gọi khi iteration == freeze_dc_start_iter.
    # ============================================================
    # [NOT USE] DIAG tool — isolate DC drift, không dùng trong production recipe.
    def freeze_dc(self):
        """[CRSGaussian DIAG A2] Freeze chỉ f_dc param group (lr=0)."""
        if self._dc_frozen:
            return
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "f_dc":
                param_group['lr'] = 0.0
        self._dc_frozen = True
        print(f"[DIAG A2] DC-only frozen (f_dc lr=0). "
              f"f_rest/xyz/opacity/scaling/rotation still update.")


    # Sinh danh sách tên cột cho file .ply (mỗi Gaussian là 1 dòng, mỗi thuộc tính là các cột).
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']   # vị trí + normal (normal luôn 0, giữ cho đúng format .ply)
        # Các kênh màu (trừ 3 kênh DC), tên f_dc_0, f_dc_1, ...
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    # Lưu toàn bộ Gaussian ra file .ply (định dạng chuẩn 3DGS, mở được bằng viewer).
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))   # tạo folder nếu chưa có

        # Đưa mọi thuộc tính từ GPU → CPU numpy (detach = ngắt gradient trước khi lưu)
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)     # normal = 0 (3DGS không dùng, chỉ để đủ format)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # Dựng cấu trúc numpy có tên cột rồi ghép mọi thuộc tính lại thành 1 mảng (N, tổng_cột)
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)   # ghi ra đĩa

    # Reset opacity mọi Gaussian về tối đa 0.05 (dùng trong lịch densify gốc 3DGS để "làm mờ lại").
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.05))
        if len(self.optimizer.state.keys()):
            # Phải thay tensor TRONG optimizer (không chỉ gán self._opacity) để Adam state khớp
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]

    # [NOT USE] reset màu về 0 — không thấy gọi trong production recipe.
    def reset_color(self):
        self.active_sh_degree = 0
        new_features_dc = torch.zeros_like(self._features_dc)
        new_features_rest = torch.zeros_like(self._features_rest)
        # opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.05))
        if len(self.optimizer.state.keys()):
            optimizable_tensors = self.replace_tensor_to_optimizer(new_features_dc, "f_dc")
            self._features_dc = optimizable_tensors["f_dc"]
            optimizable_tensors = self.replace_tensor_to_optimizer(new_features_rest, "f_rest")
            self._features_rest = optimizable_tensors["f_rest"]

    # [NOT USE] Nạp Gaussian từ file .ply — chỉ dùng khi resume (scene/__init__.py nhánh loaded_iter).
    # Production luôn init từ point cloud qua create_from_pcd, không qua đây.
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree   # load .ply = model đã train → dùng full SH degree


    # Thay 1 tensor thuộc tính trong optimizer (giữ đồng bộ Adam state), KHÔNG đổi số Gaussian.
    # Dùng bởi reset_opacity/reset_color.
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                # Reset Adam momentum (exp_avg) + variance (exp_avg_sq) về 0 cho tensor mới
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]                    # xóa state cũ
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))  # gắn param mới
                self.optimizer.state[group['params'][0]] = stored_state         # gắn lại state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # Cắt (prune) Gaussian theo mask — giữ phần True. Cắt CẢ Adam state để khớp shape mới.
    # Helper chung, được prune_points/dist_prune gọi.
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ['bg_color']:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # [NOT USE] Prune Gaussian đi quá xa point cloud gốc (chamfer > 3.0). Không thấy gọi trong production.
    def dist_prune(self):
        dist = chamfer_dist(self.init_point, self._xyz)   # khoảng cách tới point cloud init
        valid_points_mask = (dist < 3.0)                  # giữ Gaussian còn gần init
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]


    # Xóa các Gaussian có mask=True (mask = "cần xóa"). Cắt đồng bộ MỌI buffer per-Gaussian.
    def prune_points(self, mask, iter):
        if iter > self.args.prune_from_iter:        # chỉ prune sau iter ngưỡng (tránh xóa sớm khi chưa ổn định)
            valid_points_mask = ~mask               # đảo mask: True = GIỮ LẠI
            optimizable_tensors = self._prune_optimizer(valid_points_mask)  # cắt 6 thuộc tính + Adam state

            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

            # Cắt các buffer densify + CRS theo cùng mask (nếu không sẽ lệch shape → crash)
            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
            self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
            self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[valid_points_mask]

            self.denom = self.denom[valid_points_mask]
            self.max_radii2D = self.max_radii2D[valid_points_mask]
            self.confidence = self.confidence[valid_points_mask]
            # ── [CRSGaussian T2.2] Prune CRS cùng với Gaussian ──
            self._crs_score = self._crs_score[valid_points_mask]
            # ── [CRSGaussian P37 H1] Prune Q_init cùng Gaussian ──
            # Guard 2 lớp: buffer tồn tại VÀ shape khớp mask. Không khớp thì bỏ
            # qua im lặng là SAI — nhưng ở đây shape luôn khớp vì buffer được
            # tạo cùng lúc với _crs_score và đi qua đúng những chỗ này.
            if getattr(self, "_q_init", None) is not None \
                    and self._q_init.shape[0] == valid_points_mask.shape[0]:
                self._q_init = self._q_init[valid_points_mask]
            # ── [CRSGaussian Hướng D MVP] Prune spawn_iter + _rc_smooth + _crs_rnrc ──
            if hasattr(self, "spawn_iter") and self.spawn_iter.shape[0] == valid_points_mask.shape[0]:
                self.spawn_iter = self.spawn_iter[valid_points_mask]
            if hasattr(self, "_rc_smooth") and self._rc_smooth.shape[0] == valid_points_mask.shape[0]:
                self._rc_smooth = self._rc_smooth[valid_points_mask]
            if hasattr(self, "_crs_rnrc") and self._crs_rnrc.shape[0] == valid_points_mask.shape[0]:
                self._crs_rnrc = self._crs_rnrc[valid_points_mask]

            # ── [CRSGaussian Phase 13] LFCF attrs slicing — ADD, không xóa existing ──
            # Guard via numel() — no-op khi LFCF chưa init (use_lfcf=False ở phase 8).
            if self.lff_xyz_grad_accum.numel() > 0:
                self.lff_xyz_grad_accum = self.lff_xyz_grad_accum[valid_points_mask]
                self.lff_denom = self.lff_denom[valid_points_mask]
                self.prev_lff_xyz_grad = self.prev_lff_xyz_grad[valid_points_mask]
            if self.prev_selected_pts_mask_bool.numel() > 0:
                self.prev_selected_pts_mask_bool = self.prev_selected_pts_mask_bool[valid_points_mask]


    # Ngược với _prune_optimizer: NỐI THÊM Gaussian mới vào optimizer (khi densify clone/split).
    # Adam state của Gaussian mới init = 0.
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ['bg_color']:   # bg_color không phải per-Gaussian → bỏ qua
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # "postfix" = bước chốt sau densify: nhận danh sách Gaussian mới (từ clone/split),
    # nối vào model + optimizer, reset các buffer accum, và gán CRS/spawn_iter cho chúng.
    # Mọi nhánh densify (clone, split, LFCF) đều kết thúc bằng cách gọi hàm này.
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, parent_crs_logits=None, eta=0.0,
                              new_prev_selected_bool=None):
        """Append new Gaussians vào model + optimizer.

        Args:
            parent_crs_logits: (M, 1) tensor hoặc None — CRS logit của parents.
                [CRSGaussian T5.4] Was used by crs_densify_inherit (removed Phase 24).
                Kept default None — child CRS₀ neutral (sigmoid 0.5).
            eta: float — inherit factor. 0.0 = neutral (behavior cũ).
            new_prev_selected_bool: (M,) bool tensor — [Phase 13] LFCF prev_selected
                state cho new Gaussians. None (non-LFCF path) → default False.
                LFCF split children pass True (just got "selected" via split).
        """
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.confidence = torch.cat([self.confidence, torch.ones(new_opacities.shape, device="cuda")], 0)
        # ── [CRSGaussian T5.4] CRS cho Gaussians mới ──
        # eta > 0: conservative inherit — child CRS₀ = clip(η*CRS_parent, 0, 0.5)
        #   Child không bao giờ trên neutral → phải "earn" CRS cao qua D_i/R_i.
        #   parent_crs_logits truyền từ densify_and_split/clone.
        # eta = 0: neutral (behavior cũ) — logit=0 → sigmoid=0.5
        if eta > 0 and parent_crs_logits is not None:
            # sigmoid: logit → CRS ∈ [0, 1] của parent
            parent_crs = torch.sigmoid(parent_crs_logits).squeeze(-1)   # (M,)
            # η * parent_crs: scale down, cap tại 0.5
            # Lý do cap=0.5: child ở vị trí khác parent (split sample noise,
            # clone copy) → geometry chưa proven → không nên bắt đầu trên neutral.
            # 1e-6 lower bound: tránh log(0) trong inverse sigmoid
            inherited = (parent_crs * eta).clamp(1e-6, 0.5)
            # Inverse sigmoid: CRS → logit để lưu vào _crs_score (logit space)
            child_logit = torch.log(inherited / (1 - inherited))        # inverse sigmoid
            new_crs = child_logit.unsqueeze(-1)                         # (M, 1)
        else:
            # Behavior cũ: neutral logit=0 → sigmoid=0.5
            new_crs = torch.zeros((new_xyz.shape[0], 1), device="cuda")
        self._crs_score = torch.cat([self._crs_score, new_crs], dim=0)
        # ── [CRSGaussian P37 H1] Append Q_init cho Gaussians mới ──
        # Gaussian con nhận TRUNG BÌNH toàn cục của Q_init lúc khởi tạo.
        # Sau khi center_q trừ trung bình, giá trị này thành ~0 → con KHÔNG
        # được thưởng cũng KHÔNG bị phạt. Trung tính đúng nghĩa.
        #
        # ⚠ ĐÂY LÀ LỰA CHỌN CÓ CHỦ Ý CHO S2 (LOG-ONLY), KHÔNG PHẢI luật cuối.
        #   Ngữ nghĩa đúng hơn là KẾ THỪA — con của một điểm triangulate tồi
        #   cũng đứng sai chỗ (clone copy vị trí, split lấy mẫu quanh đó).
        #   Nhưng kế thừa cần truyền parent index qua 3-4 chỗ gọi
        #   densification_postfix (split / clone / LFCF) = đụng nhiều code
        #   production. HOÃN tới S3, và chỉ làm nếu S2 chứng minh tín hiệu đáng.
        #
        #   Gaussian gốc vẫn phân biệt được bằng spawn_iter == 0 (đã có sẵn),
        #   nên S2 đo tương quan trên đúng tập mang tín hiệu — không cần
        #   bookkeeping mới.
        if getattr(self, "_q_init", None) is not None:
            fill = float(getattr(self, "_q_init_mean", 0.5))
            new_q = torch.full((new_xyz.shape[0], 1), fill,
                               device="cuda", dtype=self._q_init.dtype)
            self._q_init = torch.cat([self._q_init, new_q], dim=0)
        # ── [CRSGaussian Hướng D MVP] Append spawn_iter cho Gaussians mới ──
        # Read iter từ instance attr set bởi densify_and_prune. Default 0 nếu chưa set
        # (e.g. proximity() hoặc init path).
        if hasattr(self, "spawn_iter"):
            cur_iter = getattr(self, "_densify_current_iter", 0)
            new_spawn = torch.full((new_xyz.shape[0],), cur_iter,
                                    device="cuda", dtype=torch.int32)
            self.spawn_iter = torch.cat([self.spawn_iter, new_spawn], dim=0)

        # ── [CRSGaussian Phase 13] LFCF attrs append cho new Gaussians ──
        # Note: xyz_gradient_accum/denom RESET to zeros (size N_total) above.
        # lff_xyz_grad_accum/lff_denom theo cùng pattern (reset to zeros).
        # prev_lff_xyz_grad: KHÔNG reset — append zeros (children chưa có history).
        # prev_selected_pts_mask_bool: append per new_prev_selected_bool kwarg.
        if self.lff_xyz_grad_accum.numel() > 0:
            n_total = self.get_xyz.shape[0]
            n_new = new_xyz.shape[0]
            new_zeros = torch.zeros((n_new, 1), device="cuda")
            self.lff_xyz_grad_accum = torch.zeros((n_total, 1), device="cuda")
            self.lff_denom = torch.zeros((n_total, 1), device="cuda")
            self.prev_lff_xyz_grad = torch.cat([self.prev_lff_xyz_grad, new_zeros], dim=0)

            if new_prev_selected_bool is not None:
                # LFCF split children: inherit "selected" state from parent (True)
                appendee = new_prev_selected_bool.to(device="cuda")
            else:
                # Standard path (clone/split): children not in LFCF selection
                appendee = torch.zeros(n_new, dtype=torch.bool, device="cuda")
            self.prev_selected_pts_mask_bool = torch.cat(
                [self.prev_selected_pts_mask_bool, appendee], dim=0
            )


    # [NOT USE] Không có caller nào trong codebase — thêm Gaussian ở trung điểm các cặp
    # điểm xa nhau. Có thể là code thử nghiệm cũ, giữ lại nhưng không chạy.
    def proximity(self, scene_extent, N = 3):
        dist, nearest_indices = distCUDA2(self.get_xyz)
        selected_pts_mask = torch.logical_and(dist > (5. * scene_extent),
                                              torch.max(self.get_scaling, dim=1).values > (scene_extent))

        new_indices = nearest_indices[selected_pts_mask].reshape(-1).long()
        source_xyz = self._xyz[selected_pts_mask].repeat(1, N, 1).reshape(-1, 3)
        target_xyz = self._xyz[new_indices]
        new_xyz = (source_xyz + target_xyz) / 2
        new_scaling = self._scaling[new_indices]
        new_rotation = torch.zeros_like(self._rotation[new_indices])
        new_rotation[:, 0] = 1
        new_features_dc = torch.zeros_like(self._features_dc[new_indices])
        new_features_rest = torch.zeros_like(self._features_rest[new_indices])
        new_opacity = self._opacity[new_indices]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)



    # [CRSGaussian T5.4] +eta param cho conservative CRS inherit
    # SPLIT = tách 1 Gaussian TO thành N Gaussian nhỏ (dùng cho vùng cần chi tiết mà Gaussian đang phủ quá rộng).
    # Điều kiện: gradient cao VÀ kích thước lớn (> percent_dense × scene).
    def densify_and_split(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, iter, N=2,
                          cameras=None, aligned_depth_dict=None, depth_range=None,
                          eta=0.0):
        n_init_points = self.get_xyz.shape[0]
        # Chọn Gaussian có gradient vị trí ≥ ngưỡng (dấu hiệu vùng đang under-fit, cần thêm chi tiết)
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # ── AbsGS (thesis khối 6) ──: thêm tiêu chí gradient TUYỆT ĐỐI, gộp bằng OR.
        # Lý do: vùng texture mịn có gradient triệt tiêu nhau (ngược dấu) → tiêu chí gốc bỏ sót.
        if self.absdensify:
            padded_grad_abs = torch.zeros((n_init_points), device="cuda")
            padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
            selected_pts_mask_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)
            selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        # Chỉ split Gaussian ĐỦ TO (kích thước > percent_dense × scene) — Gaussian nhỏ thì clone thay vì split
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        dist, _ = distCUDA2(self.get_xyz)
        selected_pts_mask2 = torch.logical_and(dist > (self.args.dist_thres * scene_extent),
                                               torch.max(self.get_scaling, dim=1).values > ( scene_extent))
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask2)

        # Sinh N vị trí con: lấy mẫu ngẫu nhiên theo phân bố Gaussian gốc (std = scale của nó),
        # xoay theo rotation gốc rồi cộng vào tâm gốc → N con nằm trong "đám mây" của Gaussian cha.
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)                 # sample offset ngẫu nhiên
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        # Con nhỏ hơn cha: chia scale cho 0.8·N (tổng thể tích con ≈ cha nhưng chi tiết hơn)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        # ── [CRSGaussian T4.1] Position constraint — DISABLED ──
        # Tắt sau thực nghiệm: DAV2 depth prior noise → reject Gaussians
        # hợp lệ → chất lượng giảm rõ rệt.
        # Depth loss + CRS pruning đủ kiểm soát floater.
        # if aligned_depth_dict is not None and cameras is not None and depth_range is not None:
        #     epsilon_depth = 0.05 * depth_range
        #     keep = _depth_constraint_mask(new_xyz, cameras, aligned_depth_dict, epsilon_depth)
        #     if keep.sum() < new_xyz.shape[0]:
        #         new_xyz = new_xyz[keep]
        #         new_scaling = new_scaling[keep]
        #         new_rotation = new_rotation[keep]
        #         new_features_dc = new_features_dc[keep]
        #         new_features_rest = new_features_rest[keep]
        #         new_opacity = new_opacity[keep]

        # [CRSGaussian T5.4] Truyền parent CRS logit cho conservative inherit.
        # Split tạo N children mỗi parent → repeat N lần để match new_xyz.
        _parent_crs = self._crs_score[selected_pts_mask].repeat(N, 1) if eta > 0 else None
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   parent_crs_logits=_parent_crs, eta=eta)

        # Sau khi tạo N con, XÓA Gaussian cha (mask=True cho cha, False cho con vừa thêm)
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, iter)


    # CLONE = NHÂN ĐÔI 1 Gaussian NHỎ (copy y nguyên, giữ cả cha). Dùng cho vùng under-reconstruct
    # mà Gaussian đang quá nhỏ (< percent_dense × scene). Khác split ở chỗ giữ cha + không thu nhỏ.
    def densify_and_clone(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent,
                          cameras=None, aligned_depth_dict=None, depth_range=None,
                          eta=0.0):
        # Chọn Gaussian gradient cao (giống split), + AbsGS OR (gradient tuyệt đối)
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        if self.absdensify:
            selected_pts_mask_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False)
            selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        # Chỉ clone Gaussian NHỎ (≤ ngưỡng) — đây là điều kiện ngược với split
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        # ── [CRSGaussian T4.1] Position constraint — DISABLED ──
        # DAV2 noise → reject cả Gaussian hợp lệ.
        # if aligned_depth_dict is not None and cameras is not None and depth_range is not None:
        #     epsilon_depth = 0.05 * depth_range
        #     keep = _depth_constraint_mask(new_xyz, cameras, aligned_depth_dict, epsilon_depth)
        #     if keep.sum() < new_xyz.shape[0]:
        #         new_xyz = new_xyz[keep]
        #         new_features_dc = new_features_dc[keep]
        #         new_features_rest = new_features_rest[keep]
        #         new_opacities = new_opacities[keep]
        #         new_scaling = new_scaling[keep]
        #         new_rotation = new_rotation[keep]

        # [CRSGaussian T5.4] Truyền parent CRS logit cho conservative inherit.
        # Clone copy nguyên parent → 1:1 mapping.
        _parent_crs = self._crs_score[selected_pts_mask] if eta > 0 else None
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, parent_crs_logits=_parent_crs, eta=eta)


    # ⭐ HÀM ĐIỀU PHỐI densify (thesis khối 6) — train.py gọi hàm này mỗi 100 iter.
    # Nó quyết định: chạy LFCF (mỗi iter chia hết 200) HAY standard clone+split (AbsGS),
    # rồi prune Gaussian opacity thấp / quá to. Là "nhạc trưởng" của toàn bộ densification.
    # [CRSGaussian T5.4] +eta param cho conservative CRS inherit → chain xuống clone/split
    # [Phase 13] +is_lfcf_iter/lfcf_opts/cameras_for_lfcf cho LFCF mode (default OFF)
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iter,
                          cameras=None, aligned_depth_dict=None, depth_range=None,
                          T_warmup=1000, tau_crs=0.35, tau_isolated=0.1,
                          crs_prune_dict=None, eta=0.0,
                          is_lfcf_iter=False, lfcf_opts=None, cameras_for_lfcf=None):
        # ── [CRSGaussian Hướng D MVP] Lưu iter để densification_postfix track spawn ──
        self._densify_current_iter = int(iter)

        if is_lfcf_iter and lfcf_opts is not None:
            # ── [CRSGaussian Phase 13] LFCF path ──
            # Replace standard clone+split với tolerance-based decision + diffscale.
            # Uses separate lff_xyz_grad_accum (parallel to standard grad accum).
            from utils.densify.lfcf import compute_lfcf_decisions

            denom_safe = self.lff_denom.clamp(min=1.0)
            grads = self.lff_xyz_grad_accum / denom_safe
            grads[grads.isnan()] = 0.0

            result = compute_lfcf_decisions(
                grads=grads,
                prev_lff_xyz_grad=self.prev_lff_xyz_grad,
                prev_selected_pts_mask_bool=self.prev_selected_pts_mask_bool,
                xyz=self.get_xyz,
                scaling=self._scaling.data,
                cameras=cameras_for_lfcf,
                grad_threshold=max_grad,
                **lfcf_opts,  # scaling_multiplier_max/min, training_percent_powered,
                              # splitting_ub, splitting_lb, tolerance, diffscale
            )

            enlarged_mask = result['enlarged_mask']
            splitted_mask = result['splitted_mask']
            enlarged_changes = result['enlarged_scaling_changes']
            splitted_changes = result['splitted_scaling_changes']
            log_mult = result['log_scaling_multiplier']
            interval_coef = result['interval_coef']
            selected_now_bool = result['selected_pts_mask_bool']

            # Apply enlarge (in-place .data modify, no optimizer sync — EFA-GS pattern)
            if enlarged_mask.any():
                self.set_attributes("scaling", enlarged_mask, enlarged_changes)

            # Apply split shrink trên parents + probabilistic lottery
            if splitted_mask.any():
                self.set_attributes("scaling", splitted_mask, splitted_changes)

                # Probabilistic split lottery (depth-aware, EFA-GS:653-655)
                prob = (
                    interval_coef * (lfcf_opts['splitting_ub'] - lfcf_opts['splitting_lb'])
                    + lfcf_opts['splitting_lb']
                ).squeeze(-1)
                lottery = torch.rand_like(prob) <= prob
                final_split = splitted_mask & lottery

                if final_split.any():
                    self._lfcf_split_children(
                        final_split, log_mult, eta, lfcf_opts['diffscale'],
                        selected_now_at_call_time=selected_now_bool,
                    )
                else:
                    # Không có split children → save selected state ngay
                    self.prev_selected_pts_mask_bool = selected_now_bool
            else:
                self.prev_selected_pts_mask_bool = selected_now_bool

            # Save grads cho next LFCF iter tolerance compare
            self.prev_lff_xyz_grad = grads.clone()

            # Reset LFCF accumulator (post-LFCF iter — fresh for next interval)
            self.lff_xyz_grad_accum.zero_()
            self.lff_denom.zero_()

        else:
            # ── Standard path (AbsGS, không LFCF) ──
            grads = self.xyz_gradient_accum / self.denom     # gradient vị trí trung bình (accum / số lần)
            grads[grads.isnan()] = 0.0

            grads_abs = self.xyz_gradient_accum_abs / self.denom   # gradient TUYỆT ĐỐI trung bình (AbsGS)
            grads_abs[grads_abs.isnan()] = 0.0
            # Ngưỡng AbsGS thích ứng: ratio = tỉ lệ Gaussian vượt ngưỡng gốc;
            # Q = quantile (1-ratio) của grads_abs → chọn đúng cùng tỉ lệ Gaussian bằng tiêu chí tuyệt đối.
            ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
            Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)

            # [CRSGaussian T4.1] Truyền depth constraint params xuống clone/split.
            # cameras=None → skip constraint (backward compat khi không dùng --use_depth_prior).
            self.densify_and_clone(grads, max_grad, grads_abs, Q, extent,
                                   cameras=cameras, aligned_depth_dict=aligned_depth_dict, depth_range=depth_range,
                                   eta=eta)
            self.densify_and_split(grads, max_grad, grads_abs, Q, extent, iter,
                                   cameras=cameras, aligned_depth_dict=aligned_depth_dict, depth_range=depth_range,
                                   eta=eta)

        # ── Legacy pruning (3DGS gốc) — xóa Gaussian opacity quá thấp / quá to ──
        prune_mask = (self.get_opacity < min_opacity).squeeze()   # opacity < ngưỡng → coi như vô hình, xóa
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size            # to trên màn hình 2D
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent  # to trong không gian 3D
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        # ── [CRSGaussian T4.2] CRS pruning — Option C ──
        # prune = (CRS < tau_crs AND isolated) OR legacy
        # Lý do Option C: T2.7 cho thấy floater opacity=0.90 →
        # AND(CRS, opacity<0.005) vô hiệu. Option C tách CRS thành
        # kênh prune riêng, không cần opacity thấp.
        #
        # Chỉ active sau T_warmup: CRS trước T_warmup là noise
        # (chưa đủ EMA updates) → prune sai nếu dùng sớm.
        #
        # isolated = knn_dist > tau_isolated * extent:
        # Floater thật đứng một mình trong 3D space.
        # Gaussian đang học có CRS thấp nhưng nằm gần surface
        # (có neighbors) → không bị prune nhầm.
        if iter > T_warmup and crs_prune_dict is not None:
            crs_scores = self.get_crs.squeeze()                    # (N,) [0,1]
            crs_low = (crs_scores < tau_crs)                       # floater candidate

            # distCUDA2 trả (dist, indices) — dist là khoảng cách tới
            # nearest neighbor, đã import sẵn từ simple_knn._C
            knn_dist, _ = distCUDA2(self.get_xyz)                  # (N,)
            isolated = (knn_dist > tau_isolated * extent)           # đứng một mình

            crs_prune = crs_low & isolated                         # floater thật
            prune_mask = torch.logical_or(prune_mask, crs_prune)

        self.prune_points(prune_mask, iter)

        # [CRSGaussian Phase 2d Stage A] Invalidate density cache khi population
        # thay đổi (densify tạo thêm / prune xóa bớt → density outdated).
        # anchor_dropout_mask sẽ recompute ở iter sau.
        if hasattr(self, "_cached_density"):
            self._cached_density = None

        torch.cuda.empty_cache()


    # ============================================================
    # [CRSGaussian Phase 13] LFCF split children helper
    # Source: EFA-GS gaussian_model.py:657-672 (split children gen)
    # Khác EFA-GS:
    #   - N=1 children per parent (LFCF default)
    #   - CRS inherit T5.5 via parent_crs_logits + eta
    #   - Children prev_selected_bool = True (just got "selected" via split)
    # ============================================================
    def _lfcf_split_children(self, splitted_mask, log_scaling_multiplier, eta, diffscale,
                              selected_now_at_call_time):
        """Generate split children + prune parents (LFCF path).

        Args:
            splitted_mask: (N_before,) bool — parents to split
            log_scaling_multiplier: (N_before, 1) — per-Gaussian log mult
            eta: float — CRS inherit factor (T5.5 conservative)
            diffscale: bool — pass-through to stds computation
            selected_now_at_call_time: (N_before,) bool — selection mask
                tại thời điểm vào LFCF iter (lưu vào prev_selected_pts_mask_bool)
        """
        from utils.densify.lfcf import compute_split_stds_diffscale
        from utils.general_utils import build_rotation

        N_before = self.get_xyz.shape[0]
        n_new = int(splitted_mask.sum().item())
        if n_new == 0:
            return

        # ── Compute LFCF stds (diffscale-aware) ──
        # Use activated scaling (get_scaling = exp(_scaling)) for std cho normal sampling
        stds = compute_split_stds_diffscale(
            self.get_scaling[splitted_mask],
            log_scaling_multiplier[splitted_mask],
            diffscale, self.split_multiplier,
        )
        stds = torch.where(torch.isnan(stds), torch.full_like(stds, 1e-3), stds)

        # ── Generate children (N=1 per parent) ──
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[splitted_mask])
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[splitted_mask]
        new_scaling = self.scaling_inverse_activation(self.get_scaling[splitted_mask])
        new_rotation = self._rotation[splitted_mask]
        new_features_dc = self._features_dc[splitted_mask]
        new_features_rest = self._features_rest[splitted_mask]
        new_opacity = self._opacity[splitted_mask]

        # ── CRS inherit via T5.5 (parent_crs_logits + eta) ──
        parent_crs = self._crs_score[splitted_mask] if eta > 0 else None

        # ── Save updated prev_selected BEFORE postfix (postfix concats children) ──
        # selected_now_at_call_time là (N_before,) bool — gán trực tiếp.
        # Children sẽ được append True (per EFA-GS pattern line 557) trong postfix
        # qua new_prev_selected_bool kwarg.
        self.prev_selected_pts_mask_bool = selected_now_at_call_time
        children_prev_selected = torch.ones(n_new, dtype=torch.bool, device="cuda")

        # ── densification_postfix appends children + handles CRS inherit + LFCF attrs ──
        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacity, new_scaling, new_rotation,
            parent_crs_logits=parent_crs, eta=eta,
            new_prev_selected_bool=children_prev_selected,
        )

        # ── Prune parents (children survive — placed AFTER parents trong tensor) ──
        N_after = self.get_xyz.shape[0]
        assert N_after == N_before + n_new, \
            f"LFCF split shape mismatch: N_after={N_after} vs N_before+n_new={N_before + n_new}"
        prune_mask = torch.cat([
            splitted_mask,                                          # (N_before,) — prune parents
            torch.zeros(n_new, dtype=torch.bool, device="cuda"),    # children survive
        ], dim=0)
        self.prune_points(prune_mask, iter=self._densify_current_iter)


    # Gọi MỖI ITER (train.py:667/672): tích lũy gradient vị trí của Gaussian visible, để
    # densify_and_prune (mỗi 100 iter) lấy trung bình và quyết định densify chỗ nào.
    # update_filter = mask Gaussian visible trong render vừa rồi.
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # gradient THƯỜNG (2 chiều xy màn hình) — tiêu chí densify gốc 3DGS
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        # gradient TUYỆT ĐỐI (channel 2:) — cho AbsGS (không triệt tiêu khi ngược dấu)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs_max[update_filter] = torch.max(self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True))
        self.denom[update_filter] += 1   # đếm số lần accum để lấy trung bình

        # ── [Phase 13] LFCF parallel grad accumulation (guarded by numel) ──
        # Track LFCF accumulator parallel với standard. Cost: 1 extra norm + add.
        # Reset trong densification_postfix sau densify (cùng pattern standard).
        if self.lff_xyz_grad_accum.numel() > 0:
            self.lff_xyz_grad_accum[update_filter] += torch.norm(
                viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
            )
            self.lff_denom[update_filter] += 1
        
    
    
    # ══════════════════════════════════════════════════════════════════
    # [NOT USE] 5 helper dưới đây (clone_from_mask, split_from_mask,
    # compute_prune_mask, prune_from_mask, reset_opacity_from_mask) KHÔNG
    # có caller trong production (chỉ prune_from_mask xuất hiện ở 1 dòng đã
    # comment tại train.py:890). Là API dự phòng / thử nghiệm cũ. Densify
    # thật đi qua densify_and_prune, không qua các hàm này.
    # ══════════════════════════════════════════════════════════════════
    def clone_from_mask(self, selected_pts_mask, repeat=1):
        for i in range(repeat):
            new_xyz = self._xyz[selected_pts_mask]
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacities = self._opacity[selected_pts_mask]
            new_scaling = self._scaling[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
            selected_pts_mask = torch.cat((selected_pts_mask, torch.zeros(new_xyz.size(0), device="cuda", dtype=bool)))

    def split_from_mask(self, selected_pts_mask, iter, N=2, repeat=1):
        for i in range(repeat):
            stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
            new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
            new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
            selected_pts_mask = torch.cat((selected_pts_mask, torch.zeros(new_xyz.size(0), device="cuda", dtype=bool)))

        prune_filter = selected_pts_mask
        self.prune_points(prune_filter, iter)

    def compute_prune_mask(self, max_grad, min_opacity, extent, max_screen_size, iter, low_density=False, prox=False, ld_iter=2000):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        return prune_mask
    
    def prune_from_mask(self, prune_mask, iter):
        self.prune_points(prune_mask, iter)

    def reset_opacity_from_mask(self, mask):
        valid_points_mask = ~mask
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.05))
        opacities_new[valid_points_mask] = self.get_opacity[valid_points_mask]
        if len(self.optimizer.state.keys()):
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]
    