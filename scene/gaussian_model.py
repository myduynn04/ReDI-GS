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

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, args):
        self.args = args
        self.active_sh_degree = 0
        self.max_sh_degree = args.sh_degree
        self.init_point = torch.empty(0)
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.bg_color = torch.empty(0)
        # ── [CRSGaussian DIAG E1] Freeze SH flag ──
        # Khi True → f_dc/f_rest lr đã bị set=0, gradient vẫn flow nhưng
        # param không update. Dùng để test SH overfit hypothesis.
        self._sh_frozen = False
        # ── [CRSGaussian DIAG A2] Freeze DC-only flag ──
        # Khi True → chỉ f_dc lr=0, f_rest vẫn tự do. Dùng để isolate
        # đóng góp của DC drift vs higher-order SH trong overfit.
        self._dc_frozen = False
        self.confidence = torch.empty(0)
        # self.absdensify = args.absdensify
        self.absdensify = False

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._crs_score,  # [CRSGaussian T2.2] Save CRS to checkpoint
        )

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

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        w = self.rotation_activation(self._rotation)
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    # ── [CRSGaussian T2.2] CRS property ──
    @property
    def get_crs(self):
        """CRS in [0, 1]. Logit stored in self._crs_score."""
        return torch.sigmoid(self._crs_score)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float,
                         informed_crs0=None):
        """Khởi tạo Gaussians từ COLMAP point cloud.

        Args:
            pcd: BasicPointCloud — points, colors, normals
            spatial_lr_scale: float — scale cho learning rate
            informed_crs0: (N,) torch.Tensor logit hoặc None
                [CRSGaussian T5.3] Informed CRS₀ từ compute_informed_crs0().
                None → neutral CRS₀=0.5 (behavior cũ).
        """
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).cuda().float()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        features = torch.zeros((fused_point_cloud.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        if self.args.use_color:
            features[:, :3, 0] =  fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        self.init_point = fused_point_cloud

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud)[0], 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.confidence = torch.ones_like(opacities, device="cuda")
        # ── [CRSGaussian T5.3] CRS score per Gaussian — informed hoặc neutral ──
        # Logit space: sigmoid(0) = 0.5 (neutral).
        # Khi informed_crs0 được truyền vào: dùng geometry prior thay neutral.
        # KHÔNG dùng self.confidence (đã dành cho rasterizer).
        if informed_crs0 is not None:
            self._crs_score = informed_crs0.unsqueeze(-1).to("cuda")
        else:
            self._crs_score = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")
        if self.args.train_bg:
            self.bg_color = nn.Parameter((torch.zeros(3, 1, 1) + 0.).cuda().requires_grad_(True))




    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]
        if self.args.train_bg:
            l.append({'params': [self.bg_color], 'lr': 0.001, "name": "bg_color"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        xyz_lr = self.xyz_scheduler_args(iteration)
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


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
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

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.05))
        if len(self.optimizer.state.keys()):
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]

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

        self.active_sh_degree = self.max_sh_degree


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

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

    def dist_prune(self):
        dist = chamfer_dist(self.init_point, self._xyz)
        valid_points_mask = (dist < 3.0)
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


    def prune_points(self, mask, iter):
        if iter > self.args.prune_from_iter:
            valid_points_mask = ~mask
            optimizable_tensors = self._prune_optimizer(valid_points_mask)

            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
            self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
            self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[valid_points_mask]

            self.denom = self.denom[valid_points_mask]
            self.max_radii2D = self.max_radii2D[valid_points_mask]
            self.confidence = self.confidence[valid_points_mask]
            # ── [CRSGaussian T2.2] Prune CRS cùng với Gaussian ──
            self._crs_score = self._crs_score[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ['bg_color']:
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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, parent_crs_logits=None, eta=0.0):
        """Append new Gaussians vào model + optimizer.

        Args:
            parent_crs_logits: (M, 1) tensor hoặc None — CRS logit của parents.
                [CRSGaussian T5.4] Dùng khi crs_densify_inherit=True.
            eta: float — inherit factor. 0.0 = neutral (behavior cũ).
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
    def densify_and_split(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, iter, N=2,
                          cameras=None, aligned_depth_dict=None, depth_range=None,
                          eta=0.0):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        if self.absdensify:
            padded_grad_abs = torch.zeros((n_init_points), device="cuda")
            padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
            selected_pts_mask_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)
            selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        dist, _ = distCUDA2(self.get_xyz)
        selected_pts_mask2 = torch.logical_and(dist > (self.args.dist_thres * scene_extent),
                                               torch.max(self.get_scaling, dim=1).values > ( scene_extent))
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask2)

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

        # ── [CRSGaussian T4.1] Position constraint — DISABLED ──
        # Tắt sau thực nghiệm: DAV2 depth prior noise → reject Gaussians
        # hợp lệ → PSNR giảm 3 dB. Xem decisions_log 2026-04.
        # Depth loss + CRS pruning đủ kiểm soát floater.
        # Code giữ lại để enable nếu cần (e.g. depth prior chính xác hơn).
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

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, iter)


    # [CRSGaussian T5.4] +eta param cho conservative CRS inherit
    def densify_and_clone(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent,
                          cameras=None, aligned_depth_dict=None, depth_range=None,
                          eta=0.0):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        if self.absdensify:
            selected_pts_mask_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False)
            selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
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
        # Xem decisions_log 2026-04. DAV2 noise → reject hợp lệ → -3 dB.
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


    # [CRSGaussian T5.4] +eta param cho conservative CRS inherit → chain xuống clone/split
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iter,
                          cameras=None, aligned_depth_dict=None, depth_range=None,
                          T_warmup=1000, tau_crs=0.35, tau_isolated=0.1,
                          crs_prune_dict=None, eta=0.0):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0
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

        # ── Legacy pruning (3DGS gốc) — giữ nguyên ──
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
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
        torch.cuda.empty_cache()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs_max[update_filter] = torch.max(self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True))
        self.denom[update_filter] += 1
        
    
    
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
    