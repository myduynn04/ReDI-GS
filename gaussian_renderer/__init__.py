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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh



def render(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0,
           override_color = None, white_bg = False, disable_dropout: bool = False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!

    [CRSGaussian Track B] disable_dropout=True: bỏ qua toàn bộ dropout block
    dù pipe.use_dropout=True. Dùng cho eval/pseudo/GUI path để đảm bảo
    render đầy đủ (không mất Gaussian random). Training view dùng default
    disable_dropout=False để dropout hoạt động theo pipe flags.
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if min(pc.bg_color.shape) != 0:
        # print("bg_color set to 000")
        bg_color = torch.tensor([0., 0., 0.]).cuda()

    # ── [CRSGaussian Combined Dropout] Gate cho B1 (Track B) + DropAnSH ──
    # 2 regularization độc lập, có thể combine:
    #   B1:       per-Gaussian uniform/sh_norm/hybrid dropout (fine-grained)
    #   DropAnSH: anchor + k-NN cluster dropout (spatial) + SH degree dropout
    # Logic: compute cả 2 masks trên full N → AND → slice once.
    # SH degree dropout modify _features_rest in-place → snapshot để restore sau rasterizer.
    N_full = pc.get_xyz.shape[0]
    dropansh_snapshot = None  # để restore SH rest coefficients sau forward

    apply_b1 = (
        getattr(pipe, "use_dropout", False)
        and (not disable_dropout)
        and getattr(pipe, "current_iter", 0) >= getattr(pipe, "dropout_start_iter", 0)
    )
    apply_dropansh = (
        getattr(pipe, "use_dropansh", False)
        and (not disable_dropout)
    )
    # apply_dropout = bất kỳ regularization nào active → cần slice tensors.
    apply_dropout = apply_b1 or apply_dropansh

    if apply_dropout:
        # Start với mask toàn 1 (giữ tất cả)
        keep_mask = torch.ones(N_full, device=pc.get_xyz.device, dtype=torch.bool)

        if apply_b1:
            from utils.regularizer.sh_dropout import compute_dropout_mask
            b1_mask, _drop_prob = compute_dropout_mask(
                pc,
                mode=getattr(pipe, "dropout_mode", "uniform"),
                base=getattr(pipe, "dropout_base", 0.1),
                w_crs=getattr(pipe, "dropout_w_crs", 0.0),
                w_sh=getattr(pipe, "dropout_w_sh", 0.0),
                max_drop=getattr(pipe, "dropout_max", 0.6),
            )
            keep_mask &= b1_mask

        if apply_dropansh:
            from utils.regularizer.dropansh import (
                anchor_dropout_mask, sh_degree_dropout
            )
            # ── [CRSGaussian Phase 26 C1] Density-as-frequency p_sh modulation ──
            # Thay p_sh scalar (đều tay mọi Gaussian) bằng per-Gaussian array
            # điều biến theo mật độ+scale cục bộ. Default OFF → p_sh scalar
            # như cũ (byte-identical baseline).
            _dropansh_psh = getattr(pipe, "dropansh_psh", 0.2)
            if getattr(pipe, "use_density_freq_modulate", False):
                from utils.regularizer.density_freq_modulate import (
                    compute_frequency_signal, modulate_probability
                )
                _freq = compute_frequency_signal(
                    pc, method=getattr(pipe, "density_freq_method", "voxel"),
                )
                _dropansh_psh = modulate_probability(
                    _dropansh_psh, _freq,
                    strength=getattr(pipe, "density_freq_strength", 1.0),
                    min_prob_ratio=getattr(pipe, "density_freq_min_ratio", 0.0),
                )
            # SH degree dropout: modify _features_rest IN-PLACE → lưu snapshot.
            dropansh_snapshot = sh_degree_dropout(
                pc,
                iteration=getattr(pipe, "current_iter", 0),
                p_sh=_dropansh_psh,
                schedule=(
                    getattr(pipe, "dropansh_sched0", 2000),
                    getattr(pipe, "dropansh_sched1", 4000),
                    getattr(pipe, "dropansh_sched2", 6000),
                ),
                crs_modulated=getattr(pipe, "dropansh_crs_sh", False),
            )
            # Anchor cluster dropout mask (spatial)
            anchor_mask = anchor_dropout_mask(
                pc,
                iteration=getattr(pipe, "current_iter", 0),
                total_iter=getattr(pipe, "dropansh_total_iter", 10000),
                k=getattr(pipe, "dropansh_k", 10),
                pa_max=getattr(pipe, "dropansh_pa", 0.02),
                crs_guided=getattr(pipe, "dropansh_crs_anchor", False),
            )
            keep_mask &= anchor_mask

        # Nếu không có Gaussian nào bị drop (rare edge case) → set None để skip slicing path
        if keep_mask.all():
            keep_mask = None
            apply_dropout = False
    else:
        keep_mask = None

    # Confidence: slice nếu dropout applied (custom CRSGaussian field, size N).
    confidence = pc.confidence if pipe.use_confidence else torch.ones_like(pc.confidence)
    if apply_dropout:
        confidence = confidence[keep_mask]
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = bg_color, #torch.tensor([1., 1., 1.]).cuda() if white_bg else torch.tensor([0., 0., 0.]).cuda(), #bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        confidence=confidence
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # ── [CRSGaussian Hướng D MVP] Layer 3: Differentiable α-coupling ──
    # Khi RNRC active mode "full" và _crs_rnrc đã set bởi train.py:
    #   α_eff = α × CRS.detach()  → CRS gates rendering signal trong forward path.
    #   Gradient flow back qua opacity raw (CRS detached), nên opacity tự update
    #   theo CRS-weighted error → mechanism khác hoàn toàn post-hoc gate.
    # Khi flag OFF hoặc mode khác hoặc _crs_rnrc chưa init → opacity = raw (no change).
    if (getattr(pc, "_rnrc_l3_active", False)
            and hasattr(pc, "_crs_rnrc")
            and pc._crs_rnrc.shape[0] == opacity.shape[0]):
        crs_gate = pc._crs_rnrc.detach().unsqueeze(-1)  # (N, 1) match opacity shape
        opacity = opacity * crs_gate

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # ── [CRSGaussian Track B] Index tất cả tensor theo keep_mask ──
    # Pattern Co-Adapt renderer:98-108. Phải slice sau khi tensors đã ready,
    # trước khi gọi rasterizer. None-check vì các nhánh có thể để None.
    if apply_dropout:
        means3D = means3D[keep_mask]
        means2D = means2D[keep_mask]
        opacity = opacity[keep_mask]
        if scales is not None:
            scales = scales[keep_mask]
        if rotations is not None:
            rotations = rotations[keep_mask]
        if shs is not None:
            shs = shs[keep_mask]
        if colors_precomp is not None:
            colors_precomp = colors_precomp[keep_mask]
        if cov3D_precomp is not None:
            cov3D_precomp = cov3D_precomp[keep_mask]

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth, alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # ── [CRSGaussian DropAnSH] Restore _features_rest sau forward ──
    # SH degree dropout đã zero some coefficients in-place để forward pass.
    # Giờ rasterizer done → phải restore để backward + subsequent renders đúng.
    # No-op nếu snapshot=None (DropAnSH off hoặc iter ≥ schedule[2] hoặc p_sh=0).
    if dropansh_snapshot is not None:
        from utils.regularizer.dropansh import restore_sh_dropout
        restore_sh_dropout(pc, dropansh_snapshot)

    if min(pc.bg_color.shape) != 0:
        rendered_image = rendered_image + (1 - alpha) * torch.sigmoid(pc.bg_color)  # torch.ones((3, 1, 1)).cuda()

    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    color = torch.clamp_min(sh2rgb + 0.5, 0.0)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # [CRSGaussian Track B] dropout_mask: full-size (N,) bool khi apply_dropout=True,
    # None khi không drop. Caller (train.py) dùng để reconstruct full-size
    # visibility_filter cho densification bookkeeping (pattern Co-Adapt train.py:187-190).
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth,
            "alpha": alpha,
            "opacity": opacity,
            "color": color,
            "dropout_mask": keep_mask}