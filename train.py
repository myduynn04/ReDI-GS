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
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import math
import torchvision
import numpy as np
import matplotlib.cm as cm
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim, loss_photometric, pearson_depth_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import time
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
import random

from utils.visualization_utils import depth2image, visualize_cmap

import kmeans1d
import open3d as o3d

import copy

# ── [CRSGaussian] Depth prior: DAV2 + COLMAP alignment ──
from utils.depth import precompute_depth_priors, align_depth_to_colmap
# ── [CRSGaussian T2.6] CRS module ──
from utils.crs import update_crs
# ── [CRSGaussian Hướng D MVP] RNRC compute helpers ──
from utils.crs.crs_module import compute_render_contribution, compute_crs_rnrc
# ── [CRSGaussian] CRS diagnostics ──
from utils.crs.crs_diagnostics import (
    log_di_ri_scatter, log_crs_stability,
    render_crs_heatmap, log_dav2_noise, log_pruning_stats
)
# [CRSGaussian Tier A] Formula-level diagnostics — gated bởi --tier_a_diag
from utils.crs.tier_a_diag import (
    tier_a_dump_distributions,
    tier_a_synthetic_floater_test,
    tier_a_occlusion_test,
)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def training(dataset, opt, pipe, args):
    # implenmetation of more than 2 3d gaussian radiance fields currently is not supported in this code
    assert args.gaussiansN >= 1 and args.gaussiansN <=2
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    print(f"scene.bounds is {scene.bounds}")
    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    GsDict = {}
    for i in range(args.gaussiansN):
        if i == 0:
            GsDict[f"gs{i}"] = gaussians
        elif i > 0:
            GsDict[f"gs{i}"] = GaussianModel(args)
            GsDict[f"gs{i}"].create_from_pcd(scene.init_point_cloud, scene.cameras_extent)
            GsDict[f"gs{i}"].training_setup(opt)
            print(f"Create gaussians{i}")
    print(f"GsDict.keys() is {GsDict.keys()}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    viewpoint_stack, pseudo_stack = None, None
    pseudo_stack_co = None

    allCameras = scene.getTrainCameras().copy()

    # ── [CRSGaussian T1.3] Precompute depth priors + align với COLMAP ──
    # Chạy 1 lần trước training loop. Gated bởi --use_depth_prior flag.
    # Khi False: training chạy như CoR-GS gốc, không tốn thời gian DAV2.
    # aligned_depth_dict: {cam.uid: Tensor (H,W)} — metric-scale depth
    # depth_range: float — dùng cho epsilon_depth và normalize D_i
    aligned_depth_dict, depth_range = {}, 1.0
    if dataset.use_depth_prior:
        depth_prior_dict = precompute_depth_priors(
            allCameras, dataset.dav2_path, encoder=dataset.dav2_encoder)
        aligned_depth_dict, depth_range = align_depth_to_colmap(
            depth_prior_dict, allCameras, dataset.source_path, dataset.n_views)
        del depth_prior_dict  # không cần nữa, chỉ giữ aligned version
        # [CRSGaussian] DIAG 4 — DAV2 noise vs SfM (chạy 1 lần)
        log_dav2_noise(aligned_depth_dict, allCameras,
                       dataset.source_path, dataset.n_views,
                       depth_range, tb_writer)

    # ── [CRSGaussian Phase 17 — C1] One-shot load DSINE normal priors ──
    # Gated --use_c1_normal. Key theo image STEM (preprocess là process
    # riêng → cam.uid không ổn định cross-process; stem ảnh ổn định).
    # Default OFF → c1_normal_dict={} → hook skip → A3/Phase-13
    # byte-identical (Quy tắc 11). HONEST: C1a = adaptation, xem
    # utils/loss/c1_normal.py (dn-splatter detach depth + supervise C1b).
    c1_normal_dict = {}
    if dataset.use_c1_normal:
        _c1dir = os.path.join(dataset.source_path, dataset.c1_normal_dir)
        for _cam in allCameras:
            _stem = os.path.basename(_cam.image_name).split(".")[0]
            _fp = os.path.join(_c1dir, _stem + ".npy")
            if os.path.isfile(_fp):
                c1_normal_dict[_cam.uid] = torch.from_numpy(
                    np.load(_fp)).float().cuda()         # (H,W,3) [0,1]
        print(f"[Phase 17 C1] loaded {len(c1_normal_dict)}/"
              f"{len(allCameras)} DSINE normal maps from {_c1dir}")

    # ── [CRSGaussian Phase 11 Step 1] Precompute covisibility maps ──
    # One-shot: forward-warp aligned depth_A → depth_B cho mỗi cặp (A, B),
    # đếm số views consistent at each pixel của A. Camera poses fixed →
    # cov_maps không cần update trong loop. Reuse aligned_depth_dict.
    # Default OFF → cov_maps=None → block compose (Section B) skip.
    cov_maps = None
    if dataset.use_coreliability_reweight and dataset.use_depth_prior:
        from utils.loss.covisibility_depth import compute_covisibility_maps
        cov_maps = compute_covisibility_maps(
            cameras=allCameras,
            aligned_depths=aligned_depth_dict,
            depth_consistency_thr=dataset.coreliability_depth_consistency_thr,
        )
        n_views_total = len(allCameras)
        n_cov = sum(int((m > 0).sum().item()) for m in cov_maps.values())
        n_total_pixels = sum(int(m.numel()) for m in cov_maps.values())
        cov_frac = n_cov / max(n_total_pixels, 1)
        print(f"[Phase 11 Step 1] cov_maps: {len(cov_maps)} cams, "
              f"max_cov={n_views_total - 1}, "
              f"covisible_pixel_frac={cov_frac:.3f}")


    # ── [CRSGaussian] Collect eval results cho summary table cuối training ──
    eval_history = []

    # ── [CRSGaussian] Timing ──
    train_start_time = time.time()
    crs_update_time_total = 0.0   # tích lũy thời gian update_crs()
    densify_time_total = 0.0      # tích lũy thời gian densification

    ema_loss_for_log = 0.0
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    # [CRSGaussian Track B] GUI render: disable_dropout để tránh flicker
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer,
                                       disable_dropout=True)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            for i in range(args.gaussiansN):
                GsDict[f"gs{i}"].oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        gt_image = viewpoint_cam.original_image.cuda()

        if 'DTU' in scene.source_path:
            if 'scan110' not in scene.source_path:
                bg_mask = (gt_image.max(0, keepdim=True).values < 30/255)
            else:
                bg_mask = (gt_image.max(0, keepdim=True).values < 15/255)
            bg_mask_clone = bg_mask.clone()
            for i in range(1, 50):
                bg_mask[:, i:] *= bg_mask_clone[:, :-i]
            gt_image[bg_mask.repeat(3,1,1)] = 0.
        else:
            bg_mask = None

        RenderDict = {}
        LossDict = {}
        logDict = {}

        # render for main viewpoint
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # ── [CRSGaussian Track B] Pass iteration context cho dropout gating ──
        # Renderer dùng pipe.current_iter để gate với dropout_start_iter.
        # Training view dùng disable_dropout=False (default) → dropout theo pipe flags.
        pipe.current_iter = iteration

        for i in range(args.gaussiansN):
            RenderDict[f"render_pkg_gs{i}"] = render(viewpoint_cam, GsDict[f'gs{i}'], pipe, bg)
            RenderDict[f"image_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["render"]
            RenderDict[f"depth_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["depth"]
            RenderDict[f"alpha_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["alpha"]
            RenderDict[f"viewspace_point_tensor_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["viewspace_points"]
            RenderDict[f"visibility_filter_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["visibility_filter"]
            RenderDict[f"radii_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["radii"]
            RenderDict[f"dropout_mask_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["dropout_mask"]

        # ── [CRSGaussian Track B+DropAnSH] Log combined drop rate ──
        # Combined mask = B1 AND DropAnSH. Log cả 2 flags để biết source.
        if getattr(pipe, "use_dropout", False) or getattr(pipe, "use_dropansh", False):
            _dm = RenderDict.get("dropout_mask_gs0", None)
            if _dm is not None:
                _drop_rate_actual = 1.0 - _dm.float().mean().item()
                if tb_writer is not None and iteration % 500 == 0:
                    tb_writer.add_scalar('dropout/actual_drop_rate', _drop_rate_actual, iteration)
                    tb_writer.add_scalar('dropout/n_kept', int(_dm.sum().item()), iteration)
                    tb_writer.add_scalar('dropout/n_total', _dm.shape[0], iteration)
                if iteration % 1000 == 0:
                    _src = []
                    if getattr(pipe, "use_dropout", False): _src.append("B1")
                    if getattr(pipe, "use_dropansh", False): _src.append("DropAnSH")
                    print(f"[DIAG Dropout {'+'.join(_src)}] iter={iteration} "
                          f"drop_rate={_drop_rate_actual:.3f} "
                          f"kept={int(_dm.sum().item())}/{_dm.shape[0]}")

        # Loss
        for i in range(args.gaussiansN):
            image_i = RenderDict[f"image_gs{i}"]

            # ── [CRSGaussian Tier 2-min] Loss reweighter — per-pixel CRS weight ──
            # Gated: --use_loss_reweight AND iter ≥ d_cycle_warmup (reuse warmup
            # vì cùng prerequisite: CRS đã ổn định) AND i==0 (CRS gắn vào gs0).
            # Default OFF → weight_map=None → fall back loss_photometric cũ.
            # Cache: per-cam dict, refresh mỗi lossw_render_freq iter của CAM ĐÓ
            #         (mỗi cam được train ~1/N_cam số iter → effective refresh
            #          ≈ N_cam × lossw_render_freq global iters).
            weight_map = None
            if (opt.use_loss_reweight
                    and iteration >= opt.d_cycle_warmup
                    and i == 0):
                gs0 = GsDict["gs0"]
                if not hasattr(gs0, "_crs_map_cache_dict"):
                    gs0._crs_map_cache_dict = {}
                    gs0._crs_map_render_iter = {}
                cam_uid = viewpoint_cam.uid
                last_iter = gs0._crs_map_render_iter.get(cam_uid, -10**9)
                # Re-render khi: (a) chưa cache cho cam này, (b) đã quá freq iters,
                # (c) shape mismatch (densify/prune đổi N_gauss → phải refresh).
                cached = gs0._crs_map_cache_dict.get(cam_uid)
                need_recompute = (
                    cached is None
                    or (iteration - last_iter) >= opt.lossw_render_freq
                )
                if need_recompute:
                    from utils.crs.crs_module import render_crs_map as _rcm
                    # Dùng background (deterministic) thay vì bg (có thể random)
                    # để CRS map ổn định giữa các iter.
                    _crs_map = _rcm(gs0, viewpoint_cam, render, pipe, background)
                    gs0._crs_map_cache_dict[cam_uid] = _crs_map.detach()
                    gs0._crs_map_render_iter[cam_uid] = iteration
                    cached = gs0._crs_map_cache_dict[cam_uid]
                # w(p) = γ + (1-γ) · CRS_pix(p), clamp [γ, 1.0]
                # γ=0.5 default → w ∈ [0.5, 1.0]: pixel reliable giữ loss đầy đủ,
                # pixel unreliable giảm loss tối đa 50%, KHÔNG triệt tiêu (gradient
                # vẫn flow → late-bloomer recoverable, không như gate prune cứng).
                weight_map = (
                    opt.lossw_gamma + (1.0 - opt.lossw_gamma) * cached
                ).clamp(opt.lossw_gamma, 1.0)

            # ── [CRSGaussian Phase 11 Step 1] Covisibility-based weight ──
            # Override / set weight_map từ cov_maps + (optional) CRS_pix combine.
            # Gated: use_coreliability_reweight, cov_maps đã build, áp dụng cho gs0.
            # Nếu Phase 7 (use_loss_reweight) cũng bật, Phase 11 OVERRIDE — tránh
            # double-reweight. Diagnostic A1 chạy Phase 11 standalone (Phase 7 OFF).
            if (dataset.use_coreliability_reweight
                    and cov_maps is not None
                    and viewpoint_cam.uid in cov_maps
                    and i == 0):
                from utils.loss.covisibility_depth import compute_reliability_weight
                n_others = max(len(scene.getTrainCameras()) - 1, 1)
                cov_norm = cov_maps[viewpoint_cam.uid].float() / n_others  # (H, W) ∈ [0, 1]

                crs_pix_for_weight = None
                if dataset.coreliability_combine_crs:
                    # Reuse Phase 7 cache nếu có sẵn (rerender expensive),
                    # else render fresh CRS map.
                    gs0 = GsDict["gs0"]
                    cam_uid = viewpoint_cam.uid
                    if (hasattr(gs0, "_crs_map_cache_dict")
                            and cam_uid in gs0._crs_map_cache_dict):
                        crs_pix_for_weight = gs0._crs_map_cache_dict[cam_uid]
                    else:
                        from utils.crs.crs_module import render_crs_map as _rcm
                        crs_pix_for_weight = _rcm(
                            gs0, viewpoint_cam, render, pipe, background)

                weight_map = compute_reliability_weight(
                    cov_norm, crs_pix_for_weight,
                    gamma=dataset.coreliability_gamma,
                    combine_mode=dataset.coreliability_combine_mode,
                )
                # ── Section C: TB log ──
                if (tb_writer is not None and iteration % 100 == 0):
                    tb_writer.add_scalar(
                        'phase11s1/mean_weight',
                        float(weight_map.mean().item()),
                        iteration)

            if not bg_mask is None:
                valid = (~bg_mask).float()
                if weight_map is not None:
                    # Combined: valid binary × weight continuous.
                    # Manual weighted L1 — chia theo sum(weights) để giữ scale
                    # tương thích l1_loss baseline (mean over weighted region).
                    w3 = (weight_map.unsqueeze(0) * valid).expand(3, -1, -1)
                    Ll1 = (w3 * torch.abs(image_i - gt_image)).sum() / (w3.sum() + 1e-6)
                    # SSIM dùng valid mask binary (không weighted) — SSIM per-window
                    # khó kết hợp continuous weight, giữ tiêu chuẩn để không bias.
                    L_phot = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image_i, gt_image, mask=valid))
                else:
                    L_phot = loss_photometric(image_i, gt_image, opt=opt, valid=valid)
                LossDict[f"loss_gs{i}"] = L_phot + (RenderDict[f"alpha_gs{i}"][bg_mask]**2).mean()
            else:
                if weight_map is not None:
                    w3 = weight_map.unsqueeze(0).expand(3, -1, -1)
                    Ll1 = (w3 * torch.abs(image_i - gt_image)).sum() / (w3.sum() + 1e-6)
                    LossDict[f"loss_gs{i}"] = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image_i, gt_image))
                else:
                    LossDict[f"loss_gs{i}"] = loss_photometric(image_i, gt_image, opt=opt)


        # ── [CRSGaussian T3.3] Fixed Pearson depth loss ──
        # Gated bởi --use_depth_prior: không ảnh hưởng baseline khi tắt.
        # Depth loss = correction: kéo Gaussians về đúng depth.
        # CRS = elimination: prune floater (Phase 4). Hai cơ chế tách biệt.
        if dataset.use_depth_prior and viewpoint_cam.uid in aligned_depth_dict:
            rendered_depth = RenderDict["depth_gs0"]              # (1,H,W) GPU
            depth_prior = aligned_depth_dict[viewpoint_cam.uid]   # (H,W) GPU
            L_depth = 0.05 * pearson_depth_loss(rendered_depth, depth_prior)
            LossDict["loss_gs0"] += L_depth

        # ── [CRSGaussian Phase 17 — C1] DSINE normal-prior loss (C1a) ──
        # Gated --use_c1_normal AND iter≥start AND có normal map cho cam.
        # Pattern y hệt L_depth. Default OFF → không cộng → A3 baseline.
        # HONEST: C1a deviation (dn-splatter detach depth + CUDA C1b);
        # pre-registered prediction (gradient depth→normal nhiễu) ở
        # utils/loss/c1_normal.py. λ=c1_normal_lambda (chính 0.10).
        if (dataset.use_c1_normal
                and iteration >= dataset.c1_normal_start_iter
                and viewpoint_cam.uid in c1_normal_dict):
            from utils.loss.c1_normal import compute_c1_normal_loss
            _W = viewpoint_cam.image_width
            _H = viewpoint_cam.image_height
            _fx = _W / (2.0 * math.tan(viewpoint_cam.FoVx * 0.5))
            _fy = _H / (2.0 * math.tan(viewpoint_cam.FoVy * 0.5))
            L_c1 = compute_c1_normal_loss(
                RenderDict["depth_gs0"], RenderDict["alpha_gs0"],
                c1_normal_dict[viewpoint_cam.uid],
                _fx, _fy, _W / 2.0, _H / 2.0, dataset.c1_normal_lambda)
            if L_c1 is not None:
                LossDict["loss_gs0"] += L_c1

        # ── [CRSGaussian Phase 26 A2] Temporal parameter regularization ──
        # L_temporal so tham số 3D hiện tại (_xyz/_scaling/_rotation, TRƯỚC
        # backward của iteration này) với EMA của chính nó từ các iteration
        # trước — loss tác động trực tiếp lên không gian tham số 3D, khác
        # L1/D-SSIM/depth (đều so sánh trên pixel 2D). Gọi compute TRƯỚC
        # update EMA để EMA phản ánh lịch sử "quá khứ" chứ không lẫn giá trị
        # vừa tính loss trên nó. Default OFF (use_temporal_reg=False) → 0
        # overhead, byte-identical baseline.
        if opt.use_temporal_reg and iteration >= opt.temporal_reg_start_iter:
            from utils.loss.temporal_reg import compute_temporal_reg_loss, update_temporal_ema
            L_temporal = compute_temporal_reg_loss(
                gaussians,
                lambda_xyz=opt.lambda_temporal_xyz,
                lambda_shape=opt.lambda_temporal_shape,
                crs_weighted=opt.temporal_crs_weighted,
            )
            if L_temporal is not None:
                LossDict["loss_gs0"] += L_temporal
            update_temporal_ema(gaussians, beta=opt.temporal_ema_beta)

        loss = LossDict["loss_gs0"]
        for i in range(args.gaussiansN):
            LossDict[f"loss_gs{i}"].backward()

        if args.save_log_images and (iteration % 100 == 0):
            with torch.no_grad():
                eval_cam = allCameras[random.randint(0, len(allCameras) -1)]
                
                # [CRSGaussian Track B] Eval render (log images): disable_dropout=True
                render_results = render(eval_cam, GsDict[f'gs0'], pipe, bg, disable_dropout=True)
                image = torch.clamp(render_results["render"], 0.0, 1.0)
                gt_image = torch.clamp(eval_cam.original_image.to("cuda"), 0.0, 1.0)
                black = torch.zeros_like(gt_image).to(gt_image.device)
                render_depth = render_results["depth"]
                render_depth_image = depth2image(render_depth, inverse=True, rgb=True)
                render_opacity_image = render_results["alpha"].repeat(3, 1, 1)

                if args.gaussiansN > 1:
                    # [CRSGaussian Track B] Eval render gs1: disable_dropout=True
                    render_results_gs1 = render(eval_cam, GsDict[f'gs1'], pipe, bg, disable_dropout=True)
                    image_gs1 = torch.clamp(render_results_gs1["render"], 0.0, 1.0)
                    render_depth_gs1 = render_results_gs1["depth"]
                    render_depth_image_gs1 = depth2image(render_depth_gs1, inverse=True, rgb=True)
                    render_opacity_image_gs1 = render_results_gs1["alpha"].repeat(3, 1, 1)

            row0 = torch.cat([gt_image, black, black], dim=2)
            row1 = torch.cat([image, render_depth_image, render_opacity_image], dim=2)
            if args.gaussiansN > 1:
                row2 = torch.cat([image_gs1, render_depth_image_gs1, render_opacity_image_gs1], dim=2)
            else:
                row2 = torch.cat([black, black, black], dim=2)

            image_to_show = torch.cat([row0, row1, row2], dim=1)
            image_to_show = torch.clamp(image_to_show, 0, 1)
            
            os.makedirs(f"{dataset.model_path}/log_images_train", exist_ok = True)
            torchvision.utils.save_image(image_to_show, f"{dataset.model_path}/log_images_train/{iteration}.jpg")

        with torch.no_grad():
            # Progress bar — hiện Loss + #Gaussians + speed
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                n_gs = gaussians.get_xyz.shape[0]
                elapsed = time.time() - train_start_time
                it_per_s = iteration / max(elapsed, 1e-6)
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.7f}",
                    "N": f"{n_gs//1000}k",
                    "it/s": f"{it_per_s:.1f}"
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # ── [CRSGaussian] Periodic stats log — mỗi 500 iter ──
            if iteration % 500 == 0:
                n_gs = gaussians.get_xyz.shape[0]
                elapsed = time.time() - train_start_time
                print(f"\n[STATS] iter={iteration} | "
                      f"N={n_gs} | "
                      f"elapsed={elapsed:.1f}s | "
                      f"it/s={iteration/max(elapsed,1e-6):.1f}")

            training_report(args, tb_writer, iteration, loss, l1_loss,
                            testing_iterations, scene, render, (pipe, background),
                            GsDict=GsDict, eval_history=eval_history)

            if iteration > first_iter and (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration > first_iter and (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((GsDict["gs0"].capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # ── [CRSGaussian T2.6] CRS update — LOG ONLY ──
            # Tính D_i + R_i → update _crs_score bằng EMA.
            # Gated: --use_depth_prior AND sau T_warmup AND mỗi 100 iter.
            # KHÔNG dùng CRS cho loss/pruning ở bước này — chỉ log stats
            # để verify signal trước khi bật (T3.x, T4.x).
            if (dataset.use_depth_prior
                    and iteration > opt.T_warmup
                    and iteration % opt.crs_update_interval == 0):
                _t0 = time.time()
                # ── [CRSGaussian Tier 2-min] D signal switch ──
                # Khi --use_d_cycle bật, update_crs sẽ thay D_DAV2 bằng D_cycle
                # (cycle-depth qua training views, no DAV2). Cần render_func +
                # pipe + background để render N_cams depth maps cho cycle test.
                # Cache trong gaussians._d_cycle_cache, refresh mỗi
                # opt.d_cycle_update_freq iter (default 100, trùng CRS interval).
                D_diag, R_diag = update_crs(
                    gaussians, allCameras, aligned_depth_dict,
                    depth_range, ema=opt.crs_ema_decay,
                    use_d_cycle=opt.use_d_cycle,
                    iter=iteration,
                    d_cycle_warmup=opt.d_cycle_warmup,
                    d_cycle_sigma=opt.d_cycle_sigma,
                    d_cycle_update_freq=opt.d_cycle_update_freq,
                    render_func=render,
                    pipe=pipe,
                    bg=background,
                    # ── [CRSGaussian Phase 8b] S_stability ──
                    use_sh_reliability=opt.use_sh_reliability,
                    sh_stability_warmup=opt.sh_stability_warmup,
                    sh_stability_ema_beta=opt.sh_stability_ema_beta,
                    crs_w_s=opt.crs_w_s,
                    # ── [CRSGaussian Phase 9] D-only formula ──
                    disable_r_signal=opt.disable_r_signal,
                    # ── [CRSGaussian Phase 26 A1] V_stability ──
                    use_v_stability=opt.use_v_stability,
                    v_stability_warmup=opt.v_stability_warmup,
                    crs_w_v=opt.crs_w_v,
                )
                crs_update_time_total += time.time() - _t0
                # Log CRS distribution
                crs_vals = gaussians.get_crs.detach()
                if iteration % 500 == 0:
                    print(f"\n[CRS] iter={iteration} | "
                          f"N={crs_vals.shape[0]} | "
                          f"mean={crs_vals.mean():.4f} | "
                          f"std={crs_vals.std():.4f} | "
                          f"min={crs_vals.min():.4f} | "
                          f"max={crs_vals.max():.4f} | "
                          f"<0.35={( crs_vals < 0.35).sum().item()} | "
                          f">0.65={(crs_vals > 0.65).sum().item()}")

                # ── [CRSGaussian Tier 2-min] Logging — D_cycle + Loss reweighter stats ──
                # Print compact line mỗi 1000 iter khi flag bật. Giúp debug
                # khi compare ablation: D distribution, weight map mean, N_gauss.
                if (opt.use_d_cycle or opt.use_loss_reweight) and iteration % 1000 == 0:
                    _msg = f"[T2min] iter={iteration} N={gaussians.get_xyz.shape[0]}"
                    if opt.use_d_cycle:
                        _msg += (
                            f" | D_med={D_diag.median().item():.3f}"
                            f" D_p10={torch.quantile(D_diag, 0.1).item():.3f}"
                            f" D_p90={torch.quantile(D_diag, 0.9).item():.3f}"
                        )
                    if opt.use_loss_reweight:
                        # Weight map mean across cached cams (proxy cho overall reweight strength)
                        _gs0 = GsDict["gs0"]
                        if hasattr(_gs0, "_crs_map_cache_dict") and _gs0._crs_map_cache_dict:
                            _w_means = []
                            for _m in _gs0._crs_map_cache_dict.values():
                                _w_means.append(_m.mean().item())
                            if _w_means:
                                _w_avg = sum(_w_means) / len(_w_means)
                                _msg += f" | CRSmap_mean={_w_avg:.3f} cached_cams={len(_w_means)}"
                    print(_msg)

                # [CRSGaussian] DIAG 2 — CRS stability (mỗi CRS update)
                log_crs_stability(gaussians, D_diag, R_diag, iteration, tb_writer)

                # [CRSGaussian] DIAG 1 — Scatter D vs R (tại milestones)
                diag_dir = f"{dataset.model_path}/crs_diag"
                if iteration in [1100, 2000, 5000]:
                    log_di_ri_scatter(D_diag, R_diag, iteration, tb_writer, diag_dir)

                # [CRSGaussian] DIAG 3 — CRS heatmap (tại milestones)
                if iteration in [1100, 3000, 5000, 10000]:
                    render_crs_heatmap(gaussians, allCameras[0],
                                       render, pipe, background,
                                       iteration, diag_dir)

                # ── [CRSGaussian Tier A] Formula diagnostics ──
                # Gated bởi --tier_a_diag. Chỉ dump tại các iter milestone
                # quy định trong --tier_a_diag_iters (default 1100,3000,5000,10000).
                # Read-only — không sửa training behavior.
                if dataset.tier_a_diag:
                    try:
                        tier_a_iters = [int(x) for x in dataset.tier_a_diag_iters.split(',')]
                    except Exception:
                        tier_a_iters = [1100, 3000, 5000, 10000]
                    if iteration in tier_a_iters:
                        tier_a_dir = f"{dataset.model_path}/tier_a_diag"
                        # A1: dump D, R, CRS, opacity tensors
                        tier_a_dump_distributions(
                            gaussians, D_diag, R_diag, iteration, tier_a_dir
                        )
                        # A3: synthetic floater discrimination
                        tier_a_synthetic_floater_test(
                            gaussians, allCameras, aligned_depth_dict, depth_range,
                            iteration, tier_a_dir,
                        )
                        # A4: occlusion contamination
                        tier_a_occlusion_test(
                            gaussians, allCameras, render, pipe, background,
                            iteration, tier_a_dir,
                        )

            # Densification
            if  iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                for i in range(args.gaussiansN):
                    viewspace_point_tensor = RenderDict[f"viewspace_point_tensor_gs{i}"]
                    visibility_filter = RenderDict[f"visibility_filter_gs{i}"]
                    radii = RenderDict[f"radii_gs{i}"]
                    dropout_mask = RenderDict[f"dropout_mask_gs{i}"]

                    # ── [CRSGaussian Track B] Reconstruct full-size visibility ──
                    # Khi dropout active: visibility_filter + radii có size N_keep
                    # (chỉ Gaussian được giữ), nhưng max_radii2D size N (full).
                    # Pattern Co-Adapt train.py:187-190: reconstruct combined_mask
                    # = (Gaussian kept) AND (visible) → index vào full-size array.
                    if dropout_mask is not None:
                        true_indices = torch.nonzero(dropout_mask, as_tuple=True)[0]     # (N_keep,)
                        filtered_indices = true_indices[visibility_filter]               # (N_visible_kept,)
                        combined_mask = torch.zeros_like(dropout_mask, dtype=torch.bool)  # (N,)
                        combined_mask[filtered_indices] = True
                        GsDict[f"gs{i}"].max_radii2D[combined_mask] = torch.max(
                            GsDict[f"gs{i}"].max_radii2D[combined_mask], radii[visibility_filter])
                        GsDict[f"gs{i}"].add_densification_stats(viewspace_point_tensor, combined_mask)
                    else:
                        # No dropout — original path (size N đồng bộ).
                        GsDict[f"gs{i}"].max_radii2D[visibility_filter] = torch.max(
                            GsDict[f"gs{i}"].max_radii2D[visibility_filter], radii[visibility_filter])
                        GsDict[f"gs{i}"].add_densification_stats(viewspace_point_tensor, visibility_filter)
            
                # density and prune
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    _t0 = time.time()
                    size_threshold = None
                    # size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    # [CRSGaussian T4.1] Truyền depth constraint params.
                    # Khi --use_depth_prior: position constraint chặn floater sinh ra.
                    # [CRSGaussian] Position constraint gated bởi --use_pos_constraint
                    # Cần --use_depth_prior làm prerequisite
                    _pc_cams = allCameras if (dataset.use_depth_prior and opt.use_pos_constraint) else None
                    _pc_depth = aligned_depth_dict if (dataset.use_depth_prior and opt.use_pos_constraint) else None
                    _pc_range = depth_range if (dataset.use_depth_prior and opt.use_pos_constraint) else None

                    # ── [CRSGaussian Phase 13] LFCF opts builder ──
                    # is_lfcf_iter = True khi (iter % (interval_times × densify_interval) == 0).
                    # lfcf_opts dict KHÔNG include is_lfcf_iter — separate top-level flag.
                    # Default OFF (use_lfcf=False) → is_lfcf_iter_now=False → standard path.
                    is_lfcf_iter_now = False
                    lfcf_opts = None
                    cameras_for_lfcf = None
                    if opt.use_lfcf and iteration < opt.densify_until_iter:
                        is_lfcf_iter_now = (
                            iteration % (opt.lfcf_interval_times * opt.densification_interval) == 0
                        )
                        if is_lfcf_iter_now:
                            from utils.densify.lfcf import calculate_training_percent_powered
                            # Compute decay: scaling multiplier max sẽ decay từ init → last
                            # qua densify range. percent_lb = log ratio cho exact decay curve.
                            percent_lb = (
                                math.log(opt.lfcf_last_scaling_max) / math.log(opt.lfcf_init_scaling_max)
                                if opt.lfcf_init_scaling_max != 1.0 else 1.0
                            )
                            tpp = calculate_training_percent_powered(
                                iteration, opt.densify_from_iter, opt.densify_until_iter,
                                opt.lfcf_pow, percent_lb,
                            )
                            # splitting_lb decays linear từ 1.0 → 0 over densify range
                            # → Early training: split aggressive
                            # → Late training: split conservative (mostly enlarge)
                            splitting_lb_now = 1.0 - (iteration - opt.densify_from_iter) / max(
                                opt.densify_until_iter - opt.densify_from_iter, 1
                            )
                            lfcf_opts = {
                                'scaling_multiplier_max': opt.lfcf_init_scaling_max,
                                'scaling_multiplier_min': opt.lfcf_init_scaling_min,
                                'training_percent_powered': tpp,
                                'splitting_ub': opt.lfcf_splitting_ub,
                                'splitting_lb': splitting_lb_now,
                                'tolerance': opt.lfcf_tolerance,
                                'diffscale': opt.lfcf_diffscale,
                            }
                            cameras_for_lfcf = allCameras

                    for i in range(args.gaussiansN):
                        _n_before = GsDict[f"gs{i}"].get_xyz.shape[0]
                        GsDict[f"gs{i}"].densify_and_prune(
                            opt.densify_grad_threshold, opt.prune_threshold,
                            scene.cameras_extent, size_threshold, iteration,
                            cameras=_pc_cams,
                            aligned_depth_dict=_pc_depth,
                            depth_range=_pc_range,
                            T_warmup=opt.T_warmup,
                            tau_crs=opt.tau_crs,
                            tau_isolated=opt.tau_isolated,
                            # ── [Phase 13] LFCF kwargs (default OFF) ──
                            is_lfcf_iter=is_lfcf_iter_now,
                            lfcf_opts=lfcf_opts,
                            cameras_for_lfcf=cameras_for_lfcf)
                        _n_after = GsDict[f"gs{i}"].get_xyz.shape[0]
                        # [CRSGaussian] DIAG 5 — Pruning stats
                        if iteration % 500 == 0:
                            log_pruning_stats(_n_before, _n_after, iteration, tb_writer)
                    densify_time_total += time.time() - _t0

            # ── [CRSGaussian Hướng D MVP] RNRC update (post-densify) ──
            # Đặt SAU densify_and_prune để tránh shape mismatch giữa
            # max_radii2D / combined_mask đã tính từ render (line ~565)
            # và N sau khi RNRC prune.
            # Mode-specific:
            #   "L1g":  snapshot every 100 iter, prune at snapshot
            #   "L12g": continuous EMA every iter, prune every iter
            #   "full": continuous EMA + Layer 3 differentiable α-coupling (next iter render)
            # Gated bởi --use_rnrc (default OFF → behavior cũ).
            if opt.use_rnrc:
                if opt.rnrc_mode == "L1g":
                    do_rnrc_update = (iteration % 100 == 0)
                else:
                    do_rnrc_update = True
                if do_rnrc_update:
                    rc_thisstep = compute_render_contribution(
                        gaussians, allCameras, render, (pipe, background))
                    spawn_iter_t = getattr(gaussians, "spawn_iter", None)
                    crs_rnrc = compute_crs_rnrc(
                        gaussians, rc_thisstep, opt, iteration, spawn_iter_t)
                    gaussians._crs_rnrc = crs_rnrc
                    # Layer 3 toggle for renderer (đọc ở iter sau)
                    gaussians._rnrc_l3_active = (opt.rnrc_mode == "full")
                    # L1g/L12g: pruning gate (sau warmup)
                    if opt.rnrc_mode in ("L1g", "L12g") and iteration > opt.rnrc_warmup:
                        prune_mask_rnrc = (crs_rnrc < opt.rnrc_tau)
                        if prune_mask_rnrc.any().item():
                            gaussians.prune_points(prune_mask_rnrc, iteration)
                    # Log
                    if iteration % 1000 == 0:
                        print(f"[RNRC] iter {iteration} mode={opt.rnrc_mode}: "
                              f"RC_med={rc_thisstep.median().item():.4f}, "
                              f"CRS_med={crs_rnrc.median().item():.4f}, "
                              f"CRS_p10={crs_rnrc.quantile(0.1).item():.4f}, "
                              f"N_gauss={gaussians.get_xyz.shape[0]}")

            # ── [CRSGaussian Phase 8c] CRS-modulated SH freeze ──
            # Per-Gaussian thay thế global freeze_sh_after. Zero out
            # _features_rest.grad cho Gaussians có CRS < tau_freeze.
            # Phải gọi SAU densification + RNRC (modify Gaussian count),
            # TRƯỚC optimizer.step() để zero grad có effect.
            # Default OFF (use_crs_modulated_sh_freeze=False) → no-op.
            if opt.use_crs_modulated_sh_freeze:
                from utils.crs.sh_freeze import apply_crs_modulated_sh_freeze
                for i in range(args.gaussiansN):
                    apply_crs_modulated_sh_freeze(
                        GsDict[f"gs{i}"],
                        iter=iteration,
                        freeze_start=opt.crs_freeze_start,
                        tau_freeze=opt.crs_freeze_tau,
                    )

            # ── [CRSGaussian Phase 26 A1] CRS-modulated geometric freeze ──
            # Đối xứng block trên nhưng freeze _xyz/_scaling/_rotation grad
            # (thay vì SH) cho Gaussians có V_stability thấp — chặn hình học
            # đang trôi dạt/méo trước khi optimizer.step() áp dụng update.
            # Default OFF (use_crs_modulated_geom_freeze=False) → no-op.
            if opt.use_crs_modulated_geom_freeze:
                from utils.crs.geom_freeze import apply_crs_modulated_geom_freeze
                for i in range(args.gaussiansN):
                    apply_crs_modulated_geom_freeze(
                        GsDict[f"gs{i}"],
                        iter=iteration,
                        freeze_start=opt.geom_freeze_start,
                        tau_freeze=opt.geom_freeze_tau,
                    )

            # Optimizer step
            if iteration < opt.iterations:
                for i in range(args.gaussiansN):
                    GsDict[f"gs{i}"].optimizer.step()
                    GsDict[f"gs{i}"].optimizer.zero_grad(set_to_none = True)

            # ── [CRSGaussian Phase 8b] SH stability EMA tracking ──
            # Update EMA mean + variance của _features_rest mỗi crs_update_interval
            # iter (đồng bộ với CRS update). Gọi SAU optimizer.step() để EMA
            # capture trạng thái post-update của SH coefficients.
            # Default OFF (use_sh_reliability=False) → no-op.
            if (opt.use_sh_reliability
                    and iteration > opt.sh_stability_warmup
                    and iteration % opt.crs_update_interval == 0):
                from utils.crs.sh_stability import update_sh_stability
                for i in range(args.gaussiansN):
                    update_sh_stability(
                        GsDict[f"gs{i}"],
                        beta=opt.sh_stability_ema_beta,
                    )

            # ── [CRSGaussian Phase 26 A1] Geometric (V) stability EMA tracking ──
            # Update EMA mean + variance của _xyz/_scaling/_rotation, cùng nhịp
            # + vị trí gọi với update_sh_stability (SAU optimizer.step() để
            # capture trạng thái post-update của tham số hình học).
            # Default OFF (use_v_stability=False) → no-op.
            if (opt.use_v_stability
                    and iteration > opt.v_stability_warmup
                    and iteration % opt.crs_update_interval == 0):
                from utils.crs.v_stability import update_v_stability
                for i in range(args.gaussiansN):
                    update_v_stability(GsDict[f"gs{i}"])

            # ── [CRSGaussian Phase 2c] Opacity decay hook ──
            # Multiply opacity mỗi iter sau densify_from_iter → continuous pressure.
            # Khác CRS pruning (sparse, mỗi 100 iter): decay liên tục giúp zombie
            # Gaussian opacity giảm dần → bị loại tự nhiên bởi legacy opacity prune
            # hoặc CRS prune.
            # Gated bởi --use_opacity_decay (default False) → baseline không đổi.
            if (dataset.use_opacity_decay
                    and iteration > opt.densify_from_iter):
                # ── [CRSGaussian Phase 26 C1] Density-as-frequency decay modulation ──
                # opacity_decay_factor scalar (đều tay) → per-Gaussian factor
                # gần 1 (decay chậm) tại vùng tần số cao. Prerequisite: cả
                # dataset.opacity_decay_freq_modulate VÀ pipe.use_density_freq_modulate.
                # Default OFF → factor scalar như cũ (byte-identical baseline).
                if (dataset.opacity_decay_freq_modulate
                        and getattr(pipe, "use_density_freq_modulate", False)):
                    from utils.regularizer.density_freq_modulate import (
                        compute_frequency_signal, modulate_probability
                    )
                    for i in range(args.gaussiansN):
                        _gs = GsDict[f"gs{i}"]
                        _freq = compute_frequency_signal(
                            _gs, method=getattr(pipe, "density_freq_method", "voxel"),
                        )
                        # decay factor: 1 - (1-base_factor) * (1 - strength*freq)
                        # freq cao → factor gần 1 (decay chậm); freq thấp → factor gốc.
                        _decay_gap = 1.0 - dataset.opacity_decay_factor
                        _factor_per = 1.0 - modulate_probability(
                            _decay_gap, _freq,
                            strength=getattr(pipe, "density_freq_strength", 1.0),
                            min_prob_ratio=getattr(pipe, "density_freq_min_ratio", 0.0),
                        )
                        _gs.opacity_decay(factor=_factor_per)
                else:
                    for i in range(args.gaussiansN):
                        GsDict[f"gs{i}"].opacity_decay(factor=dataset.opacity_decay_factor)
                # One-shot extend densify — Binocular3DGS style (densify suốt training)
                if (dataset.opacity_decay_extend_densify
                        and iteration == opt.densify_from_iter + 1):
                    opt.densify_until_iter = opt.iterations

            # ── [CRSGaussian DIAG E1] Freeze SH hook ──
            # Sau iter opt.freeze_sh_after → set lr=0 cho f_dc + f_rest.
            # Gọi sau optimizer.step() để step cuối trước freeze vẫn dùng
            # lr bình thường (gradient flow vẫn có, chỉ không update tiếp).
            # freeze_sh() tự idempotent (self._sh_frozen flag) nên gọi lặp OK.
            #
            # [CRSGaussian Phase 9] disable_global_sh_freeze=True → bypass global
            # freeze hoàn toàn (cho A1B1_BEST config: thay global freeze bằng
            # CRS-modulated per-Gaussian freeze).
            if (opt.freeze_sh_after > 0
                    and iteration >= opt.freeze_sh_after
                    and not opt.disable_global_sh_freeze):
                for i in range(args.gaussiansN):
                    GsDict[f"gs{i}"].freeze_sh()

            # ── [CRSGaussian DIAG A2] Freeze DC-only hook ──
            # Chỉ freeze f_dc, f_rest vẫn update → isolate DC contribution.
            # Idempotent qua self._dc_frozen flag.
            if opt.freeze_dc_only and iteration >= opt.freeze_dc_start_iter:
                for i in range(args.gaussiansN):
                    GsDict[f"gs{i}"].freeze_dc()

            for i in range(args.gaussiansN):
                GsDict[f"gs{i}"].update_learning_rate(iteration)
                if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
                        iteration > args.start_sample_pseudo:
                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    print(f"reset opacity of gaussians-{i} at iteration {iteration}")
                    GsDict[f"gs{i}"].reset_opacity()
             

    # ── [CRSGaussian] Timing + stats summary ──
    train_elapsed = time.time() - train_start_time
    final_n = gaussians.get_xyz.shape[0]
    avg_it_s = opt.iterations / max(train_elapsed, 1e-6)
    print(f"\n[TIMING] Total training: {train_elapsed:.1f}s ({train_elapsed/60:.1f}min)")
    print(f"[TIMING] Avg speed:     {avg_it_s:.1f} it/s")
    print(f"[TIMING] CRS update:    {crs_update_time_total:.1f}s ({crs_update_time_total/max(train_elapsed,1e-6)*100:.1f}%)")
    print(f"[TIMING] Densification: {densify_time_total:.1f}s ({densify_time_total/max(train_elapsed,1e-6)*100:.1f}%)")
    print(f"[TIMING] Final #Gaussians: {final_n} ({final_n/1000:.1f}k)")

    # ── [CRSGaussian] Summary table — in kết quả tổng hợp cuối training ──
    if eval_history:
        print("\n" + "=" * 90)
        print("  SUMMARY — Evaluation Results")
        print("=" * 90)
        print(f"  {'Iter':>6} | {'Split':>5} | {'PSNR':>8} | {'SSIM':>8} | {'LPIPS':>8} | {'L1':>10} | {'#Gaussians':>11}")
        print("-" * 90)
        for row in eval_history:
            print(f"  {row['iter']:>6} | {row['split']:>5} | "
                  f"{row['psnr']:>8.4f} | {row['ssim']:>8.4f} | "
                  f"{row['lpips']:>8.4f} | {row['l1']:>10.6f} | "
                  f"{row['n_gaussians']:>11}")
        print("=" * 90)

        # Tìm best test PSNR
        test_rows = [r for r in eval_history if r['split'] == 'test']
        if test_rows:
            best = max(test_rows, key=lambda r: r['psnr'])
            print(f"  Best test PSNR: {best['psnr']:.4f} at iter {best['iter']} "
                  f"(N={best['n_gaussians']})")
        print("=" * 90 + "\n")


def prepare_output_and_logger(args):
    """
    Tạo output folder, lưu config args vào cfg_args, khởi tạo TensorBoard writer.
    Nếu model_path chưa được set, tự sinh tên từ OAR job ID (HPC) hoặc UUID.
    Trả về tb_writer (None nếu TensorBoard không có).
    """
    # Tự tạo model_path nếu chưa có 
    # model_path là đường dẫn thư mục output để lưu các kết quả huấn luyện, nếu không được cung cấp, nó sẽ tạo một thư mục mới với tên duy nhất dựa trên OAR_JOB_ID hoặc UUID.
    # Trong arguments/__init__.py, model_path được đăng ký với tên ngắn -m
    if not args.model_path: # Nếu không truyền -m
        if os.getenv('OAR_JOB_ID'): # kiểm tra có đang chạy trên HPC cluster không
            unique_str=os.getenv('OAR_JOB_ID')# có → dùng job ID làm tên
        else:
            unique_str = str(uuid.uuid4())# không → tạo chuỗi random
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    
    # tạo thư mực output và lưu config
    os.makedirs(args.model_path, exist_ok = True)
    
    # Lưu toàn bộ args vào file cfg_args trong output folder — để sau này biết run đó dùng flags gì.
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(args, tb_writer, iteration, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs, GsDict=None, eval_history=None):
    if tb_writer:
        # tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
                
    if 'DTU' in scene.source_path:
        depth_rgb = True
    else:
        depth_rgb = True
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})
        
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                MetricDict = {}
                for i in range(args.gaussiansN):
                    if i != 0:
                        MetricDict[f"l1_test_gs{i}"], MetricDict[f"psnr_test_gs{i}"], MetricDict[f"ssim_test_gs{i}"], MetricDict[f"lpips_test_gs{i}"] = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)   
                    black = torch.zeros_like(gt_image).to(gt_image.device) 
                    RenderResults = {}
                    
                    # [CRSGaussian Track B] training_report eval — force disable_dropout=True
                    render_results = renderFunc(viewpoint, scene.gaussians, *renderArgs, disable_dropout=True)
                    render_image = torch.clamp(render_results["render"], 0.0, 1.0)
                    render_depth = render_results["depth"]
                    render_depth_image = depth2image(render_depth, inverse=True, rgb=depth_rgb)
                    render_opacity_image = render_results["alpha"].repeat(3, 1, 1)

                    if args.gaussiansN > 1:
                        # [CRSGaussian Track B] eval gs1 — force disable_dropout=True
                        render_results_gs1 = renderFunc(viewpoint, GsDict['gs1'], *renderArgs, disable_dropout=True)
                        render_image_gs1 = torch.clamp(render_results_gs1["render"], 0.0, 1.0)
                        render_depth_gs1 = render_results_gs1["depth"]
                        render_depth_image_gs1 = depth2image(render_depth_gs1, inverse=True, rgb=depth_rgb)
                        render_opacity_image_gs1 = render_results_gs1["alpha"].repeat(3, 1, 1)

                               


                    if tb_writer and (idx < 8):
                        row0 = torch.cat([gt_image, black, black], dim=2)
                        row1 = torch.cat([render_image, render_depth_image, render_opacity_image], dim=2)
                        if args.gaussiansN > 1:
                            row2 = torch.cat([render_image_gs1, render_depth_image_gs1, render_opacity_image_gs1], dim=2)
                        else:
                            row2 = torch.cat([black, black, black], dim=2)
                        
                        image_to_show = torch.cat([row0, row1, row2], dim=1)
                        image_to_show = torch.clamp(image_to_show, 0, 1)
                        
                        save_path = f"{args.model_path}/save_images_{config['name']}/view_{viewpoint.image_name}"
                        os.makedirs(save_path, exist_ok = True)
                        torchvision.utils.save_image(image_to_show, save_path + f"/{iteration}.jpg") 

                        tb_writer.add_images(config['name'] + "_view_{}/render_image".format(viewpoint.image_name), render_image[None], global_step=iteration)
                        # tb_writer.add_images(config['name'] + "_view_{}/render_depth".format(viewpoint.image_name), render_depth_image[None], global_step=iteration)
                        # tb_writer.add_images(config['name'] + "_view_{}/alpha".format(viewpoint.image_name), alpha[None], global_step=iteration)

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)


                    l1_test += l1_loss(render_image, gt_image).mean().double()

                    _mask = None
                    _psnr = psnr(render_image, gt_image, _mask).mean().double()
                    _ssim = ssim(render_image, gt_image, _mask).mean().double()
                    _lpips = lpips(render_image, gt_image, _mask, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips

                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                # ── [CRSGaussian] Thu thập kết quả cho summary table ──
                if eval_history is not None:
                    N_gs = scene.gaussians.get_xyz.shape[0]
                    eval_history.append({
                        'iter': iteration,
                        'split': config['name'],
                        'psnr': float(psnr_test),
                        'ssim': float(ssim_test),
                        'lpips': float(lpips_test),
                        'l1': float(l1_test),
                        'n_gaussians': N_gs,
                    })
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training")  # Đóng vai trò dịch từ ngôn ngữ terminal sang biến python? chấp nhận việc chạy script? => tạo khung dịch thuật
    # Gán 3 nhóm flags lớn => gắn thêm các bộ từ điển chuyên ngành vào khung phía trên
    # 3 dòng dưới đăng ký các nhóm CLI arguments vào parser
    lp = ModelParams(parser)         # Đăng ký nhóm tham số mô hình (scene + Gaussian config), định nghĩa trong arguments/__init__.py
    op = OptimizationParams(parser)  # Đăng ký nhóm tham số tối ưu hóa (training schedule, LR, densification...)
    pp = PipelineParams(parser)      # Đăng ký nhóm tham số pipeline/rendering
    parser.add_argument('--ip', type=str, default="127.0.0.1") # Không bật,	Địa chỉ mạng cho cái GUI xem 3D real-time. Nhưng mà hiện không dùng nên tắt
    parser.add_argument('--port', type=int, default=6009) # Không bật,	Địa chỉ mạng cho cái GUI xem 3D real-time. Nhưng mà hiện không dùng nên tắt
    parser.add_argument('--debug_from', type=int, default=-1) #Bắt đầu bật chế độ debug từ vòng lặp số mấy. -1 = không debug.
    parser.add_argument('--detect_anomaly', action='store_true', default=False) # Bật chế độ dò lỗi tính toán (chạy chậm hơn). Mặc định tắt.

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500, 2000, 3000, 5000, 7000, 10000, 15000, 30000])  # Số vòng lặp được chấm điểm 
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000, 30000])   # vòng lặp lưu kết quả .ply
    parser.add_argument("--quiet", action="store_true") #Dùng để tăt toàn bộ  mọi print() trong code , tắt log in ra màn hình terminal 
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10_000]) # Tại vòng nào thì lưu checkpoint (để train tiếp sau này được).
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--train_bg", action="store_true")  # Có train luôn cả phần nền (background) hay không. mặc định tắt khi 3 views

    parser.add_argument('--gaussiansN', type=int, default=1) #Không bật, số Gaussian field song song, base của CoRGS, của mình chạy theo 3dgs gốc thì là 1 field


    parser.add_argument("--save_log_images", action="store_true") #Mỗi 100 iter, render một camera ngẫu nhiên và lưu ảnh debug dạng lưới 3×2

    # [CRSGaussian multi-seed] verify CUDA noise — default=42 giữ backward-compat
    parser.add_argument('--seed', type=int, default=42) # Quy định seed thôi, mình dùng 3 seed là 42 137 9999, do CUDA noise , chính vì atomicAdd non-determinism của rasterizer gây ±1.3 dB variance single-scene
    """
    42: seed chuẩn của cộng đồng ML ("the answer to everything"), dùng từ đầu dự án → backward-compatible với mọi single-seed run cũ
    137: số nguyên tố, không có pattern đặc biệt, chỉ cần "khác 42 và không liên quan"
    9999: số lớn tròn, dễ nhớ, rõ ràng khác hoàn toàn về magnitude
    Yêu cầu thực tế chỉ là 3 seed độc lập nhau để average ra noise. Bất kỳ bộ 3 seed nào khác (ví dụ 0 1 2 hay 100 200 300) cũng cho kết quả thống kê tương đương. Chọn bộ này từ sớm rồi giữ nguyên xuyên suốt Phase 13→25 để các Δ có thể so sánh paired trực tiếp.
    
    """
    
    # [CRSGaussian Phase 13] AbsGS flag auto-registered via OptimizationParams
    # (arguments/__init__.py:225 self.absdensify=False). DO NOT add here — gây
    # argparse conflict "conflicting option string: --absdensify".
    # parser.add_argument("--absdensify", action="store_true")
    """
    Block comment --absdensify

    --absdensify điều khiển AbsGS densification. Tưởng phải khai báo ở đây nhưng không được — vì OptimizationParams trong arguments/__init__.py:225 đã tự đăng ký --absdensify vào cùng parser rồi. Nếu khai báo thêm ở đây sẽ bị argparse báo lỗi conflicting option string. 
    Comment này là cảnh báo để người sau không vô tình thêm lại.
    """

    args = parser.parse_args(sys.argv[1:]) # Parse toàn bộ CLI flags thành object args. 
    args.save_iterations.append(args.iterations) #Tự động thêm iter cuối cùng (thường = 10000) vào danh sách save.Ở đây save .ply 
    """
    File .ply này chứa toàn bộ tham số của các Gaussian tại iter đó: vị trí xyz, SH coefficients (màu sắc), opacity, scale, rotation. Đây là "mô hình đã train xong" dùng để render ảnh sau này.
    """


    print(args.test_iterations)
    print("Optimizing " + args.model_path)

    seed_everything(args.seed) #Set cùng một seed cho tất cả các thư viện đang dùng, nhằm đảm bảo rằng nếu chạy lại chương trình với cùng seed (ví dụ 42), thì các thao tác có yếu tố ngẫu nhiên sẽ diễn ra theo cùng một trình tự. Biết rằng với cùng một seed, Bộ sinh số ngẫu nhiên sẽ sinh ra cùng một chuỗi số mỗi lần chạy.
    

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=args.seed)


    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly) #Không bật, dùng để trace nan
    """
    Gọi hàm training, tách arg thành 3 nhóm tham số (Model, Optimization, Pipeline) rồi truyền vào training()
    args ban đầu = một namespace chứa tất cả 50+ flags trộn lẫn vào nhau — do argparse parse từ CLI ra
    lp.extract(args) → tách ra chỉ lấy phần thuộc ModelParams
    op.extract(args) → tách ra chỉ lấy phần thuộc OptimizationParams
    pp.extract(args) → tách ra chỉ lấy phần thuộc PipelineParams
    Nhóm thực sự đã được định nghĩa từ trước trong __init__ của mỗi class (ở arguments/__init__.py). Đến đây chỉ là tách args theo đúng nhóm đó rồi truyền vào training().
    training(
        dataset,   # chỉ có ModelParams flags: source_path, sh_degree, use_depth_prior, ...
        opt,       # chỉ có OptimizationParams flags: lr, iterations, densify, CRS flags, ...
        pipe,      # chỉ có PipelineParams flags: dropansh config, convert_SHs_python, ...
        args       # toàn bộ namespace gốc: seed, gaussiansN, save_log_images, ...
    )
 
    """
    training(lp.extract(args), op.extract(args), pp.extract(args), args) #Hàm trainning

    # All done
    print("\nTraining complete.")