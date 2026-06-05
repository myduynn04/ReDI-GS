# ============================================================
# [CRSGaussian Path A B3 REWRITE] CrsGaussianModel — full Phase 22 plug-in
# File: crsgaussian_plugin/crsgaussian_model.py  (KEEP LOCAL — upload server)
#
# Mục đích: Subclass nerfstudio Model — wrap CRSGaussian (CoR-GS fork) GaussianModel
#           + render + 7 Phase 22 callbacks (full A3-TRIM 8-module recipe).
#
# Source-of-truth: doc 09_path_a_phase22_mapping.md
#                  CRSGaussian/scripts/p22_pilot_run.sh + p20_ablation_dense_run.sh
#                  CRSGaussian/train.py per-iter loop
#
# Phase 22 invariants (KHÔNG được vi phạm):
#   1. CRS update F TRƯỚC densify G
#   2. SH freeze H SAU densify, TRƯỚC optimizer.step I
#   3. S_stability EMA J SAU optimizer.step
#   4. Opacity decay K mỗi iter sau densify_from
#   5. Opacity reset M chỉ ở iter cụ thể (501, 3501, 6501, 9501)
# ============================================================
"""[Path A B3] Full Phase 22 plug-in for Nerfstudio."""

import math
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.models.base_model import Model

from .corgs_imports import (
    CorGsGaussianModel,
    corgs_render,
    corgs_ssim,
)
from .camera_adapter import ns_camera_to_corgs_minicam, ns_camera_to_corgs_camera
from .crsgaussian_config import CrsGaussianModelConfig
from .depth_loss import load_aligned_depth_dict, pearson_depth_loss
from .dropansh import (
    anchor_dropout_keep_mask,
    sh_degree_dropout_inplace,
    restore_sh_dropout,
    apply_anchor_dropout_to_opacity,
    restore_opacity,
)
from .callbacks.corgs_optimizer import make_corgs_optimizer_step_fn
from .callbacks.densify import make_densify_step_fn
from .callbacks.opacity_decay import make_opacity_decay_step_fn
from .callbacks.opacity_reset import make_opacity_reset_step_fn
from .callbacks.crs_update import make_crs_update_step_fn
from .callbacks.sh_freeze import make_sh_freeze_step_fn
from .callbacks.sh_stability import make_sh_stability_step_fn


class _ArgsNamespace:
    """Wrapper fallback default cho mọi attr missing.

    CoR-GS GaussianModel + Scene access ~150 attr. Wrapper này safe fallback.
    """

    def __init__(self, **kwargs):
        self._data = dict(kwargs)

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        # Sensible defaults
        if name == "sh_degree":
            return 3
        if name in ("white_background", "train_bg", "use_depth_prior", "use_color"):
            return False
        return False

    def __setattr__(self, name, value):
        if name == "_data":
            super().__setattr__(name, value)
        else:
            self._data[name] = value


class CrsGaussianModel(Model):
    """[Path A B3] Full Phase 22 plug-in — A3-TRIM 8-module recipe."""

    config: CrsGaussianModelConfig

    def __init__(
        self,
        config: CrsGaussianModelConfig,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # Set seed_points + metadata TRƯỚC super().__init__ (super calls populate_modules)
        self.seed_points = seed_points
        self._ds_metadata = metadata or {}
        super().__init__(config=config, metadata=metadata, **kwargs)

    # ════════════════════════════════════════════════════════
    # populate_modules — init CoR-GS GaussianModel + load depth + build CoR-GS cams
    # ════════════════════════════════════════════════════════
    def populate_modules(self):
        super().populate_modules()

        # ── Verify seed_points injected từ DataParser ──
        if self.seed_points is None:
            raise RuntimeError(
                "[Path A B3] seed_points missing — DataParser must inject points3D_xyz/rgb"
            )

        xyz_seed = self.seed_points[0]   # (N, 3) float
        rgb_seed = self.seed_points[1]   # (N, 3) uint8 hoặc float

        n_points = xyz_seed.shape[0]
        print(f"[Path A B3] Init CoR-GS GaussianModel from {n_points} seed points")

        # ── Build args namespace cho GaussianModel ──
        gaussian_args = _ArgsNamespace(
            sh_degree=self.config.sh_degree,
            resolution=self.config.resolution,
            white_background=self.config.white_background,
            n_views=self.config.n_views,
            use_color=False,
            train_bg=False,
            use_depth_prior=self.config.use_depth_prior,
        )

        self.gaussians = CorGsGaussianModel(gaussian_args)

        # ── create_from_pcd ──
        from utils.graphics_utils import BasicPointCloud

        # Spatial lr scale = bbox diameter (CoR-GS heuristic — Scene tự tính)
        bbox_diag = (xyz_seed.max(dim=0).values - xyz_seed.min(dim=0).values).norm().item()
        spatial_lr_scale = bbox_diag * 0.5

        rgb_float = rgb_seed.float()
        if rgb_float.max() > 1.0:
            rgb_float = rgb_float / 255.0
        normals = torch.zeros_like(xyz_seed)
        pcd = BasicPointCloud(
            points=xyz_seed.cpu().numpy(),
            colors=rgb_float.cpu().numpy(),
            normals=normals.cpu().numpy(),
        )
        self.gaussians.create_from_pcd(pcd, spatial_lr_scale)

        # ── training_setup ──
        opt_args = _ArgsNamespace(
            iterations=self.config.iterations,
            position_lr_init=self.config.position_lr_init,
            position_lr_final=self.config.position_lr_final,
            position_lr_delay_mult=self.config.position_lr_delay_mult,
            position_lr_max_steps=self.config.position_lr_max_steps,
            feature_lr=self.config.feature_lr,
            opacity_lr=self.config.opacity_lr,
            scaling_lr=self.config.scaling_lr,
            rotation_lr=self.config.rotation_lr,
            percent_dense=self.config.percent_dense,
            lambda_dssim=self.config.lambda_dssim,
            densification_interval=self.config.densification_interval,
            opacity_reset_interval=self.config.opacity_reset_interval,
            densify_from_iter=self.config.densify_from_iter,
            densify_until_iter=self.config.densify_until_iter,
            densify_grad_threshold=self.config.densify_grad_threshold,
            train_bg=False,
        )
        self.gaussians.training_setup(opt_args)

        # ── pipe namespace cho render ──
        self._pipe = _ArgsNamespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=False,
            use_confidence=False,
        )

        # ── background tensor ──
        bg_color = [1, 1, 1] if self.config.white_background else [0, 0, 0]
        self._background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # ── cameras_extent — match CoR-GS Scene.cameras_extent ──
        self._cameras_extent = spatial_lr_scale

        # ── Render state placeholders cho densify callback ──
        self._last_viewspace_pts = None
        self._last_visibility = None
        self._last_radii = None

        # ── Iter counter ──
        self._iter_counter = 0

        print(f"[Path A B3] GaussianModel ready: N={self.gaussians._xyz.shape[0]}, extent={self._cameras_extent:.3f}")

        # ════════════════════════════════════════════════════════
        # Load aligned DAV2 depth + build idx→depth lookup
        # ════════════════════════════════════════════════════════
        self.depth_by_idx: List[Optional[torch.Tensor]] = []
        self.depth_range: float = 1.0
        data_root = self._ds_metadata.get("data_root", None)
        aligned_depth_dir = self._ds_metadata.get("aligned_depth_dir", None)
        image_filenames = self._ds_metadata.get("image_filenames", [])
        # [B3 split-fix 2026-06-04] Use aligned_depth_dir metadata (full path)
        # KHÔNG dùng data_root + hardcode "aligned_depth_a23" — split-fix moved
        # depth folder vào fern/3_views/aligned_depth_a23/
        if self.config.use_depth_prior and aligned_depth_dir and image_filenames:
            depth_dict_by_stem, depth_range = load_aligned_depth_dict(Path(aligned_depth_dir))
            self.depth_range = depth_range
            for fn in image_filenames:
                stem = Path(str(fn)).stem
                self.depth_by_idx.append(depth_dict_by_stem.get(stem, None))
            n_d = sum(1 for d in self.depth_by_idx if d is not None)
            print(f"[Path A B3] Depth dict: {n_d}/{len(self.depth_by_idx)} train cams")

        # ════════════════════════════════════════════════════════
        # [Path A B2] Build CoR-GS FULL Camera list cho TRAIN cam (callbacks)
        # PORTED FROM: CRSGaussian/train.py:114 `allCameras = scene.getTrainCameras().copy()`
        # Phase 22 dùng full Camera cho:
        #   - LFCF compute_3D_interval (cần cam.R, cam.T numpy)
        #   - CRS update compute_reprojection_consistency (cần cam.original_image)
        #   - D_cycle compute (cần cam.world_view_transform, cam.uid)
        #
        # MiniCam KHÔNG đủ → phải build full Camera với image loaded.
        # Image filename + downscale_factor=8 (DataParser config) → ns Cameras
        # image_width/height match PIL load exact.
        #
        # Storage rule (VERIFY 2 result): Camera là nn.Module nhưng tensor attrs
        # plain assign → KHÔNG enter state_dict. Storage = Python plain list
        # (NOT nn.ModuleList) → tránh nn.Module trace Camera làm submodule.
        # ════════════════════════════════════════════════════════
        self._train_corgs_cams: List = []
        self._aligned_depth_dict_by_uid: Dict[int, torch.Tensor] = {}
        train_cameras = self._ds_metadata.get("train_cameras", None)
        if train_cameras is not None and image_filenames:
            n_cams = train_cameras.camera_to_worlds.shape[0]
            for idx in range(n_cams):
                cam = ns_camera_to_corgs_camera(
                    train_cameras[idx],
                    image_filenames[idx],
                    idx=idx,
                    data_device="cuda",
                )
                self._train_corgs_cams.append(cam)
                # Map aligned depth by uid (parallel với image_filenames index)
                if idx < len(self.depth_by_idx) and self.depth_by_idx[idx] is not None:
                    self._aligned_depth_dict_by_uid[idx] = self.depth_by_idx[idx]
            print(f"[Path A B2] Built {len(self._train_corgs_cams)} full CoR-GS Cameras "
                  f"(depth keys: {len(self._aligned_depth_dict_by_uid)})")
        elif train_cameras is not None:
            # Fallback: KHÔNG có image_filenames → build MiniCam (skip R compute path)
            n_cams = train_cameras.camera_to_worlds.shape[0]
            for idx in range(n_cams):
                minicam = ns_camera_to_corgs_minicam(train_cameras[idx])
                minicam.uid = idx
                self._train_corgs_cams.append(minicam)
            print(f"[Path A B2] WARN: fallback {len(self._train_corgs_cams)} MiniCams "
                  f"— R compute sẽ fail nếu chạy")

        # Reference to render function (cho crs_update callback)
        self._corgs_render_fn = corgs_render

    # ════════════════════════════════════════════════════════
    # forward — skip ns collider (full-image model)
    # ════════════════════════════════════════════════════════
    def forward(self, camera: Cameras) -> Dict[str, torch.Tensor]:
        return self.get_outputs(camera)

    def get_outputs_for_camera(self, camera: Cameras, obb_box=None) -> Dict[str, torch.Tensor]:
        return self.get_outputs(camera)

    # ════════════════════════════════════════════════════════
    # get_outputs — render + DropAnSH snapshot + capture render state
    # ════════════════════════════════════════════════════════
    def get_outputs(self, camera: Cameras) -> Dict[str, torch.Tensor]:
        minicam = ns_camera_to_corgs_minicam(camera)

        # ── DropAnSH: apply BEFORE render, restore AFTER (training only) ──
        opacity_snapshot = None
        sh_snapshot = None
        if self.training and self.config.use_dropansh:
            keep_mask = anchor_dropout_keep_mask(
                self.gaussians._xyz,
                self._iter_counter,
                total_iter=self.config.dropansh_total_iter,
                k=self.config.dropansh_k,
                pa_max=self.config.dropansh_pa,
            )
            opacity_snapshot = apply_anchor_dropout_to_opacity(
                self.gaussians._opacity, keep_mask
            )
            sh_snapshot = sh_degree_dropout_inplace(
                self.gaussians._features_rest,
                self._iter_counter,
                p_sh=self.config.dropansh_psh,
                schedule=(
                    self.config.dropansh_schedule_0,
                    self.config.dropansh_schedule_1,
                    self.config.dropansh_schedule_2,
                ),
            )

        # ── Render với background random nếu config.random_background (Phase 22) ──
        if self.training and self.config.random_background:
            bg = torch.rand((3,), device="cuda")
        else:
            bg = self._background

        render_pkg = corgs_render(
            viewpoint_camera=minicam,
            pc=self.gaussians,
            pipe=self._pipe,
            bg_color=bg,
        )

        # ── Restore DropAnSH state ──
        if opacity_snapshot is not None:
            restore_opacity(self.gaussians._opacity, opacity_snapshot)
        if sh_snapshot is not None:
            restore_sh_dropout(self.gaussians._features_rest, sh_snapshot)

        # ── Extract render outputs ──
        rendered_image = render_pkg["render"]
        depth = render_pkg.get("depth")
        alpha = render_pkg.get("alpha")
        viewspace_pts = render_pkg.get("viewspace_points")
        visibility = render_pkg.get("visibility_filter")
        radii = render_pkg.get("radii")

        # ── Capture render state cho densify callback ──
        if self.training and self.config.use_densify:
            self._last_viewspace_pts = viewspace_pts
            self._last_visibility = visibility
            self._last_radii = radii

        # ── Permute cho ns convention ──
        rgb = rendered_image.permute(1, 2, 0).clamp(0.0, 1.0)
        outputs = {"rgb": rgb}
        if depth is not None:
            outputs["depth"] = depth.permute(1, 2, 0)
        if alpha is not None:
            outputs["accumulation"] = alpha.permute(1, 2, 0)
        return outputs

    # ════════════════════════════════════════════════════════
    # get_loss_dict — L1 + SSIM + Pearson depth (Phase 22 D step)
    # ════════════════════════════════════════════════════════
    def get_loss_dict(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
        metrics_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        gt = batch["image"].to(outputs["rgb"].device)
        if gt.dtype == torch.uint8:
            gt = gt.float() / 255.0

        # L1
        loss_l1 = torch.nn.functional.l1_loss(outputs["rgb"], gt)

        # SSIM (CoR-GS ssim expects (1, C, H, W))
        rendered_chw = outputs["rgb"].permute(2, 0, 1).unsqueeze(0)
        gt_chw = gt.permute(2, 0, 1).unsqueeze(0)
        ssim_val = corgs_ssim(rendered_chw, gt_chw)
        loss_dssim = 1.0 - ssim_val

        # Phase 22 blend
        lambda_dssim = self.config.lambda_dssim
        loss_main = (1.0 - lambda_dssim) * loss_l1 + lambda_dssim * loss_dssim

        loss_dict = {"main_loss": loss_main}

        # Depth loss
        if (
            self.config.use_depth_prior
            and "depth" in outputs
            and len(self.depth_by_idx) > 0
        ):
            idx = batch.get("image_idx", None)
            if idx is not None:
                if isinstance(idx, torch.Tensor):
                    idx = int(idx.item())
                if 0 <= idx < len(self.depth_by_idx):
                    depth_gt = self.depth_by_idx[idx]
                    if depth_gt is not None:
                        rendered_depth = outputs["depth"].squeeze(-1)
                        depth_l = pearson_depth_loss(rendered_depth, depth_gt)
                        loss_dict["depth_loss"] = self.config.depth_loss_weight * depth_l

        return loss_dict

    def get_metrics_dict(self, outputs, batch):
        gt = batch["image"].to(outputs["rgb"].device)
        if gt.dtype == torch.uint8:
            gt = gt.float() / 255.0
        mse = torch.nn.functional.mse_loss(outputs["rgb"], gt)
        psnr = -10.0 * torch.log10(mse.clamp(min=1e-10))
        return {"psnr": psnr, "mse": mse}

    def get_image_metrics_and_images(self, outputs, batch):
        gt = batch["image"].to(outputs["rgb"].device)
        if gt.dtype == torch.uint8:
            gt = gt.float() / 255.0
        mse = torch.nn.functional.mse_loss(outputs["rgb"], gt)
        psnr = (-10.0 * torch.log10(mse.clamp(min=1e-10))).item()
        return {"psnr": psnr}, {"img": outputs["rgb"], "gt": gt}

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Empty — CoR-GS optimizer riêng, bypass ns Optimizer factory."""
        return {}

    # ════════════════════════════════════════════════════════
    # get_training_callbacks — Phase 22 ORDER (5 invariants)
    # ════════════════════════════════════════════════════════
    def get_training_callbacks(
        self,
        training_callback_attributes: TrainingCallbackAttributes,
    ) -> List[TrainingCallback]:
        """Phase 22 callback order per train.py:

        BEFORE_TRAIN_ITERATION:
          - bump_iter_counter (sync với step)
          - oneupSHdegree (mỗi 500 iter) — train.py:120

        AFTER_TRAIN_ITERATION (loss.backward đã chạy):
          F. crs_update (CRS update sau warmup, mỗi 100 iter)         — train.py:514-549
          G. densify (capture render state + densify_and_prune)        — train.py:626-732
          H. sh_freeze (zero _features_rest.grad)                       — train.py:776-784
          I. corgs_optimizer (step + zero_grad + update_lr + SH up)    — train.py:786-790
          J. sh_stability (EMA post-update SH)                         — train.py:797-805
          K. opacity_decay (mỗi iter sau densify_from)                 — train.py:813-816
          M. opacity_reset (conditional 501/3501/6501/9501)            — train.py:846-850
        """
        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self._bump_iter_counter,
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self._sh_increment_step,
            ),
            # F → G → H → I → J → K → M
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=make_crs_update_step_fn(self),
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=make_densify_step_fn(self),
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=make_sh_freeze_step_fn(self),
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=make_corgs_optimizer_step_fn(self),
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=make_sh_stability_step_fn(self),
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=make_opacity_decay_step_fn(self),
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=make_opacity_reset_step_fn(self),
            ),
        ]

    def _bump_iter_counter(self, step: int):
        self._iter_counter = step

    def _sh_increment_step(self, step: int):
        """oneupSHdegree mỗi sh_increment_interval (CoR-GS train.py:120-122)."""
        if step > 0 and step % self.config.sh_increment_interval == 0:
            self.gaussians.oneupSHdegree()

    # ════════════════════════════════════════════════════════
    # [B3 FIX v2] PyTorch native extra_state — save Gaussian params
    #
    # Verified Nerfstudio path:
    #   - Trainer.save_checkpoint (trainer.py:467) → pipeline.state_dict()
    #   - Pipeline (nn.Module) → recursive collect all submodules incl _model
    #   - nn.Module._save_to_state_dict calls get_extra_state() if overridden
    #   - On load: Trainer._load_checkpoint → pipeline.load_pipeline →
    #              pipeline.load_state_dict (nn.Module default) →
    #              recursive set_extra_state() dispatch xuống _model
    #
    # eval_utils.eval_setup gọi same path (verified eval_utils.py:63).
    #
    # Bug trước (state_dict override): Pipeline wraps key as "_model._crsg_..."
    # nn.Module.load_state_dict flat lookup tìm Tensor → tuple bị silently skip.
    # extra_state mechanism native handle non-Tensor state correctly.
    # ════════════════════════════════════════════════════════
    def get_extra_state(self):
        """[B3 FIX v2] PyTorch hook — save CoR-GS Gaussian state to checkpoint."""
        if hasattr(self, "gaussians") and self.gaussians is not None:
            return self.gaussians.capture()
        return None

    def load_state_dict(self, state_dict, **kwargs):
        """[B3 FIX v3] Override để manual dispatch _extra_state.

        PORTED FROM: nerfstudio/models/splatfacto.py:343-356 pattern (Splatfacto
        cũng override để handle dynamic gauss_params resize before super).

        PyTorch native _load_from_state_dict auto-dispatch _extra_state via
        getattr-check, NHƯNG qua Pipeline.load_state_dict strict=True+fallback
        path KHÔNG fire reliably (verified empirically: ckpt 33MB có
        _model._extra_state với tuple 13 fields, nhưng set_extra_state KHÔNG
        được gọi → print "Restored Gaussians" không xuất hiện).

        Fix: extract _extra_state TRƯỚC super (super skip với strict=False),
        rồi call set_extra_state explicit.
        """
        extra_state = None
        if isinstance(state_dict, dict) and "_extra_state" in state_dict:
            extra_state = state_dict.pop("_extra_state")
        # Super load remaining (empty cho em vì 0 params/buffers)
        result = super().load_state_dict(state_dict, strict=False)
        # Manual dispatch extra_state
        if extra_state is not None:
            self.set_extra_state(extra_state)
        return result

    def set_extra_state(self, state):
        """[B3 FIX v2] PyTorch hook — restore CoR-GS Gaussian state from checkpoint."""
        if state is None or not hasattr(self, "gaussians"):
            return
        # [B3 FIX v5+v6] Move loaded tensors to CUDA + wrap nn.Parameter for
        # per-Gaussian params (indices 1-6 in capture() tuple — gaussian_model.py:169-184):
        #   [0] active_sh_degree int
        #   [1-6] _xyz, _features_dc, _features_rest, _scaling, _rotation, _opacity → nn.Parameter
        #   [7-9] max_radii2D, xyz_gradient_accum, denom → plain Tensor cuda
        #   [10] opt_dict
        #   [11] spatial_lr_scale float
        #   [12] _crs_score → plain Tensor cuda
        #
        # Issues:
        #   v5: .cuda() creates non-leaf → Adam (training_setup line 370) raises
        #       "can't optimize a non-leaf Tensor" → restore() crash
        #   Fix: .detach().cuda() → leaf, then wrap nn.Parameter for idx 1-6
        #        (restore() assigns these to self._xyz etc. — Adam expects Parameter
        #        with requires_grad=True)
        PARAM_INDICES = {1, 2, 3, 4, 5, 6}
        def _to_cuda(t, idx):
            if not isinstance(t, torch.Tensor):
                return t
            t = t.detach().cuda()
            if idx in PARAM_INDICES:
                return torch.nn.Parameter(t.requires_grad_(True))
            return t
        state = tuple(_to_cuda(s, i) for i, s in enumerate(state))
        opt_args = _ArgsNamespace(
            iterations=self.config.iterations,
            position_lr_init=self.config.position_lr_init,
            position_lr_final=self.config.position_lr_final,
            position_lr_delay_mult=self.config.position_lr_delay_mult,
            position_lr_max_steps=self.config.position_lr_max_steps,
            feature_lr=self.config.feature_lr,
            opacity_lr=self.config.opacity_lr,
            scaling_lr=self.config.scaling_lr,
            rotation_lr=self.config.rotation_lr,
            percent_dense=self.config.percent_dense,
            lambda_dssim=self.config.lambda_dssim,
            densification_interval=self.config.densification_interval,
            opacity_reset_interval=self.config.opacity_reset_interval,
            densify_from_iter=self.config.densify_from_iter,
            densify_until_iter=self.config.densify_until_iter,
            densify_grad_threshold=self.config.densify_grad_threshold,
            train_bg=False,
        )
        self.gaussians.restore(state, opt_args)
        # [B3 FIX v4] Re-init confidence sau restore — capture() KHÔNG save attr này
        # (gaussian_model.py:169-184). Sau restore _xyz resize tới N_trained nhưng
        # confidence vẫn ở N_init → renderer line 120 torch.ones_like(pc.confidence)
        # → shape mismatch → CUDA illegal memory access.
        # Init giống create_from_pcd line 322: torch.ones_like(opacities) → (N, 1).
        n_loaded = self.gaussians._xyz.shape[0]
        self.gaussians.confidence = torch.ones((n_loaded, 1), device="cuda")
        print(f"[Path A B3] Restored Gaussians from checkpoint: N={n_loaded} (confidence re-init)")
