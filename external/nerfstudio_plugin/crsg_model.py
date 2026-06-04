# ============================================================
# [CRSGaussian Plug-in A2.1] CrsgSplatfactoModel — splatfacto + opacity decay
# File: crsgaussian_plugin/crsg_model.py  (KEEP LOCAL — upload server)
#
# Subclass SplatfactoModel — override step_cb() để inject opacity decay logic.
#
# Mechanism (verbatim từ CRSGaussian Phase 22, Binocular3DGS pattern):
#   - Mỗi iter sau opacity_decay_start_iter (default 500 = densify_from_iter)
#   - opacity_new = opacity_old × factor (default 0.999)
#   - → opacity giảm liên tục → Gaussian "zombie" bị loại tự nhiên qua opacity prune
#
# Implementation note:
#   - splatfacto lưu opacity ở RAW LOGIT space (sigmoid được apply khi render)
#   - Để multiply factor: sigmoid(new_raw) = factor × sigmoid(old_raw)
#       → new_raw = logit(factor × sigmoid(old_raw))
#
# Tier 2 verify:
#   2.1 OFF flag (use_opacity_decay=False) → khớp splatfacto-roma ±0.10 dB
#   2.2 ON flag → opacity median giảm sau decay_start_iter (log per N step)
# ============================================================
"""[CRSGaussian Plug-in A2.1] Subclass SplatfactoModel — thêm opacity decay."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Type, Union

import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig

from .dropansh import (
    anchor_dropout_keep_mask,
    sh_degree_dropout_inplace,
    restore_sh_dropout,
)
from .depth_loss import load_aligned_depth_dict, pearson_depth_loss
from .crs_module_a24 import compute_depth_consistency, update_crs_ema


@dataclass
class CrsgSplatfactoModelConfig(SplatfactoModelConfig):
    """[CRSGaussian Plug-in A2.x] Config — extends SplatfactoModelConfig.

    Thêm field cho opacity decay (A2.1) + DropAnSH (A2.2).
    Mỗi flag có master switch — OFF → byte-identical parent.
    """

    _target: Type = field(default_factory=lambda: CrsgSplatfactoModel)

    # ── [Plug-in A2.1] Opacity decay (Phase 22 contribution #2) ──
    use_opacity_decay: bool = True
    """[A2.1] Master switch — multiply opacity factor mỗi iter sau start_iter."""

    opacity_decay_factor: float = 0.999
    """[A2.1] Decay multiplier per iter (Phase 22 default cho 10k budget)."""

    opacity_decay_start_iter: int = 500
    """[A2.1] Iter bắt đầu decay (default = densify_from_iter)."""

    # ── [Plug-in A2.2] DropAnSH (Phase 22 contribution #3) ──
    use_dropansh: bool = True
    """[A2.2] Master switch — apply anchor + SH degree dropout during forward."""

    dropansh_pa_max: float = 0.02
    """[A2.2] Anchor sample rate max (Phase 22 default = 2%)."""

    dropansh_k: int = 10
    """[A2.2] k-NN per anchor (Phase 22 default)."""

    dropansh_psh: float = 0.2
    """[A2.2] Per-Gaussian SH dropout probability (Phase 22 default)."""

    dropansh_total_iter: int = 10000
    """[A2.2] Tổng iter training cho linear ramp pa (match max_num_iterations)."""

    dropansh_schedule: Tuple[int, int, int] = (2000, 4000, 6000)
    """[A2.2] 3 checkpoints SH degree schedule: lmax 0→1→2 → off."""

    # ── [Plug-in A2.3] DAV2 depth loss (Phase 22 contribution #4) ──
    use_depth_loss: bool = True
    """[A2.3] Master switch — add λ × pearson_depth_loss vào get_loss_dict."""

    depth_loss_weight: float = 0.05
    """[A2.3] λ — Phase 22 / Phase 3 default = 0.05."""

    # ── [Plug-in A2.4] CRS score module (Phase 22 contribution #5) ──
    # A2.4-minimal: D-only (skip R_i — cần access GT images, complex)
    # SH freeze deferred A3 — splatfacto step_cb không có hook BEFORE_OPTIMIZER_STEP,
    # densify replace features_rest tensor → grad hook không persist.
    use_crs: bool = True
    """[A2.4] Master switch — compute CRS score per-Gaussian periodic."""

    crs_update_interval: int = 100
    """[A2.4] Update CRS mỗi N iter (Phase 22 default = 100)."""

    crs_update_warmup: int = 1000
    """[A2.4] Iter bắt đầu update CRS (Phase 22 T_warmup = 1000)."""

    crs_ema_decay: float = 0.9
    """[A2.4] EMA decay smoothing CRS score."""

    crs_logit_scale: float = 5.0
    """[A2.4] Scale factor cho logit space — Phase 22 default 5.0."""


class CrsgSplatfactoModel(SplatfactoModel):
    """[CRSGaussian Plug-in A2.1] SplatfactoModel + opacity decay.

    Mọi method khác inherit nguyên SplatfactoModel.
    """

    config: CrsgSplatfactoModelConfig

    def populate_modules(self):
        super().populate_modules()
        # ── [Plug-in A2.3] Pre-load aligned depth dict ──
        self._aligned_depth_dict = {}
        self._idx_to_stem: Dict[int, str] = {}
        if not self.config.use_depth_loss:
            return

        # Lấy data root từ ENV var (set bởi run script)
        # Pattern: CRSG_A23_DATA_ROOT=/home/.../<scene>/3_views/
        import os as _os
        data_root_str = _os.environ.get("CRSG_A23_DATA_ROOT", None)
        if not data_root_str:
            print(
                "[Plug-in A2.3] WARNING: CRSG_A23_DATA_ROOT chưa set → skip depth loss.\n"
                "  Run script phải export CRSG_A23_DATA_ROOT=<scene>/3_views/"
            )
            return

        data_root = Path(data_root_str)
        self._aligned_depth_dict, self._depth_range = load_aligned_depth_dict(data_root)
        if not self._aligned_depth_dict:
            print(
                f"[Plug-in A2.3] WARNING: KHÔNG tìm thấy aligned_depth_a23/*.npy trong {data_root}\n"
                f"  → Run preprocess: SCENE=<scene> python crsgaussian_plugin/preprocess_depth_a23.py"
            )
            return

        # ── [Plug-in A2.4] Cache train cameras từ metadata kwargs ──
        # RomaDataParser inject 'train_cameras' vào metadata
        metadata = self.kwargs.get("metadata", {}) if hasattr(self, "kwargs") else {}
        self._train_cameras_for_crs = metadata.get("train_cameras", None)
        if self.config.use_crs and self._train_cameras_for_crs is None:
            print(
                "[Plug-in A2.4] WARNING: train_cameras không có trong metadata → CRS skip.\n"
                "  Cần RomaDataParser inject vào metadata."
            )

        # Map image_idx → stem (cùng convention ColmapDataParser eval-interval=8)
        images_dir = data_root / "images"
        if images_dir.is_dir():
            all_stems = sorted(
                [p.stem for p in images_dir.iterdir()
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
            )
            eval_interval = 8  # match RomaDataParserConfig default
            train_stems = [s for i, s in enumerate(all_stems) if i % eval_interval != 0]
            self._idx_to_stem = {i: s for i, s in enumerate(train_stems)}
            print(
                f"[Plug-in A2.3] Loaded {len(self._aligned_depth_dict)} depth maps, "
                f"mapped {len(self._idx_to_stem)} train_idx → stem"
            )

    def step_cb(self, optimizers: Optimizers, step):
        # ── [CRSGaussian Plug-in A2.1] Call parent FIRST để giữ behavior gốc ──
        super().step_cb(optimizers, step)

        # ── [CRSGaussian Plug-in A2.1] Inject opacity decay ──
        if not self.config.use_opacity_decay:
            return  # Tier 2.1: OFF → byte-identical parent

        if step <= self.config.opacity_decay_start_iter:
            return  # Chưa đến lúc decay

        # Multiply opacity (post-sigmoid) by factor
        # Splatfacto lưu opacity ở RAW LOGIT space, sigmoid apply khi render
        # → adjust raw sao cho sigmoid(new_raw) = factor × sigmoid(old_raw)
        with torch.no_grad():
            raw = self.gauss_params["opacities"].data
            opacity = torch.sigmoid(raw)
            opacity_decayed = opacity * self.config.opacity_decay_factor
            # Clamp tránh logit(0) hoặc logit(1) → inf
            opacity_decayed = opacity_decayed.clamp(1e-6, 1.0 - 1e-6)
            self.gauss_params["opacities"].data = torch.logit(opacity_decayed)

        # Log mỗi 500 step (verify Tier 2.2 — opacity giảm thực sự)
        if step % 500 == 0:
            with torch.no_grad():
                opa_now = torch.sigmoid(self.gauss_params["opacities"].data)
                print(
                    f"[CRSGaussian Plug-in A2.1] iter={step} "
                    f"opacity median={opa_now.median().item():.4f} "
                    f"mean={opa_now.mean().item():.4f} "
                    f"N_gauss={opa_now.shape[0]}"
                )

        # ── [CRSGaussian Plug-in A2.4] CRS score update ──
        if not self.config.use_crs:
            return
        if step <= self.config.crs_update_warmup:
            return
        if step % self.config.crs_update_interval != 0:
            return
        if not self._aligned_depth_dict or not self._idx_to_stem:
            return

        means = self.gauss_params["means"]
        N_current = means.shape[0]

        # Re-init _crs_score nếu shape mismatch (sau densify/prune)
        if not hasattr(self, "_crs_score") or self._crs_score.shape[0] != N_current:
            self._crs_score = torch.zeros(N_current, 1, device=means.device)

        # Get train cameras từ datamanager
        train_cams = getattr(self, "_train_cameras_for_crs", None)
        if train_cams is None:
            return  # cameras chưa cached → skip

        with torch.no_grad():
            D = compute_depth_consistency(
                means=means,
                cameras=train_cams,
                aligned_depth_dict=self._aligned_depth_dict,
                idx_to_stem=self._idx_to_stem,
                depth_range=self._depth_range,
            )
            # A2.4-minimal: D-only (skip R for simplicity)
            self._crs_score = update_crs_ema(
                self._crs_score, D, None,
                w1=1.0, w2=0.0,
                scale=self.config.crs_logit_scale,
                ema=self.config.crs_ema_decay,
            )

            # Log mỗi 500 step
            if step % 500 == 0:
                crs = torch.sigmoid(self._crs_score).squeeze(-1)
                print(
                    f"[CRSGaussian Plug-in A2.4] iter={step} "
                    f"CRS median={crs.median().item():.3f} "
                    f"p10={crs.quantile(0.1).item():.3f} "
                    f"p90={crs.quantile(0.9).item():.3f} "
                    f"D_median={D.median().item():.3f} "
                    f"N={N_current}"
                )

    # ── [CRSGaussian Plug-in A2.2] Override get_outputs để inject DropAnSH ──
    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """[A2.2] Apply anchor + SH degree dropout trước rasterize, restore sau."""
        # OFF flag hoặc không phải train mode → fallback parent
        if not self.config.use_dropansh or not self.training:
            return super().get_outputs(camera)

        # ── 1. Anchor dropout: mask Gaussians bằng cách set opacity rất thấp ──
        opa_param = self.gauss_params["opacities"]
        means = self.gauss_params["means"]

        keep_mask = anchor_dropout_keep_mask(
            means.detach(),
            iteration=self.step,
            total_iter=self.config.dropansh_total_iter,
            k=self.config.dropansh_k,
            pa_max=self.config.dropansh_pa_max,
        )
        drop_mask = ~keep_mask

        opa_snapshot = None
        if drop_mask.any():
            with torch.no_grad():
                opa_snapshot = opa_param.data[drop_mask].clone()
                # Set logit cực thấp → sigmoid ≈ 0 → Gaussian không render
                opa_param.data[drop_mask] = -1e5

        # ── 2. SH degree dropout: zero features_rest above lmax in-place ──
        sh_snapshot = sh_degree_dropout_inplace(
            self.gauss_params["features_rest"],
            iteration=self.step,
            p_sh=self.config.dropansh_psh,
            schedule=self.config.dropansh_schedule,
        )

        # Log mỗi 500 step (verify Tier 2.2 — module ACTIVE)
        if self.step % 500 == 0:
            n_drop = int(drop_mask.sum().item())
            sh_active = sh_snapshot is not None
            print(
                f"[CRSGaussian Plug-in A2.2] iter={self.step} "
                f"anchor_drop={n_drop}/{means.shape[0]} "
                f"({100 * n_drop / max(means.shape[0], 1):.1f}%) "
                f"sh_dropout={'YES' if sh_active else 'NO'}"
            )

        try:
            # ── 3. Forward parent (rasterize với Gaussian đã mask) ──
            outputs = super().get_outputs(camera)
        finally:
            # ── 4. Restore opacity + features_rest dù forward fail hay không ──
            if opa_snapshot is not None:
                opa_param.data[drop_mask] = opa_snapshot
            restore_sh_dropout(self.gauss_params["features_rest"], sh_snapshot)

        return outputs

    # ── [CRSGaussian Plug-in A2.3] Override get_loss_dict thêm depth loss ──
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # OFF flag hoặc không phải train mode → fallback
        if not self.config.use_depth_loss or not self.training:
            return loss_dict
        if not self._aligned_depth_dict or not self._idx_to_stem:
            return loss_dict
        if "depth" not in outputs:
            # Splatfacto default output_depth_during_training=False
            # Cần set True trong config để có outputs["depth"]
            return loss_dict

        idx = int(batch.get("image_idx", 0))
        stem = self._idx_to_stem.get(idx)
        if stem is None or stem not in self._aligned_depth_dict:
            return loss_dict

        depth_prior = self._aligned_depth_dict[stem]
        rendered_depth = outputs["depth"]
        L_depth = self.config.depth_loss_weight * pearson_depth_loss(
            rendered_depth, depth_prior
        )
        loss_dict["depth_loss"] = L_depth

        if self.step % 500 == 0 and self.step > 0:
            print(
                f"[CRSGaussian Plug-in A2.3] iter={self.step} "
                f"depth_loss={float(L_depth):.4f} stem={stem}"
            )

        return loss_dict
