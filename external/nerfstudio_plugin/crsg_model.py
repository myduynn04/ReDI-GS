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


class CrsgSplatfactoModel(SplatfactoModel):
    """[CRSGaussian Plug-in A2.1] SplatfactoModel + opacity decay.

    Mọi method khác inherit nguyên SplatfactoModel.
    """

    config: CrsgSplatfactoModelConfig

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
