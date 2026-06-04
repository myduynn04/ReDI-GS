# ============================================================
# [CRSGaussian Path A B3 NEW] CRS update callback — wrap utils.crs.crs_module.update_crs
# File: crsgaussian_plugin/callbacks/crs_update.py  (KEEP LOCAL)
#
# Source verbatim from CRSGaussian train.py:519-549:
#   if (dataset.use_depth_prior
#           AND iteration > opt.T_warmup
#           AND iteration % opt.crs_update_interval == 0):
#       update_crs(gaussians, allCameras, aligned_depth_dict, depth_range,
#                  ema=opt.crs_ema_decay,
#                  use_d_cycle=..., iter=..., d_cycle_warmup=..., ...,
#                  render_func=render, pipe=pipe, bg=background,
#                  use_sh_reliability=..., sh_stability_warmup=..., ...,
#                  crs_w_s=..., disable_r_signal=...)
#
# update_crs signature verified in CRSGaussian/utils/crs/crs_module.py:420.
# ============================================================
"""[Path A B3] CRS update callback — D_cycle + S_stability per Phase 22."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..crsgaussian_model import CrsGaussianModel


def make_crs_update_step_fn(model: "CrsGaussianModel"):
    """Closure callback function — Phase 22 CRS update.

    Calls update_crs from utils.crs.crs_module which handles:
      - D_cycle compute (if use_d_cycle=True): renders N depth maps, cycle test
      - S_stability tracking (via use_sh_reliability=True): EMA SH variance
      - EMA update _crs_score in logit space (decay=crs_ema_decay=0.3)
    """

    def _step_fn(step: int):
        if not model.config.use_depth_prior:
            return
        if step <= model.config.T_warmup:
            return
        if step % model.config.crs_update_interval != 0:
            return

        # Lazy import update_crs (avoid CoR-GS import at module load if not needed)
        from utils.crs.crs_module import update_crs

        # update_crs signature (CRSGaussian/utils/crs/crs_module.py:420):
        #   gaussians, cameras, aligned_depth_dict, depth_range,
        #   w1=0.5, w2=0.5, scale=5.0, ema=0.9,
        #   use_d_cycle=False, iter=0, d_cycle_warmup=1000,
        #   d_cycle_sigma=5.0, d_cycle_update_freq=100,
        #   render_func=None, pipe=None, bg=None,
        #   use_sh_reliability=False, sh_stability_warmup=1000,
        #   sh_stability_ema_beta=0.95, crs_w_s=0.33,
        #   disable_r_signal=False
        update_crs(
            model.gaussians,
            model._train_corgs_cams,            # List[CorGsMiniCam] — built in populate_modules
            model._aligned_depth_dict_by_uid,   # Dict[uid → tensor] — built in populate_modules
            model.depth_range,
            w1=model.config.crs_w1,
            w2=model.config.crs_w2,
            scale=model.config.crs_scale,
            ema=model.config.crs_ema_decay,
            use_d_cycle=model.config.use_d_cycle,
            iter=step,
            d_cycle_warmup=model.config.d_cycle_warmup,
            d_cycle_sigma=model.config.d_cycle_sigma,
            d_cycle_update_freq=model.config.d_cycle_update_freq,
            render_func=model._corgs_render_fn,  # imported render function
            pipe=model._pipe,
            bg=model._background,
            use_sh_reliability=model.config.use_sh_reliability,
            sh_stability_warmup=model.config.sh_stability_warmup,
            sh_stability_ema_beta=model.config.sh_stability_ema_beta,
            crs_w_s=model.config.crs_w_s,
            disable_r_signal=model.config.disable_r_signal,
        )

    return _step_fn


__all__ = ["make_crs_update_step_fn"]
