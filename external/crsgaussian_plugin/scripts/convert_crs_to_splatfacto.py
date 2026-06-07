#!/usr/bin/env python
"""
[B6 viewer realtime] Convert crsgaussian checkpoint → splatfacto-sparse checkpoint format.

Mục đích: Cho phép xem anh's Phase 22 Gaussians qua splatfacto viewer
(gsplat rasterizer = smooth 30-60 FPS), KHÔNG bị lag Inria rasterizer 14 FPS.

Cách dùng:
    python convert_crs_to_splatfacto.py \\
        --src outputs/phase22_b4/leaves/crsgaussian/<timestamp>/nerfstudio_models/step-000009999.ckpt \\
        --dst outputs/splatfacto_3view/leaves/splatfacto-sparse/<timestamp>/nerfstudio_models/step-000009999.ckpt

Sau đó open viewer:
    RUN=$(ls -td outputs/splatfacto_3view/leaves/splatfacto-sparse/* | head -1)
    ns-viewer --load-config $RUN/config.yml --viewer.websocket-port 7008
    # → Viewer load anh's Gaussians qua gsplat → smooth realtime

CAUTION: overwrite splatfacto-sparse checkpoint. Backup nếu cần giữ.
"""

import argparse
import shutil
from pathlib import Path

import torch


def convert(src_path: Path, dst_path: Path):
    print(f"═══ Loading source (crsgaussian) ═══")
    print(f"  {src_path}")
    src_ckpt = torch.load(str(src_path), map_location="cpu", weights_only=False)

    # Extract gauss params từ _extra_state tuple (CoR-GS capture() format)
    # Tuple: (active_sh_degree, _xyz, _features_dc, _features_rest, _scaling,
    #         _rotation, _opacity, max_radii2D, xyz_grad_accum, denom, opt_dict,
    #         spatial_lr_scale, _crs_score)
    ext = src_ckpt["pipeline"]["_model._extra_state"]

    active_sh = ext[0]
    xyz = ext[1]            # (N, 3)
    feat_dc = ext[2]        # (N, 1, 3) — squeeze for splatfacto
    feat_rest = ext[3]      # (N, 15, 3)
    scaling = ext[4]        # (N, 3) log space
    rotation = ext[5]       # (N, 4) quaternion
    opacity = ext[6]        # (N, 1) logit space

    N = xyz.shape[0]
    print(f"  N_gauss = {N:,}")
    print(f"  active_sh_degree = {active_sh}")
    print(f"  features_dc shape = {feat_dc.shape} (will squeeze dim 1 for splatfacto)")

    print(f"\n═══ Loading destination (splatfacto-sparse) — preserve config ═══")
    print(f"  {dst_path}")
    dst_ckpt = torch.load(str(dst_path), map_location="cpu", weights_only=False)

    # ── Backup dst before overwrite ──
    backup_path = dst_path.with_suffix(".ckpt.original")
    if not backup_path.exists():
        shutil.copy(dst_path, backup_path)
        print(f"  Backup → {backup_path}")

    print(f"\n═══ Converting (rename keys) ═══")
    # Splatfacto gauss_params keys
    new_pipeline = dst_ckpt["pipeline"].copy()

    # Overwrite the 6 gauss param tensors
    new_pipeline["_model.gauss_params.means"] = xyz.contiguous()
    new_pipeline["_model.gauss_params.features_dc"] = feat_dc.squeeze(1).contiguous()  # (N,1,3)→(N,3)
    new_pipeline["_model.gauss_params.features_rest"] = feat_rest.contiguous()
    new_pipeline["_model.gauss_params.scales"] = scaling.contiguous()
    new_pipeline["_model.gauss_params.quats"] = rotation.contiguous()
    new_pipeline["_model.gauss_params.opacities"] = opacity.contiguous()

    # ── Reset optimizer state (KHÔNG còn relevant cho Splatfacto's optimizer schedule) ──
    # Loại bỏ các key liên quan optimizer cũ để Splatfacto reload fresh
    keys_to_drop = [k for k in new_pipeline.keys() if "optimizer" in k or "exposure" in k]
    for k in keys_to_drop:
        del new_pipeline[k]

    dst_ckpt["pipeline"] = new_pipeline

    # ── Save ──
    torch.save(dst_ckpt, str(dst_path))
    print(f"\n═══ Done ═══")
    print(f"  Output: {dst_path}")
    print(f"  N_gauss = {N:,}")
    print(f"\n  Run viewer:")
    print(f"    RUN={dst_path.parent.parent.parent}")
    print(f"    ns-viewer --load-config $RUN/config.yml --viewer.websocket-port 7007")
    print(f"  → Smooth gsplat realtime!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="crsgaussian source ckpt")
    parser.add_argument("--dst", type=Path, required=True, help="splatfacto-sparse dst ckpt (overwrite)")
    args = parser.parse_args()

    if not args.src.is_file():
        raise FileNotFoundError(f"Source not found: {args.src}")
    if not args.dst.is_file():
        raise FileNotFoundError(f"Destination not found: {args.dst}")

    convert(args.src, args.dst)
