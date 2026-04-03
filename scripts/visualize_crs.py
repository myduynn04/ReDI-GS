# ============================================================
# [CRSGaussian] Task: T2.7 — Visualize CRS distribution
# File: CRSGaussian/scripts/visualize_crs.py  (TẠO MỚI)
# Mục đích: Sau training, visualize CRS per-Gaussian:
#   1. Histogram CRS → xem separation floater vs surface
#   2. Point cloud .ply tô màu theo CRS → xem 3D spatial
#   3. Stats per band (floater / undecided / surface)
# Chạy: cd CRSGaussian && python scripts/visualize_crs.py \
#          --model_path output/fern_crs_log --iteration 3000
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend cho server
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from plyfile import PlyData, PlyElement


def load_gaussians_from_checkpoint(model_path, iteration):
    """Load xyz và _crs_score từ checkpoint."""
    chkpnt_path = os.path.join(model_path, f"chkpnt{iteration}.pth")
    if not os.path.exists(chkpnt_path):
        raise FileNotFoundError(f"Checkpoint not found: {chkpnt_path}")

    model_args, _ = torch.load(chkpnt_path, map_location="cpu")

    # capture() returns tuple: (active_sh_degree, _xyz, _features_dc,
    #   _features_rest, _scaling, _rotation, _opacity, max_radii2D,
    #   xyz_gradient_accum, denom, optimizer_state_dict, spatial_lr_scale,
    #   _crs_score)
    if len(model_args) == 13:
        xyz = model_args[1].detach().cpu().numpy()          # (N, 3)
        opacity_logit = model_args[6].detach().cpu().numpy() # (N, 1)
        crs_logit = model_args[12].detach().cpu().numpy()    # (N, 1)
    else:
        xyz = model_args[1].detach().cpu().numpy()
        opacity_logit = model_args[6].detach().cpu().numpy()
        crs_logit = None

    # sigmoid để chuyển về [0, 1]
    opacity = 1.0 / (1.0 + np.exp(-opacity_logit))
    crs = 1.0 / (1.0 + np.exp(-crs_logit)) if crs_logit is not None else None

    return xyz, opacity.squeeze(), crs.squeeze() if crs is not None else None


def save_histogram(crs, output_path, tau_crs=0.35, tau_densify=0.45):
    """Save CRS histogram với threshold lines."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: full histogram
    ax = axes[0]
    ax.hist(crs, bins=100, range=(0, 1), color='steelblue', alpha=0.8,
            edgecolor='white', linewidth=0.3)
    ax.axvline(tau_crs, color='red', linestyle='--', linewidth=1.5,
               label=f'tau_crs={tau_crs}')
    ax.axvline(tau_densify, color='orange', linestyle='--', linewidth=1.5,
               label=f'tau_densify={tau_densify}')
    ax.set_xlabel('CRS')
    ax.set_ylabel('Count')
    ax.set_title(f'CRS Distribution (N={len(crs):,})')
    ax.legend()

    # Right: log-scale to see tails
    ax = axes[1]
    ax.hist(crs, bins=100, range=(0, 1), color='steelblue', alpha=0.8,
            edgecolor='white', linewidth=0.3)
    ax.axvline(tau_crs, color='red', linestyle='--', linewidth=1.5,
               label=f'tau_crs={tau_crs}')
    ax.axvline(tau_densify, color='orange', linestyle='--', linewidth=1.5,
               label=f'tau_densify={tau_densify}')
    ax.set_xlabel('CRS')
    ax.set_ylabel('Count (log)')
    ax.set_yscale('log')
    ax.set_title('CRS Distribution (log scale)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[T2.7] Histogram saved: {output_path}")


def save_colored_ply(xyz, crs, output_path):
    """Save point cloud tô màu theo CRS.
    Đỏ = CRS thấp (floater), xanh lá = CRS cao (surface).
    Mở bằng MeshLab / CloudCompare để xem 3D.
    """
    N = xyz.shape[0]

    # Color map: red (0) → yellow (0.5) → green (1)
    # Dùng matplotlib colormap
    cmap = plt.cm.RdYlGn  # Red-Yellow-Green
    colors = (cmap(crs)[:, :3] * 255).astype(np.uint8)  # (N, 3) RGB

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    elements = np.empty(N, dtype=dtype)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['red'] = colors[:, 0]
    elements['green'] = colors[:, 1]
    elements['blue'] = colors[:, 2]

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)
    print(f"[T2.7] Colored PLY saved: {output_path}")


def save_floater_ply(xyz, crs, output_path, tau=0.35):
    """Save CHỈ Gaussians có CRS < tau — để xem floaters riêng."""
    mask = crs < tau
    if mask.sum() == 0:
        print(f"[T2.7] No Gaussians with CRS < {tau}")
        return

    xyz_f = xyz[mask]
    crs_f = crs[mask]
    N = xyz_f.shape[0]

    # Tất cả đỏ — floater candidates
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    elements = np.empty(N, dtype=dtype)
    elements['x'] = xyz_f[:, 0]
    elements['y'] = xyz_f[:, 1]
    elements['z'] = xyz_f[:, 2]
    elements['red'] = 255
    elements['green'] = 0
    elements['blue'] = 0

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)
    print(f"[T2.7] Floater PLY saved: {output_path} (N={N})")


def print_band_stats(crs, opacity, tau_crs=0.35, tau_densify=0.45):
    """In stats chi tiết theo band."""
    N = len(crs)
    bands = [
        ("Floater (CRS<0.35)", crs < tau_crs),
        ("Risky (0.35-0.45)", (crs >= tau_crs) & (crs < tau_densify)),
        ("Undecided (0.45-0.65)", (crs >= tau_densify) & (crs < 0.65)),
        ("Surface (CRS>0.65)", crs >= 0.65),
    ]

    print(f"\n{'='*65}")
    print(f"[T2.7] CRS Band Analysis — N={N:,}")
    print(f"{'='*65}")
    print(f"{'Band':<25} {'Count':>8} {'%':>7} {'Opacity mean':>13} {'CRS mean':>9}")
    print(f"{'-'*65}")

    for name, mask in bands:
        count = mask.sum()
        pct = count / N * 100
        op_mean = opacity[mask].mean() if count > 0 else 0
        crs_mean = crs[mask].mean() if count > 0 else 0
        print(f"{name:<25} {count:>8,} {pct:>6.1f}% {op_mean:>13.4f} {crs_mean:>9.4f}")

    print(f"{'='*65}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", "-m", required=True,
                        help="Path to training output (e.g. output/fern_crs_log)")
    parser.add_argument("--iteration", type=int, default=3000,
                        help="Checkpoint iteration to load")
    parser.add_argument("--tau_crs", type=float, default=0.35)
    parser.add_argument("--tau_densify", type=float, default=0.45)
    args = parser.parse_args()

    # ── Load data ──
    xyz, opacity, crs = load_gaussians_from_checkpoint(
        args.model_path, args.iteration)

    if crs is None:
        print("ERROR: Checkpoint does not contain _crs_score (old format)")
        sys.exit(1)

    # ── Output dir ──
    out_dir = os.path.join(args.model_path, "crs_analysis")
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Band stats ──
    print_band_stats(crs, opacity, args.tau_crs, args.tau_densify)

    # ── 2. Histogram ──
    save_histogram(crs, os.path.join(out_dir, "crs_histogram.png"),
                   args.tau_crs, args.tau_densify)

    # ── 3. Full point cloud colored by CRS ──
    save_colored_ply(xyz, crs, os.path.join(out_dir, "crs_colored.ply"))

    # ── 4. Floater-only point cloud ──
    save_floater_ply(xyz, crs, os.path.join(out_dir, "floaters_only.ply"),
                     tau=args.tau_crs)

    print(f"\n[T2.7] All outputs saved to: {out_dir}/")
    print(f"  - crs_histogram.png  → xem separation")
    print(f"  - crs_colored.ply    → mở MeshLab, xem toàn cảnh")
    print(f"  - floaters_only.ply  → mở MeshLab, xem floaters riêng")
