"""
Standalone script to extract CRS distribution from Phase 28b checkpoints.

Usage:
    cd ~/workspace/representation-3d/duyen/CoR-GS
    python scripts/dump_crs_histogram.py

Output:
    - Printed histogram text (per scene)
    - Saved .npz files for offline plotting
"""

import torch
import numpy as np
from pathlib import Path


def load_crs_from_checkpoint(ckpt_path):
    """Load _crs_score from chkpnt*.pth and return CRS values in [0, 1]."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # Capture format: (model_args, iteration) where model_args is a tuple
    if isinstance(ckpt, tuple) and len(ckpt) == 2:
        model_args = ckpt[0]
    else:
        model_args = ckpt

    # Find _crs_score in model_args (last element per gaussian_model.py line 183)
    # Capture returns: (active_sh_degree, _xyz, _features_dc, _features_rest,
    #                   _scaling, _rotation, _opacity, max_radii2D, xyz_grad_accum,
    #                   xyz_grad_accum_abs, xyz_grad_accum_abs_max, denom, opt_dict,
    #                   spatial_lr_scale, _crs_score)
    crs_logits = model_args[-1]  # Last element is _crs_score

    # Convert logits to CRS in [0, 1]
    crs = torch.sigmoid(crs_logits).squeeze().numpy()
    return crs


def print_histogram(crs, scene_name, n_bins=20):
    """Print ASCII histogram."""
    counts, edges = np.histogram(crs, bins=n_bins, range=(0, 1))
    N = len(crs)
    max_count = counts.max()
    bar_width = 50

    print(f"\n{'='*70}")
    print(f"  Scene: {scene_name}  (N={N:,} Gaussians)")
    print(f"  mean={crs.mean():.4f}  std={crs.std():.4f}  "
          f"min={crs.min():.4f}  max={crs.max():.4f}")
    print(f"  median={np.median(crs):.4f}  "
          f"p10={np.percentile(crs, 10):.4f}  "
          f"p90={np.percentile(crs, 90):.4f}")
    print(f"{'='*70}")
    print(f"  Bin range          | Count    | %     | Bar")
    print(f"  -----------------  | -------- | ----- | {'-'*bar_width}")
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        c = counts[i]
        pct = 100 * c / N
        bar_len = int(bar_width * c / max_count) if max_count > 0 else 0
        bar = '█' * bar_len
        print(f"  [{lo:.2f}, {hi:.2f})    | {c:>8,d} | {pct:>4.1f}% | {bar}")
    print(f"{'='*70}")


def main():
    scenes = ['trex', 'fortress', 'fern']
    all_data = {}

    for sc in scenes:
        # Try Phase 28b path first, then other phases
        candidates = [
            Path(f"/tmp/p28b_diag_{sc}/chkpnt10000.pth"),
            Path(f"output/p28_crs_boost/tau65/A3_seed42_{sc}/chkpnt10000.pth"),
            Path(f"output/p27_pillar/trim_full/A3_seed42_{sc}/chkpnt10000.pth"),
        ]

        ckpt_path = None
        for c in candidates:
            if c.exists():
                ckpt_path = c
                break

        if ckpt_path is None:
            print(f"\n⚠️  No checkpoint found for {sc}. Checked:")
            for c in candidates:
                print(f"   - {c}")
            continue

        print(f"\nLoading {sc} from {ckpt_path}...")
        crs = load_crs_from_checkpoint(ckpt_path)
        all_data[sc] = crs

        print_histogram(crs, sc)

    # Save raw data for offline plotting
    if all_data:
        output_path = Path("logs/p28b_diag/crs_histograms.npz")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **all_data)
        print(f"\n✓ Saved raw CRS values to {output_path}")
        print(f"  Load with: data = np.load('{output_path}')")
        print(f"  Access: data['{scenes[0]}'], etc.")

    # Summary comparison
    if all_data:
        print(f"\n{'='*70}")
        print(f"  CROSS-SCENE COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Scene':<12} {'N':>10} {'mean':>8} {'std':>8} {'p10':>8} {'p90':>8}")
        print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for sc, crs in all_data.items():
            print(f"  {sc:<12} {len(crs):>10,} {crs.mean():>8.4f} {crs.std():>8.4f} "
                  f"{np.percentile(crs, 10):>8.4f} {np.percentile(crs, 90):>8.4f}")


if __name__ == "__main__":
    main()
