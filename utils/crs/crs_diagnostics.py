# ============================================================
# [CRSGaussian] CRS Diagnostic Logging
# File: CRSGaussian/utils/crs/crs_diagnostics.py  (TẠO MỚI)
# Mục đích: Quan sát D_i, R_i, CRS trong training để validate
#           công thức, hyperparameters, và pruning effectiveness.
# Được gọi từ: train.py tại CRS update block và pruning block.
# ============================================================

import os
import torch
import numpy as np
from utils.graphics_utils import fov2focal


# ════════════════════════════════════
# DIAGNOSTIC 1 — Scatter D_i vs R_i
# ════════════════════════════════════

@torch.no_grad()
def log_di_ri_scatter(D, R, iteration, tb_writer, output_dir):
    """[CRSGaussian] Scatter plot D_i vs R_i + correlation.

    Mục đích: kiểm tra 2 signals có correlation không.
    - Diagonal → tương đồng → equal weight OK
    - Cluster D thấp/R cao (hoặc ngược lại) → cần weight khác
    """
    D_flat = D.squeeze()
    R_flat = R.squeeze()

    # Correlation
    corr = torch.corrcoef(torch.stack([D_flat, R_flat]))[0, 1].item()

    # Console log
    print(f"[CRS DIAG] iter={iteration} | D_R_corr={corr:.3f} | "
          f"D: mean={D_flat.mean():.3f} std={D_flat.std():.3f} | "
          f"R: mean={R_flat.mean():.3f} std={R_flat.std():.3f}")

    # TensorBoard
    if tb_writer:
        tb_writer.add_scalar('crs_diag/D_R_correlation', corr, iteration)
        tb_writer.add_scalar('crs_diag/D_mean', D_flat.mean().item(), iteration)
        tb_writer.add_scalar('crs_diag/R_mean', R_flat.mean().item(), iteration)
        tb_writer.add_scalar('crs_diag/D_std', D_flat.std().item(), iteration)
        tb_writer.add_scalar('crs_diag/R_std', R_flat.std().item(), iteration)

    # Scatter plot PNG
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        # Sample để plot không quá chậm
        n = len(D_flat)
        idx = torch.randperm(n)[:min(5000, n)]
        ax.scatter(D_flat[idx].cpu().numpy(), R_flat[idx].cpu().numpy(),
                   alpha=0.3, s=1, c='steelblue')
        ax.set_xlabel('D_i (depth consistency)')
        ax.set_ylabel('R_i (reprojection consistency)')
        ax.set_title(f'Iter {iteration} | corr={corr:.3f} | N={n}')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(f'{output_dir}/scatter_DiRi_iter{iteration}.png', dpi=100,
                    bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"[CRS DIAG] scatter plot failed: {e}")


# ════════════════════════════════════
# DIAGNOSTIC 2 — CRS Stability
# ════════════════════════════════════

@torch.no_grad()
def log_crs_stability(gaussians, D, R, iteration, tb_writer):
    """[CRSGaussian] CRS stability theo nhóm Gaussian.

    Mục đích: CRS có phân biệt rõ surface vs floater không?
    Track số lượng và mean CRS của high/low group qua iter.
    """
    crs = gaussians.get_crs.detach().squeeze()  # (N,)
    N = crs.shape[0]
    D_flat = D.squeeze()
    R_flat = R.squeeze()

    high_crs = crs > 0.65
    low_crs = crs < 0.35
    mid_crs = ~high_crs & ~low_crs

    n_high = high_crs.sum().item()
    n_low = low_crs.sum().item()
    n_mid = mid_crs.sum().item()

    if tb_writer:
        # D_i, R_i stats
        tb_writer.add_scalar('crs_diag/D_mean', D_flat.mean().item(), iteration)
        tb_writer.add_scalar('crs_diag/R_mean', R_flat.mean().item(), iteration)

        # CRS group stats
        tb_writer.add_scalar('crs_diag/n_total', N, iteration)
        tb_writer.add_scalar('crs_diag/n_high_crs', n_high, iteration)
        tb_writer.add_scalar('crs_diag/n_low_crs', n_low, iteration)
        tb_writer.add_scalar('crs_diag/n_mid_crs', n_mid, iteration)

        if n_high > 0:
            tb_writer.add_scalar('crs_diag/crs_high_mean',
                                 crs[high_crs].mean().item(), iteration)
        if n_low > 0:
            tb_writer.add_scalar('crs_diag/crs_low_mean',
                                 crs[low_crs].mean().item(), iteration)

        # Overall CRS
        tb_writer.add_scalar('crs_diag/crs_mean', crs.mean().item(), iteration)
        tb_writer.add_scalar('crs_diag/crs_std', crs.std().item(), iteration)


# ════════════════════════════════════
# DIAGNOSTIC 3 — CRS Heatmap
# ════════════════════════════════════

@torch.no_grad()
def render_crs_heatmap(gaussians, camera, render_fn, pipe, bg,
                       iteration, output_dir):
    """[CRSGaussian] Render CRS heatmap lên camera view.

    Vùng xanh (CRS cao) = surface. Vùng đỏ (CRS thấp) = floater.
    Project Gaussian positions lên image plane, average CRS tại pixel.
    """
    H = camera.image_height
    W = camera.image_width
    device = gaussians.get_xyz.device

    xyz = gaussians.get_xyz.detach()  # (N, 3)
    crs = gaussians.get_crs.detach().squeeze()  # (N,)
    N = xyz.shape[0]

    # Project all Gaussians → pixel coords
    ones = torch.ones(N, 1, device=device, dtype=xyz.dtype)
    xyz_hom = torch.cat([xyz, ones], dim=1)
    W2C = camera.world_view_transform.T
    pts_cam = (W2C @ xyz_hom.T).T
    depth = pts_cam[:, 2]

    fx = fov2focal(camera.FoVx, W)
    fy = fov2focal(camera.FoVy, H)
    cx, cy = W / 2.0, H / 2.0

    pixel_x = pts_cam[:, 0] / depth * fx + cx
    pixel_y = pts_cam[:, 1] / depth * fy + cy

    valid = ((depth > 0) & (pixel_x >= 0) & (pixel_x < W)
             & (pixel_y >= 0) & (pixel_y < H))

    if valid.sum() == 0:
        return

    px = pixel_x[valid].long().clamp(0, W - 1)
    py = pixel_y[valid].long().clamp(0, H - 1)
    crs_v = crs[valid]

    # Accumulate CRS per pixel
    crs_map = torch.zeros(H, W, device=device)
    count_map = torch.zeros(H, W, device=device)
    crs_map.index_put_((py, px), crs_v, accumulate=True)
    count_map.index_put_((py, px), torch.ones_like(crs_v), accumulate=True)

    valid_px = count_map > 0
    crs_map[valid_px] /= count_map[valid_px]
    crs_map[~valid_px] = 0.5  # neutral cho pixel trống

    # Also render GT image for side-by-side
    out = render_fn(camera, gaussians, pipe, bg)
    rgb = torch.clamp(out["render"], 0, 1)  # (3, H, W)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: rendered image
        axes[0].imshow(rgb.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title(f'Rendered — Iter {iteration}')
        axes[0].axis('off')

        # Right: CRS heatmap
        im = axes[1].imshow(crs_map.cpu().numpy(), cmap='RdYlGn',
                            vmin=0, vmax=1)
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        axes[1].set_title(f'CRS Heatmap — Iter {iteration} | N={N}')
        axes[1].axis('off')

        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(f'{output_dir}/crs_heatmap_iter{iteration}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"[CRS DIAG] heatmap failed: {e}")


# ════════════════════════════════════
# DIAGNOSTIC 4 — DAV2 Noise vs SfM
# ════════════════════════════════════

@torch.no_grad()
def log_dav2_noise(aligned_depth_dict, cameras, source_path, n_views,
                   depth_range, tb_writer):
    """[CRSGaussian] Đo noise DAV2 aligned depth so với COLMAP sparse points.

    Nếu L1 error lớn → D_i noisy → nên giảm weight D_i.
    Chạy 1 lần sau precompute.
    """
    from utils.depth.depth_alignment import _load_colmap_points, _project_points_to_camera

    sparse_xyz, sparse_errors = _load_colmap_points(source_path, n_views)
    errors_list = []

    for cam in cameras:
        if cam.uid not in aligned_depth_dict:
            continue
        d_prior = aligned_depth_dict[cam.uid]  # (H, W) GPU
        H, W = d_prior.shape

        pixels, depths_colmap = _project_points_to_camera(sparse_xyz, cam, H, W)
        if len(depths_colmap) == 0:
            continue

        px = pixels[:, 0].long().clamp(0, W - 1)
        py = pixels[:, 1].long().clamp(0, H - 1)
        d_dav2 = d_prior[py, px].cpu()
        d_sfm = depths_colmap.cpu()

        # Normalized L1 error
        l1 = torch.abs(d_dav2 - d_sfm) / max(depth_range, 1e-8)
        errors_list.append(l1)

    if errors_list:
        errors = torch.cat(errors_list).numpy()
        print(f"[CRS DIAG] DAV2 vs SfM normalized L1: "
              f"mean={errors.mean():.4f} std={errors.std():.4f} "
              f"median={np.median(errors):.4f} "
              f">0.1={((errors > 0.1).mean() * 100):.1f}%")
        if tb_writer:
            tb_writer.add_scalar('crs_diag/dav2_noise_mean', errors.mean(), 0)
            tb_writer.add_scalar('crs_diag/dav2_noise_std', errors.std(), 0)
            tb_writer.add_scalar('crs_diag/dav2_noise_median', np.median(errors), 0)


# ════════════════════════════════════
# DIAGNOSTIC 5 — Pruning Reason Stats
# ════════════════════════════════════

@torch.no_grad()
def log_pruning_stats(n_before, n_after, iteration, tb_writer):
    """[CRSGaussian] Log số Gaussians trước/sau pruning.

    Mục đích: pruning có hiệu quả không? Bao nhiêu bị xóa mỗi lần?
    """
    n_pruned = n_before - n_after
    if n_before == 0:
        return

    pct = n_pruned / n_before * 100
    print(f"[CRS DIAG] iter={iteration} | "
          f"before={n_before} after={n_after} pruned={n_pruned} ({pct:.1f}%)")

    if tb_writer:
        tb_writer.add_scalar('crs_diag/n_before_prune', n_before, iteration)
        tb_writer.add_scalar('crs_diag/n_after_prune', n_after, iteration)
        tb_writer.add_scalar('crs_diag/n_pruned', n_pruned, iteration)
        tb_writer.add_scalar('crs_diag/pct_pruned', pct, iteration)
