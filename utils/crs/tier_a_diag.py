# ============================================================
# [CRSGaussian Tier A] Formula-level diagnostics
# File: CRSGaussian/utils/crs/tier_a_diag.py  (TẠO MỚI)
# Mục đích: Trả lời "công thức CRS hiện tại có information value
#           không?" — không sửa training, chỉ instrument để dump
#           dữ liệu cho offline analysis.
#
# 4 test:
#   A1 — Dump distributions (D_i, R_i, CRS, opacity) tại milestone iter
#   A2 — Correlation D_i vs R_i (Pearson) — đã có ở log_di_ri_scatter,
#         analyzer sẽ tổng hợp từ A1 dumps
#   A3 — Synthetic floater: perturb position của Gaussian "tốt",
#         đo % bị CRS catch sau perturb
#   A4 — Occlusion contamination: % (Gaussian, view) pair được
#         R_i sample GT color trong khi Gaussian bị occlude
#
# Được gọi từ: train.py tại iter milestone, gated bởi --tier_a_diag.
# ============================================================

import os
import torch
from utils.graphics_utils import fov2focal


# ════════════════════════════════════
# A1 — Dump raw distributions
# ════════════════════════════════════

@torch.no_grad()
def tier_a_dump_distributions(gaussians, D, R, iteration, output_dir):
    """[CRSGaussian Tier A] A1 — Dump D_i, R_i, CRS, opacity tensors.

    Để analyzer offline tính histogram, bimodality, correlation.
    Lưu .pt file riêng cho mỗi iter milestone.

    Args:
        gaussians: GaussianModel — cần .get_crs (sigmoid của _crs_score)
                   và .get_opacity.
        D: (N, 1) tensor depth consistency từ update_crs return.
        R: (N, 1) tensor reprojection consistency từ update_crs return.
        iteration: int — số iter hiện tại (đặt tên file).
        output_dir: str — thư mục dump.
    """
    crs = gaussians.get_crs.detach().squeeze()
    opacity = gaussians.get_opacity.detach().squeeze()

    os.makedirs(output_dir, exist_ok=True)
    payload = {
        'iteration': iteration,
        'N': D.shape[0],
        'D': D.squeeze().cpu(),
        'R': R.squeeze().cpu(),
        'CRS': crs.cpu(),
        'opacity': opacity.cpu(),
    }
    path = f'{output_dir}/tier_a_dist_iter{iteration}.pt'
    torch.save(payload, path)

    # Quick console summary để debug runtime
    D_flat = D.squeeze()
    R_flat = R.squeeze()
    corr = torch.corrcoef(torch.stack([D_flat, R_flat]))[0, 1].item()
    print(f"[TIER A — A1] iter={iteration} | N={D.shape[0]} | "
          f"D=[{D_flat.mean():.3f}±{D_flat.std():.3f}] | "
          f"R=[{R_flat.mean():.3f}±{R_flat.std():.3f}] | "
          f"CRS=[{crs.mean():.3f}±{crs.std():.3f}] | "
          f"corr(D,R)={corr:.3f} | dumped→{path}")


# ════════════════════════════════════
# A3 — Synthetic floater discrimination
# ════════════════════════════════════

@torch.no_grad()
def tier_a_synthetic_floater_test(
    gaussians, allCameras, aligned_depth_dict, depth_range,
    iteration, output_dir,
    n_samples=5000, perturb_factor=0.3, seed=42
):
    """[CRSGaussian Tier A] A3 — Discrimination test.

    Lấy Gaussian "tốt" (CRS_orig > 0.5), perturb xyz random direction
    với magnitude = perturb_factor × depth_range, rồi recompute D, R, CRS
    với position mới. Đo % bị CRS catch (CRS_pert < 0.35) và % CRS drop ≥ 0.1.

    Discrimination tốt: % caught ≥ 80% — perturb 0.3*depth_range là khá lớn,
    nếu CRS không bắt được → công thức không có discrimination power.

    Note: Test này KHÔNG sửa gaussians thực sự — chỉ swap xyz tạm thời
    cho compute_*. Sau khi xong, gaussians không thay đổi.

    Args:
        gaussians: GaussianModel.
        allCameras: list[Camera] — training cameras.
        aligned_depth_dict: {cam.uid: Tensor (H,W)}.
        depth_range: float.
        iteration: int.
        output_dir: str.
        n_samples: số Gaussian sample (tối đa). Default 5000.
        perturb_factor: hệ số nhân với depth_range để tính magnitude.
                        Default 0.3 → khoảng cách lớn rõ rệt.
        seed: random seed cho reproducibility.
    """
    # Lazy import — tránh circular nếu crs_module import diag
    from utils.crs.crs_module import (
        compute_depth_consistency, compute_reprojection_consistency
    )

    torch.manual_seed(seed)
    device = gaussians.get_xyz.device

    xyz_orig = gaussians.get_xyz.detach().clone()
    crs_orig = gaussians.get_crs.detach().squeeze()
    N = xyz_orig.shape[0]

    # ── Sample known-good Gaussians (CRS > 0.5) ──
    # Lý do: nếu sample tất cả, một số Gaussian vốn đã là floater
    # (CRS thấp) → không học được gì. Test cần biết: "Gaussian tốt
    # bị move sai đi → CRS có catch không?"
    good_mask = crs_orig > 0.5
    good_idx_all = torch.where(good_mask)[0]
    if len(good_idx_all) < 100:
        # Fallback: scene chưa converge, lấy random thay
        good_idx = torch.randperm(N, device=device)[:min(n_samples, N)]
        sample_kind = "random_fallback"
    else:
        n = min(n_samples, len(good_idx_all))
        perm = torch.randperm(len(good_idx_all), device=device)[:n]
        good_idx = good_idx_all[perm]
        sample_kind = "high_crs"

    # ── Generate perturbations: random unit direction × magnitude ──
    direction = torch.randn(len(good_idx), 3, device=device)
    direction = direction / direction.norm(dim=1, keepdim=True).clamp(min=1e-6)
    magnitude = perturb_factor * depth_range
    perturbation = direction * magnitude

    # Tạo xyz_perturbed: thay tại idx được sample, giữ nguyên các Gaussian khác
    xyz_perturbed = xyz_orig.clone()
    xyz_perturbed[good_idx] = xyz_orig[good_idx] + perturbation

    # ── Recompute D_i, R_i với position perturbed ──
    # Hàm dưới đây nhận xyz làm arg, không phụ thuộc gaussians state
    D_pert = compute_depth_consistency(
        xyz_perturbed, allCameras, aligned_depth_dict, depth_range
    )
    R_pert = compute_reprojection_consistency(xyz_perturbed, allCameras)

    # ── Score → CRS qua công thức gốc (linear, scale=5.0) ──
    score_pert = 0.5 * D_pert + 0.5 * R_pert
    crs_logit_pert = 5.0 * (score_pert - 0.5)
    crs_pert_full = torch.sigmoid(crs_logit_pert).squeeze()

    # ── Subset: chỉ Gaussian được perturb ──
    crs_orig_sub = crs_orig[good_idx]
    crs_pert_sub = crs_pert_full[good_idx]
    D_pert_sub = D_pert.squeeze()[good_idx]
    R_pert_sub = R_pert.squeeze()[good_idx]

    # ── Verdict metrics ──
    n = len(good_idx)
    delta_crs = crs_orig_sub - crs_pert_sub  # positive = CRS dropped after perturb (good)

    n_caught = (crs_pert_sub < 0.35).sum().item()
    n_dropped_01 = (delta_crs > 0.1).sum().item()
    n_dropped_005 = (delta_crs > 0.05).sum().item()

    pct_caught = 100.0 * n_caught / n
    pct_dropped_01 = 100.0 * n_dropped_01 / n
    pct_dropped_005 = 100.0 * n_dropped_005 / n
    delta_mean = delta_crs.mean().item()

    # ── Dump ──
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        'iteration': iteration,
        'sample_kind': sample_kind,
        'n_samples': n,
        'perturb_factor': perturb_factor,
        'depth_range': depth_range,
        'magnitude': magnitude,
        'crs_orig': crs_orig_sub.cpu(),
        'crs_pert': crs_pert_sub.cpu(),
        'D_pert': D_pert_sub.cpu(),
        'R_pert': R_pert_sub.cpu(),
        'pct_caught_lt035': pct_caught,
        'pct_dropped_gt01': pct_dropped_01,
        'pct_dropped_gt005': pct_dropped_005,
        'delta_crs_mean': delta_mean,
    }
    path = f'{output_dir}/tier_a_synfloater_iter{iteration}.pt'
    torch.save(payload, path)

    print(f"[TIER A — A3] iter={iteration} | sample={sample_kind} | n={n} | "
          f"caught<0.35={pct_caught:.1f}% | "
          f"ΔCRS>0.1={pct_dropped_01:.1f}% | "
          f"ΔCRS_mean={delta_mean:+.3f} | "
          f"perturb={magnitude:.3f}")


# ════════════════════════════════════
# A4 — Occlusion contamination
# ════════════════════════════════════

@torch.no_grad()
def tier_a_occlusion_test(
    gaussians, allCameras, render_fn, pipe, bg,
    iteration, output_dir,
    occlusion_threshold=1.05
):
    """[CRSGaussian Tier A] A4 — Occlusion contamination rate.

    Hiện tại R_i sample GT color tại pixel projected của Gaussian center,
    KHÔNG check Gaussian có frontmost ở view đó không. Nếu bị Gaussian
    khác che (depth lớn hơn rendered depth) → color sample là của
    occluder, không phải surface của Gaussian này → R_i thấp giả tạo.

    Test này render scene để có depth map, rồi đếm % (Gaussian, view) pair
    được "visible" theo current criteria nhưng thực chất bị occluded
    (gauss_depth > rendered_depth × threshold).

    occluded_threshold=1.05: Gaussian center sâu hơn rendered surface
    >5% → coi là occluded (allow nhỏ epsilon cho noise).

    Args:
        gaussians: GaussianModel.
        allCameras: list[Camera].
        render_fn: gaussian_renderer.render — để gọi với (cam, gaussians, pipe, bg).
        pipe: PipelineParams.
        bg: background tensor.
        iteration: int.
        output_dir: str.
        occlusion_threshold: float — gauss_d > render_d × threshold → occluded.
    """
    xyz = gaussians.get_xyz.detach()
    N = xyz.shape[0]
    device = xyz.device

    # Chỉ process cameras có GT image (skip PseudoCamera)
    valid_cams = [c for c in allCameras if hasattr(c, 'original_image')]
    if len(valid_cams) == 0:
        print(f"[TIER A — A4] iter={iteration} | NO valid cameras")
        return

    # Homogeneous coords cho projection
    ones = torch.ones(N, 1, device=device, dtype=xyz.dtype)
    xyz_hom = torch.cat([xyz, ones], dim=1)

    total_visible = 0       # tổng (Gaussian, cam) pair theo R_i visibility (depth>0 + in-bounds)
    total_occluded = 0      # subset bị occluded
    per_cam_stats = []      # list of (cam_uid, n_visible, n_occluded)

    for cam in valid_cams:
        H, W = cam.image_height, cam.image_width

        # ── Render depth map ──
        # Renderer trả "depth": (1, H, W) hoặc (H, W).
        out = render_fn(cam, gaussians, pipe, bg)
        depth_map = out["depth"]
        if depth_map.dim() == 3:
            depth_map = depth_map.squeeze(0)  # (H, W)

        # ── Project Gaussians (cùng pattern với compute_*) ──
        W2C = cam.world_view_transform.T
        pts_cam = (W2C @ xyz_hom.T).T
        gauss_depth = pts_cam[:, 2]

        fx = fov2focal(cam.FoVx, W)
        fy = fov2focal(cam.FoVy, H)
        cx, cy = W / 2.0, H / 2.0
        pixel_x = pts_cam[:, 0] / gauss_depth * fx + cx
        pixel_y = pts_cam[:, 1] / gauss_depth * fy + cy

        # Visibility filter: cùng logic với compute_reprojection_consistency
        valid = (
            (gauss_depth > 0)
            & (pixel_x >= 0) & (pixel_x < W)
            & (pixel_y >= 0) & (pixel_y < H)
        )
        n_visible = int(valid.sum().item())
        if n_visible == 0:
            per_cam_stats.append((cam.uid, 0, 0))
            continue

        px = pixel_x[valid].long().clamp(0, W - 1)
        py = pixel_y[valid].long().clamp(0, H - 1)

        d_render = depth_map[py, px]      # rendered depth tại projected pixel
        d_gauss = gauss_depth[valid]      # Gaussian center depth in cam space

        # ── Occlusion test ──
        # Gaussian center sâu hơn rendered surface đáng kể → bị che.
        # rendered_depth ≈ alpha-weighted avg depth → frontmost surface dominate
        # Có thể rendered_depth = 0 nếu pixel ngoài coverage Gaussian → skip
        valid_render = d_render > 1e-6
        occluded = valid_render & (d_gauss > d_render * occlusion_threshold)
        n_occluded = int(occluded.sum().item())

        total_visible += n_visible
        total_occluded += n_occluded
        per_cam_stats.append((cam.uid, n_visible, n_occluded))

    pct_occluded = 100.0 * total_occluded / max(total_visible, 1)

    os.makedirs(output_dir, exist_ok=True)
    payload = {
        'iteration': iteration,
        'total_visible_pairs': total_visible,
        'occluded_pairs': total_occluded,
        'pct_occluded': pct_occluded,
        'per_cam_stats': per_cam_stats,
        'occlusion_threshold': occlusion_threshold,
        'n_cameras': len(valid_cams),
    }
    path = f'{output_dir}/tier_a_occlusion_iter{iteration}.pt'
    torch.save(payload, path)

    print(f"[TIER A — A4] iter={iteration} | "
          f"visible_pairs={total_visible} | "
          f"occluded={total_occluded} ({pct_occluded:.1f}%) | "
          f"threshold={occlusion_threshold} | dumped→{path}")
