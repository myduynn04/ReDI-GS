# ============================================================
# [CRSGaussian Phase 17 — C1] DSINE monocular-normal prior loss (C1a).
# File: utils/loss/c1_normal.py  (TẠO MỚI)
# Gọi từ: train.py, gated bởi --use_c1_normal (default OFF).
#
# ── HONEST FRAMING (verify-from-code 2026-05-19, user-corrected) ──
# C1a = ADAPTATION CỐ Ý, **KHÔNG** phải "mirror dn-splatter exactly":
#   • dn-splatter dn_model.py:590 dùng `depth_im.detach()` cho
#     surface_normal → KHÔNG differentiable; normal được supervise của
#     dn-splatter = `normals_im` = CUDA-rasterized per-Gaussian normal
#     (= C1b). CRSGaussian renderer KHÔNG có normal output → C1b = sửa
#     CUDA (loại). C1a tự làm normal-từ-rendered-depth **differentiable**
#     + L1 với DSINE → back-prop qua depth→normal (dn-splatter detach
#     ĐÚNG chỗ này) = deviation, no-CUDA, recipe-risk thấp.
#   • MIRROR thật (verified) = CHỈ các transform: surface =
#     pcd_to_normal(backproj, c2w=eye=camera-frame) @ diag([1,-1,-1])
#     → [0,1] (dn_model.py:589-603); target DSINE = raw @ diag([-1,1,1])
#     (LUF→RUF, run_monocular_dsine) → [0,1]. 2 recipe cùng camera-[0,1]
#     convention ⟹ L1 hợp lệ KHÔNG cần world/wvt/orient (né sạch ác mộng
#     convention đã phá p16 Gate). Loss = L1 (step<15k của dn-splatter
#     AdaptiveNormal cũng L1 thuần; pilot 10k → luôn L1).
#
# ── PRE-REGISTERED PREDICTION (logged TRƯỚC khi chạy pilot) ──
#   dn-splatter detach depth ở surface_normal CHÍNH XÁC vì ∇(finite-diff
#   cross-product của rendered depth) cực nhiễu — đúng nhiễu vừa phá
#   n_dep của p16 Gate. C1a back-prop qua đúng operator đó trên 3DGS
#   sparse rendered-depth (floaty nhất) → DỰ ĐOÁN: gradient-normal nhiễu
#   → nhiều khả năng degrade / no-gain, tập trung scene floaty-depth
#   (horns, foliage). Nếu pilot ra horns-catastrophe / saturate signature
#   → mechanism XÁC NHẬN, KHÔNG bất ngờ, KHÔNG re-engineer. Guard
#   horns + tiêu chí pre-registered (decisions_log Phase-17) bắt thực
#   nghiệm. smooth=False (mirror dn-splatter) — KHÔNG lén thêm mitigation.
# ============================================================

import torch
import torch.nn.functional as F


def _normal_from_depth_cam(depth, fx, fy, cx, cy):
    """Rendered-depth (1,H,W or H,W) → CAMERA-frame normal (H,W,3),
    DIFFERENTIABLE wrt depth. Backproject với c2w = identity (giữ
    camera frame, mirror dn_model.py normal_from_depth_image c2w=eye),
    rồi cross-product lân cận 1-pixel (= dn-splatter pcd_to_normal,
    verified normal_utils.py). NaN-free: trả normal thô, caller mask."""
    if depth.dim() == 3:
        depth = depth.squeeze(0)                       # (H,W)
    H, W = depth.shape
    dev = depth.device
    vv, uu = torch.meshgrid(
        torch.arange(H, device=dev, dtype=depth.dtype),
        torch.arange(W, device=dev, dtype=depth.dtype),
        indexing="ij")
    u = uu + 0.5
    v = vv + 0.5
    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    xyz = torch.stack([x, y, depth], dim=-1)           # (H,W,3) camera frame

    # pcd_to_normal: cross(right−left, top−bottom) (verified dn-splatter
    # normal_utils.pcd_to_normal indexing), pad biên = 0.
    left = xyz[1:-1, 0:-2, :]
    right = xyz[1:-1, 2:, :]
    top = xyz[0:-2, 1:-1, :]
    bottom = xyz[2:, 1:-1, :]
    n = torch.cross(right - left, top - bottom, dim=-1)
    n = F.normalize(n, p=2, dim=-1, eps=1e-8)
    n = F.pad(n.permute(2, 0, 1), (1, 1, 1, 1),
              mode="constant").permute(1, 2, 0)         # (H,W,3)
    return n


def compute_c1_normal_loss(rendered_depth, rendered_alpha, dsine_target,
                           fx, fy, cx, cy, lam):
    """[Phase 17 C1a] L = lam · L1( n_render_cam_[0,1] , DSINE_[0,1] ).

    Args:
        rendered_depth: (1,H,W) GPU — RenderDict["depth_gs0"] (accumulated).
        rendered_alpha: (1,H,W) GPU — RenderDict["alpha_gs0"].
        dsine_target:   (H,W,3) GPU [0,1] — precomputed DSINE normal,
                        camera-frame (raw@diag([-1,1,1])→(x+1)/2). DETACHED.
        fx,fy,cx,cy:    camera intrinsics (render resolution).
        lam:            λ (primary pre-registered = 0.10).

    Returns scalar tensor (already × lam) hoặc None nếu shape mismatch /
    no valid pixel (caller skip — KHÔNG cộng vào loss)."""
    # alpha-correct depth (Σwz / Σw), differentiable. Mirror p16/Gate +
    # dn-splatter alpha handling.
    depth = (rendered_depth / (rendered_alpha + 1e-6)).squeeze(0)   # (H,W)
    H, W = depth.shape
    if dsine_target.shape[0] != H or dsine_target.shape[1] != W:
        return None                                    # res mismatch → skip

    n_cam = _normal_from_depth_cam(depth, fx, fy, cx, cy)           # (H,W,3)
    # dn-splatter dn_model.py:600-603 recipe: @ diag([1,-1,-1]) → [0,1].
    n_cam = n_cam * torch.tensor([1.0, -1.0, -1.0], device=n_cam.device)
    n01 = (1.0 + n_cam) * 0.5                            # [0,1] camera frame

    tgt = dsine_target.detach()
    # valid: depth>0 finite, bỏ biên (pad=0) — nơi normal không định nghĩa.
    valid = torch.isfinite(depth) & (depth > 0)
    valid[:1, :] = False; valid[-1:, :] = False
    valid[:, :1] = False; valid[:, -1:] = False
    valid = valid & torch.isfinite(tgt[..., 0])
    if valid.sum() < 100:
        return None

    diff = torch.abs(n01 - tgt)[valid]                  # (Nvalid,3) L1
    return lam * diff.mean()
