#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 13.2.5] GDAGS Gate-2 pre-flight.
# File: scripts/p13_2_gdags_gate.py  (TẠO MỚI — standalone)
# Mục đích: TRƯỚC khi implement GDAGS, kiểm xem coherence-weight của
#           GDAGS có DEGENERATE trong regime LLFF 3-view + A3 backbone
#           không. Quyết bằng số, không đoán.
#
# GDAGS (verified GDAGS/scene/gaussian_model.py:526-527):
#   consistency = (grads + 1e-8) / (grads_abs + 1e-8)        ∈ (0,1]
#   weight      = 0.8 + 25 * (1 - consistency)^15
#   clone dùng grads/weight, split dùng grads*weight
# Hằng số (0.8, 25, pow15) tune cho Mip-NeRF360 dense-view. 3-view +
# A3-LFCF backbone phân bố grads/grads_abs khác → weight có thể:
#   - collapse ≈0.8 đều  → GDAGS ≈ no-op → KHÔNG implement
#   - spread hợp lý       → có traction → implement
#   - explode (p95 huge)  → hằng số cần re-tune, KHÔNG copy mù
#
# ZERO production touch: script standalone, ĐỌC checkpoint A3,
# render+1 backward (no optimizer step, no training, no .ply modify,
# KHÔNG sửa train.py/arguments/scene). Contract user 2026-05-17.
#
# ── CAVEAT honest (đọc kèm verdict) ──
# consistency = grads/grads_abs là đại lượng TRAINING-TIME (accumulator
# reset mỗi densify interval, KHÔNG lưu trong .ply). Script này đo PROXY:
# converged-model, L1-only, eval-render, accumulate qua 3 TRAIN cam.
# → Đủ để phát hiện weight degenerate/explode trong regime này;
#   KHÔNG phải tái tạo CHÍNH XÁC grad densify-time (iter 500-5000,
#   full Phase8 loss). Verdict mang tính directional.
#
# Reuse verified: render_scene_test arg-merge/Scene/GaussianModel/render
# pattern (đã proven p13_2_bottleneck_decompose.py trên server).
# ============================================================
"""[CRSGaussian Phase 13.2.5] GDAGS coherence-weight degeneracy gate.

Run (GPU SERVER):
    python scripts/p13_2_gdags_gate.py
    SCENES="orchids horns" python scripts/p13_2_gdags_gate.py
"""

import os
import sys
import math
import statistics

import numpy as np

sys.path.insert(0, ".")

DATA_ROOT = os.environ.get("DATA_ROOT", "data/nerf_llff_data")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "output/p13_lfcf")
SEED = os.environ.get("SEED", "42")
ITERATION = int(os.environ.get("ITERATION", "10000"))
SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()

# GDAGS formula constants (verified GDAGS/scene/gaussian_model.py:527)
GDAGS_W0 = 0.8
GDAGS_K = 25.0
GDAGS_POW = 15


def _stem(name):
    return os.path.basename(name).split(".")[0]


def _gdags_weight(consistency):
    """EXACT GDAGS:527 — weight = 0.8 + 25*(1-c)^15. c clamped to [0,1]."""
    c = np.clip(consistency, 0.0, 1.0)
    return GDAGS_W0 + GDAGS_K * np.power(1.0 - c, GDAGS_POW)


def compute_scene_consistency(scene_name):
    """Load A3 ckpt, render+backward each TRAIN cam (L1, no optim),
    accumulate ‖grad[:2]‖ + ‖grad[2:]‖ MIRROR add_densification_stats
    (gaussian_model.py:1065-1069). Returns (consistency np[N], n_cams) or None.
    """
    import torch
    from argparse import ArgumentParser, Namespace
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from arguments import ModelParams, PipelineParams

    model_path = f"{OUTPUT_ROOT}/A3_seed{SEED}_{scene_name}"
    cfg_path = os.path.join(model_path, "cfg_args")
    ply = f"{model_path}/point_cloud/iteration_{ITERATION}/point_cloud.ply"
    if not os.path.isfile(cfg_path) or not os.path.isfile(ply):
        print(f"  ERR missing cfg_args / checkpoint at {model_path}")
        return None

    # Arg-merge mirror get_combined_args (verified pattern, p13_2 bottleneck)
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    _ = lp
    args_cmdline = parser.parse_args([])
    with open(cfg_path) as f:
        cfg = eval(f.read())
    merged = vars(args_cmdline).copy()
    for k, v in vars(cfg).items():
        merged[k] = v
    args = Namespace(**merged)
    args.model_path = model_path
    args.iteration = ITERATION
    if not os.path.isdir(getattr(args, "source_path", "") or ""):
        args.source_path = os.path.join(DATA_ROOT, scene_name)
    pipe = pp.extract(args)

    torch.cuda.empty_cache()
    # Load model (no_grad ok for load); render+backward needs grad enabled.
    with torch.no_grad():
        gaussians = GaussianModel(args)
        scene = Scene(args, gaussians, load_iteration=ITERATION, shuffle=False)
        # [VERIFIED gaussian_model:153,322 + renderer:120] load_ply KHÔNG set
        # confidence (ở lại empty(0)) → CoR-GS confidence-rasterizer backward
        # crash (3 vs 0). create_from_pcd:322 set ones_like(opacities). Replicate:
        # neutral confidence=1 → ratio grads/grads_abs BẤT BIẾN → Gate-2 OK.
        gaussians.confidence = torch.ones_like(gaussians.get_opacity)
    bg_color = [1., 1., 1.] if getattr(args, "white_background", False) \
        else [0., 0., 0.]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    N = gaussians.get_xyz.shape[0]
    accum = torch.zeros((N, 1), device="cuda")       # Σ ‖grad[:2]‖ (standard)
    accum_abs = torch.zeros((N, 1), device="cuda")   # Σ ‖grad[2:]‖ (abs-grad)
    denom = torch.zeros((N, 1), device="cuda")

    train_cams = scene.getTrainCameras()
    for c in train_cams:
        pkg = render(c, gaussians, pipe, bg)
        vp = pkg["viewspace_points"]
        vp.retain_grad()                              # leaf may be non-leaf
        gt = c.original_image[:3].clamp(0, 1).cuda()
        loss = torch.abs(pkg["render"] - gt).mean()   # L1-only PROXY (caveat)
        loss.backward()
        vis = pkg["visibility_filter"]
        g = vp.grad
        if g is None:
            print(f"  WARN viewspace grad None ({scene_name}) — skip cam")
            continue
        # MIRROR add_densification_stats:1065-1067 exactly
        accum[vis] += torch.norm(g[vis, :2], dim=-1, keepdim=True)
        accum_abs[vis] += torch.norm(g[vis, 2:], dim=-1, keepdim=True)
        denom[vis] += 1
        gaussians.optimizer = None  # defensive: ensure no optim state path

    denom_safe = denom.clamp(min=1.0)
    grads = (accum / denom_safe)
    grads_abs = (accum_abs / denom_safe)
    # GDAGS:526 — consistency = (grads+1e-8)/(grads_abs+1e-8)
    consistency = ((grads + 1e-8) / (grads_abs + 1e-8)).squeeze(-1)
    seen = (denom.squeeze(-1) > 0)                    # only Gaussians ever visible
    cons = consistency[seen].detach().cpu().numpy()

    del gaussians, scene
    torch.cuda.empty_cache()
    return cons, len(train_cams)


def main():
    print("=== Phase 13.2.5 GDAGS Gate-2 (coherence-weight degeneracy) ===")
    print(f"OUTPUT_ROOT={OUTPUT_ROOT} SEED={SEED} ITER={ITERATION}")
    print(f"GDAGS weight = {GDAGS_W0} + {GDAGS_K}·(1−c)^{GDAGS_POW}  "
          f"(verified GDAGS:527)")
    print("PROXY: converged-model, L1-only, eval-render, 3 train cam "
          "(caveat — not exact densify-time grad)\n")

    rows = []
    for sc in SCENES:
        print(f"──── {sc} ────")
        r = compute_scene_consistency(sc)
        if r is None:
            print("  SKIP\n")
            continue
        cons, ncam = r
        w = _gdags_weight(cons)
        # distribution
        cp = np.percentile(cons, [5, 25, 50, 75, 95])
        wp = np.percentile(w, [5, 25, 50, 75, 95])
        collapsed = float(np.mean(w <= GDAGS_W0 * 1.01))   # ≈0.8 (no-op)
        heavy = float(np.mean(w > 10.0))                   # weight huge
        rows.append(dict(scene=sc, n=len(cons), ncam=ncam,
                         cp=cp, wp=wp, collapsed=collapsed, heavy=heavy))
        print(f"  N={len(cons)} ({ncam} train cam)")
        print(f"  consistency p[5,25,50,75,95] = "
              f"[{cp[0]:.3f} {cp[1]:.3f} {cp[2]:.3f} {cp[3]:.3f} {cp[4]:.3f}]")
        print(f"  weight      p[5,25,50,75,95] = "
              f"[{wp[0]:.2f} {wp[1]:.2f} {wp[2]:.2f} {wp[3]:.2f} {wp[4]:.2f}]")
        print(f"  collapsed(w≈0.8)={collapsed*100:.1f}%  "
              f"heavy(w>10)={heavy*100:.1f}%\n")

    if not rows:
        print("NO scenes analyzed."); sys.exit(1)

    # ── Aggregate + verdict ──
    print("=== Aggregate ===")
    agg_collapsed = statistics.fmean(r['collapsed'] for r in rows)
    agg_heavy = statistics.fmean(r['heavy'] for r in rows)
    agg_w50 = statistics.fmean(r['wp'][2] for r in rows)
    agg_w95 = statistics.fmean(r['wp'][4] for r in rows)
    agg_c50 = statistics.fmean(r['cp'][2] for r in rows)
    print(f"  consistency p50 mean = {agg_c50:.3f}")
    print(f"  weight p50 mean = {agg_w50:.2f}   p95 mean = {agg_w95:.2f}")
    print(f"  collapsed(w≈0.8) mean = {agg_collapsed*100:.1f}%")
    print(f"  heavy(w>10) mean = {agg_heavy*100:.1f}%")

    print("\n=== VERDICT ===")
    if agg_collapsed >= 0.85:
        print("  ❌ COLLAPSED — weight ≈0.8 cho ≥85% Gaussian.")
        print("  → GDAGS ≈ scale threshold bằng hằng 0.8 = gần no-op trong")
        print("    3-view regime. KHÔNG đáng implement (cùng class LFCF-alone")
        print("    +0.015). Stop GDAGS direction.")
    elif agg_w95 > 100.0 or agg_heavy >= 0.40:
        print("  ⚠️ EXPLODE — weight p95 quá lớn / heavy-fraction cao.")
        print("  → Split threshold ×weight quá khắc → gần như không split.")
        print("  → Hằng số GDAGS (0.8,25,pow15) KHÔNG hợp 3-view regime.")
        print("    Cần re-tune có cơ sở (document), KHÔNG copy mù. Risk cao.")
    else:
        print("  ✅ TRACTION — weight có spread hợp lý (không collapse, không")
        print("     explode). GDAGS coherence-policy CÓ tác dụng phân biệt")
        print("     trong regime ta → đáng implement (flag-gated, LFCF-")
        print("     preserving) + pilot A/B vs A3.")

    print("\n=== Caveat (đọc kèm) ===")
    print("  - PROXY: converged-model L1-only eval-render, KHÔNG phải grad")
    print("    densify-time (iter 500-5000, full Phase8 loss). Directional.")
    print("  - consistency ratio character ổn định qua train (2 norm dương)")
    print("    → đủ phát hiện collapse/explode; KHÔNG kết luận magnitude gain.")
    print("  - GDAGS không orthogonal (GCR=grads/grads_abs = signal AbsGS")
    print("    đã có) → đây là policy A/B trên trục proven, KHÔNG +feature.")


if __name__ == "__main__":
    main()
