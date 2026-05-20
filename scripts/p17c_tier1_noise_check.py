#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 17c — Tier 1] ∇depth noise per-scene pre-check
# File: scripts/p17c_tier1_noise_check.py  (TẠO MỚI — keep local)
#
# Diagnostic-only (no train, no production touch). Đo noise_score per scene
# từ A3 final ckpt (iter 10000), correlate với C1a-uniform Δ per-scene (parse
# từ logs/p17_c1/ + logs/p13_lfcf/), output verdict theo pre-registered Q1
# table. KHÔNG đụng `arguments`/`train.py`/`c1_normal.py` — chỉ import
# `_normal_from_depth_cam` (= chính operator C1a thực sự xử lý → metric
# "chính operator" theo Q2 lock).
#
# ──────────────────────────── PRE-REGISTRATION LOCKED ────────────────────────
# Metric:  angle(n_raw, n_smooth_σ2px)
#   - n_raw    = _normal_from_depth_cam(rendered_depth)
#   - n_smooth = _normal_from_depth_cam(GaussianBlur(rendered_depth, σ=2px))
#   - blur     = torch separable Gaussian, σ=2.0, kernel_size=13, reflect-pad
#   - aggregate= mean angle (deg) over valid pixels per train-view, then mean
#                across train-views per scene → 1 scalar noise_score per scene
#
# Q1 BIN → ACTION (Pearson r vs Δ_C1a per-scene; magnitude triage, NOT stat):
#   r ≤ −0.7       → very_strong → SKIP Tier 2 → pilot N=24 @ T=densify_until_iter
#   (−0.7, −0.5]   → not_refuted → Tier 2 REQUIRED (compute=LOOSE locked-NOW)
#   (−0.5, −0.3)   → ambiguous   → Tier 2 REQUIRED
#   r ≥ −0.3       → refuted     → SKIP all → push Gate-phẳng / C1b
#
# CAVEATS (locked, baked vào verdict — KHÔNG slip):
#   (a) N=8 thiếu power thống kê: SE_r dưới null ≈ ±0.35, ambiguous-zone
#       rộng 0.2 < SE → MAGNITUDE triage, KHÔNG significance. KHÔNG so sánh
#       within-bin (r=−0.55 KHÔNG "mạnh hơn" r=−0.45 đáng kể).
#   (b) Pseudo-replication: Δ_C1a per-scene là DATA đã motivate hypothesis.
#       Tier 1 = consistency check, KHÔNG causation. Confounder không tách
#       được: geometry-complexity → noise & loss; DSINE-reliability → loss.
#   FORBID downstream: "R² claim", "p<0.05", "noise = driver xác nhận".
#   Tier 1 pass ⟹ "hypothesis CHƯA bị bác", KHÔNG "noise được xác nhận".
#
# ROBUSTNESS (Q4 lock):
#   (1) Per-scene pixel-level box(min, q25, med, q75, max) → distinguish
#       noisy-đều vs noisy-do-outlier-pixels (mechanistic implication khác).
#   (2) Leave-one-out r₁..r₈: nếu drop scene s khiến LOO_r rơi bin khác main
#       → flag UNROBUST. KHÔNG auto-escalate (per reviewer spec) — user quyết.
#   (3) Smoke synthetic (~1s CPU):
#       plane tilted no-noise           → expect noise <  1° (gate <1°)
#       plane tilted + Gaussian noise   → expect noise > 15° (gate >15°)
#       FAIL smoke ⟹ REFUSE to interpret real numbers (code bug, not finding).
# ─────────────────────────────────────────────────────────────────────────────
# ============================================================
"""[CRSGaussian Phase 17c — Tier 1] ∇depth noise pre-check.

Server (no-train, ~15 min, 1 GPU; matplotlib optional):
    cd CRSGaussian
    python -c "import ast; ast.parse(open('scripts/p17c_tier1_noise_check.py').read()); print('syntax OK')"
    bash scripts/p17c_tier1_run.sh
  → logs/p17c/tier1_noise_check.txt + tier1_scatter.png + tier1_boxplot.png
"""

import os
import re
import sys
import math
import statistics
import numpy as np

sys.path.insert(0, ".")

DATA_ROOT   = os.environ.get("DATA_ROOT", "data/nerf_llff_data")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "output/p13_lfcf")
A3_LOG      = os.environ.get("A3_LOG", "logs/p13_lfcf")
C1_LOG      = os.environ.get("C1_LOG", "logs/p17_c1")
ITERATION   = int(os.environ.get("ITERATION", "10000"))
SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
SEEDS = os.environ.get("SEEDS", "42 137 9999").split()   # for Δ_C1a aggregation
SEED_REF = os.environ.get("SEED_REF", "42")              # A3 ckpt seed for noise
OUT_DIR = os.environ.get("OUT_DIR", "logs/p17c")

PSNR_PAT = re.compile(r"Best test PSNR:\s*([\d\.eE\-\+]+)")

# Pre-registered triage thresholds — LOCKED (Q1 table)
THR_VERY_STRONG = -0.7
THR_NOT_REFUTED = -0.5
THR_AMBIGUOUS_LO = -0.3   # r ≥ this → refuted


# ── Gaussian blur pinned: torch separable, σ=2, ksize=13, reflect ──
def _gauss_kernel_1d(sigma=2.0, ksize=13):
    import torch
    x = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).cuda()


def gaussian_blur_2d(depth_HW, sigma=2.0, ksize=13):
    """Separable Gaussian on (H,W) torch tensor; reflect-pad both axes."""
    import torch
    import torch.nn.functional as F
    k = _gauss_kernel_1d(sigma, ksize)
    pad = ksize // 2
    d = depth_HW[None, None]                              # (1,1,H,W)
    d = F.pad(d, (pad, pad, 0, 0), mode='reflect')
    d = F.conv2d(d, k.view(1, 1, 1, -1))                  # blur along W
    d = F.pad(d, (0, 0, pad, pad), mode='reflect')
    d = F.conv2d(d, k.view(1, 1, -1, 1))                  # blur along H
    return d[0, 0]


def noise_score_per_pixel(depth_HW, fx, fy, cx, cy):
    """angle(n_raw, n_smooth_σ2) per pixel, in degrees. Uses _exact_ the
    finite-diff cross-product operator that C1a back-props through."""
    import torch
    from utils.loss.c1_normal import _normal_from_depth_cam     # = chính op
    n_raw    = _normal_from_depth_cam(depth_HW, fx, fy, cx, cy)        # (H,W,3)
    d_blur   = gaussian_blur_2d(depth_HW)
    n_smooth = _normal_from_depth_cam(d_blur, fx, fy, cx, cy)          # (H,W,3)
    dot = (n_raw * n_smooth).sum(dim=-1).clamp(-1.0, 1.0)
    ang = torch.acos(dot) * (180.0 / math.pi)                          # (H,W)
    valid = torch.isfinite(depth_HW) & (depth_HW > 0)
    valid[:1, :] = False; valid[-1:, :] = False
    valid[:, :1] = False; valid[:, -1:] = False
    valid = valid & torch.isfinite(ang)
    return ang, valid


def smoke_check():
    """Synthetic gates: plane <1° AND noisy plane >15° required to PASS."""
    import torch
    H, W = 256, 256
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32),
                            indexing='ij')
    plane = (5.0 + 0.01 * xx + 0.005 * yy).cuda()
    a1, v1 = noise_score_per_pixel(plane, 300.0, 300.0, W / 2.0, H / 2.0)
    s1 = float(a1[v1].mean().item())
    torch.manual_seed(0)
    noisy = plane + 0.05 * torch.randn_like(plane)
    a2, v2 = noise_score_per_pixel(noisy, 300.0, 300.0, W / 2.0, H / 2.0)
    s2 = float(a2[v2].mean().item())
    return s1, s2, (s1 < 1.0 and s2 > 15.0)


def _parse_psnr(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="ignore") as f:
        m = PSNR_PAT.search(f.read())
    return float(m.group(1)) if m else None


def per_scene_delta(scene):
    """Mean(C1L10 − A3) across SEEDS where both logs exist (parse PSNR)."""
    ds = []
    for sd in SEEDS:
        a = _parse_psnr(f"{A3_LOG}/A3_seed{sd}_{scene}.log")
        c = _parse_psnr(f"{C1_LOG}/C1L10_seed{sd}_{scene}.log")
        if a is not None and c is not None:
            ds.append(c - a)
    return statistics.fmean(ds) if ds else None


def per_scene_noise(scene):
    """Load A3 ckpt iter 10000, render train-view depth, compute noise_score.
    Mirror p17_c1_preprocess_dsine.py:93-108 args-merge pattern (verified).
    Returns (mean_angle_deg, box_stats_5tuple) or (None, None) if missing."""
    import torch
    from argparse import ArgumentParser, Namespace
    from scene import Scene, GaussianModel
    from arguments import ModelParams, PipelineParams
    from gaussian_renderer import render

    model_path = f"{OUTPUT_ROOT}/A3_seed{SEED_REF}_{scene}"
    cfg_path = os.path.join(model_path, "cfg_args")
    ply = f"{model_path}/point_cloud/iteration_{ITERATION}/point_cloud.ply"
    if not os.path.isfile(cfg_path) or not os.path.isfile(ply):
        print(f"  SKIP {scene}: missing cfg_args/ckpt at {model_path}")
        return None, None

    parser = ArgumentParser()
    lp = ModelParams(parser); pp = PipelineParams(parser); _ = lp
    args_cmdline = parser.parse_args([])
    with open(cfg_path) as f:
        cfg = eval(f.read())
    merged = vars(args_cmdline).copy()
    for k, v in vars(cfg).items():
        merged[k] = v
    args = Namespace(**merged)
    args.model_path = model_path
    if not os.path.isdir(getattr(args, "source_path", "") or ""):
        args.source_path = os.path.join(DATA_ROOT, scene)
    pipe = pp.extract(args)

    bg = torch.tensor([0., 0., 0.], device="cuda")
    angles_all = []
    with torch.no_grad():
        gaussians = GaussianModel(args)
        scene_obj = Scene(args, gaussians, load_iteration=ITERATION,
                          shuffle=False)
        for c in scene_obj.getTrainCameras():
            rpkg = render(c, gaussians, pipe, bg)
            depth = (rpkg["depth"] / (rpkg["alpha"] + 1e-6)).squeeze(0)
            H, W = depth.shape
            fx = W / (2.0 * math.tan(c.FoVx * 0.5))
            fy = H / (2.0 * math.tan(c.FoVy * 0.5))
            ang, valid = noise_score_per_pixel(depth, fx, fy, W / 2.0, H / 2.0)
            angles_all.append(ang[valid].detach().cpu().numpy())
    del gaussians, scene_obj
    torch.cuda.empty_cache()

    if not angles_all:
        return None, None
    a = np.concatenate(angles_all)
    mean_ang = float(a.mean())
    box = (float(a.min()),
           float(np.quantile(a, 0.25)),
           float(np.median(a)),
           float(np.quantile(a, 0.75)),
           float(a.max()))
    return mean_ang, box


def pearson_r(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = statistics.fmean(xs); my = statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def r_to_bin(r):
    """Map Pearson r → pre-registered bin (Q1 table)."""
    if r is None:
        return "NO_DATA"
    if r <= THR_VERY_STRONG:
        return "very_strong"
    if r <= THR_NOT_REFUTED:
        return "not_refuted"
    if r < THR_AMBIGUOUS_LO:
        return "ambiguous"
    return "refuted"


BIN_ACTION = {
    "very_strong": "SKIP Tier 2 → pilot N=24 tại T = densify_until_iter",
    "not_refuted": "Tier 2 REQUIRED (compute=LOOSE locked) → trajectory → pilot",
    "ambiguous":   "Tier 2 REQUIRED → trajectory → pilot",
    "refuted":     "SKIP cả Tier 2 + pilot → push Gate-phẳng / C1b",
    "NO_DATA":     "(insufficient data)",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("Phase 17c — Tier 1 ∇depth noise pre-check (pre-registered LOCKED)")
    print("=" * 70)
    print(f"SCENES        = {SCENES}")
    print(f"SEED_REF (ckpt) = {SEED_REF}  ITERATION = {ITERATION}")
    print(f"SEEDS (for Δ)   = {SEEDS}")
    print(f"OUT_DIR       = {OUT_DIR}")
    print(f"Triage bins: r ≤ {THR_VERY_STRONG} | "
          f"({THR_VERY_STRONG}, {THR_NOT_REFUTED}] | "
          f"({THR_NOT_REFUTED}, {THR_AMBIGUOUS_LO}) | "
          f"≥ {THR_AMBIGUOUS_LO}")

    # ── (3) Smoke synthetic — gate hard ──
    print("\n[Smoke] synthetic gate: plane <1° AND noisy >15° required")
    s1, s2, ok = smoke_check()
    print(f"  plane (no noise)   = {s1:7.3f}°   (gate: <1°)")
    print(f"  plane + iid noise  = {s2:7.3f}°   (gate: >15°)")
    if not ok:
        print("  ❌ SMOKE FAIL — REFUSE to interpret real-scene numbers "
              "(suspect code bug, NOT finding). Exit.")
        sys.exit(1)
    print("  ✅ SMOKE PASS — proceed to real scenes.")

    # ── Per-scene noise + Δ_C1a + box stats ──
    print("\n[Per-scene] noise_score @ iter 10000 + Δ_C1a_uniform + box stats")
    print(f"  {'scene':<9}  {'noise°':>8}  {'Δ_C1a':>8}  "
          f"{'box[min,q25,med,q75,max]':<35}")
    rows = []
    for sc in SCENES:
        delta = per_scene_delta(sc)
        mean_ang, box = per_scene_noise(sc)
        rows.append((sc, mean_ang, delta, box))
        if mean_ang is None or delta is None:
            print(f"  {sc:<9}  {'---':>8}  {'---':>8}  (missing)")
        else:
            bstr = f"[{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f},{box[4]:.1f}]"
            print(f"  {sc:<9}  {mean_ang:8.3f}  {delta:+8.3f}  {bstr:<35}")

    valid = [(sc, n, d) for (sc, n, d, _) in rows
             if n is not None and d is not None]
    if len(valid) < 5:
        print("\n❌ Insufficient data (<5 scenes valid) — cannot compute corr.")
        sys.exit(2)

    xs = [n for (_, n, _) in valid]
    ys = [d for (_, _, d) in valid]
    r_main = pearson_r(xs, ys)
    main_bin = r_to_bin(r_main)
    print(f"\n[Pearson] r = {r_main:+.4f}  (N={len(valid)}, TRIAGE not stat)")

    # ── (2) Leave-one-out leverage ──
    print("[Leverage] drop-1 leave-one-out r:")
    loo = []
    for i in range(len(valid)):
        xs_i = xs[:i] + xs[i + 1:]
        ys_i = ys[:i] + ys[i + 1:]
        ri = pearson_r(xs_i, ys_i)
        loo.append((valid[i][0], ri))
        print(f"    drop {valid[i][0]:<9} → r = {ri:+.4f}  bin={r_to_bin(ri)}")
    loo_vals = [v for (_, v) in loo if v is not None]
    loo_range = (max(loo_vals) - min(loo_vals)) if loo_vals else float("nan")
    print(f"  LOO range = {loo_range:.4f}")
    unrobust = any(r_to_bin(v) != main_bin for v in loo_vals)

    # ── Verdict ──
    print("\n" + "─" * 70)
    print(f"[Verdict] main r = {r_main:+.4f} → bin = {main_bin}")
    print(f"          action: {BIN_ACTION[main_bin]}")
    if unrobust:
        flips = [(sc, r_to_bin(v)) for (sc, v) in loo
                 if v is not None and r_to_bin(v) != main_bin]
        print(f"  ⚠️ UNROBUST: LOO bin-flips on {flips}")
        print(f"     (per reviewer spec: flag noted, NO auto-escalate — user quyết)")
    else:
        print("  ✅ ROBUST: no LOO bin-flip")
    print("─" * 70)
    print("CAVEATS (locked, không slip):")
    print("  (a) N=8 thiếu power → MAGNITUDE triage, KHÔNG significance test.")
    print("      KHÔNG so sánh within-bin (SE_r ≈ ±0.35 dưới null).")
    print("  (b) Pseudo-replication trên data đã motivate hypothesis → ")
    print("      consistency check, KHÔNG causation. Confound geometry-complexity")
    print("      / DSINE-reliability không tách được. Causal test = pilot N=24.")
    print("FORBID: R² claim, 'p<0.05', 'noise = driver xác nhận'.")
    print("Tier 1 pass ⟹ 'hypothesis CHƯA bị bác', KHÔNG 'xác nhận'.")
    print("─" * 70)

    # ── (1) Scatter PNG + box plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Scatter
        plt.figure(figsize=(6, 5))
        for sc, n, d in valid:
            plt.scatter(n, d, s=40)
            plt.text(n, d, f"  {sc}", fontsize=8, va='center')
        plt.axhline(0, color="gray", lw=0.5, ls="--")
        plt.xlabel("noise_score @ iter 10000 (deg)")
        plt.ylabel("Δ_C1a_uniform (PSNR, paired mean over seeds)")
        plt.title(f"Tier 1 — Pearson r = {r_main:+.3f}  N={len(valid)}\n"
                  f"(MAGNITUDE triage, NOT statistical significance)")
        plt.tight_layout()
        fp1 = os.path.join(OUT_DIR, "tier1_scatter.png")
        plt.savefig(fp1, dpi=120); plt.close()
        print(f"[Plot] scatter  → {fp1}")
        # Box plot per scene (reconstruct from 5-stats)
        stats = []
        for (sc, n, d, box) in rows:
            if box is None:
                continue
            stats.append({"med": box[2], "q1": box[1], "q3": box[3],
                          "whislo": box[0], "whishi": box[4],
                          "fliers": [], "label": sc})
        if stats:
            plt.figure(figsize=(9, 5))
            plt.bxp(stats, showfliers=False)
            plt.ylabel("per-pixel angle(n_raw, n_smooth_σ2) (deg)")
            plt.title("Tier 1 — per-scene noise distribution @ iter 10000")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            fp2 = os.path.join(OUT_DIR, "tier1_boxplot.png")
            plt.savefig(fp2, dpi=120); plt.close()
            print(f"[Plot] boxplot → {fp2}")
    except Exception as e:
        print(f"[Plot] SKIPPED (matplotlib err: {e})")

    print("\nDone.")


if __name__ == "__main__":
    main()
