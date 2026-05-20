#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 17c — Tier 2] ∇depth noise TRAJECTORY pre-check
# File: scripts/p17c_tier2_noise_trajectory.py  (TẠO MỚI — keep local)
#
# Đo noise_score(scene, iter) trên A3 intermediate ckpts (nếu có). Trả lời
# câu hỏi mà Tier 1 KHÔNG trả lời được: "noise có drop theo iter không?"
# (logic gap Tier 1 = chỉ đo terminal, ranking-stay-same nếu noise giảm đều).
#
# Workflow:
#   1) SCAN: liệt kê iter ckpts có sẵn mỗi scene
#   2) Nếu < 3 iter/scene cho ≥ 5/8 scenes → INSUFFICIENT, print re-train cmd, exit
#   3) Else: render train-view depth từng (scene, iter), compute noise_score
#   4) Per-scene drop_rel + classify STRONG/WEAK/NO
#   5) Cross-tab với Δ_C1a per-scene (parse logs/p17_c1/, logs/p13_lfcf/)
#   6) Plot trajectory + apply pre-registered rule → verdict
#
# Diagnostic-only. KHÔNG đụng production code; chỉ import
# `_normal_from_depth_cam` từ `utils.loss.c1_normal` (= chính operator C1a).
#
# ────────────────────────── PRE-REGISTRATION LOCKED ────────────────────────
# Metric: angle(n_raw, n_smooth_σ2px) — SAME as Tier 1 (consistency).
#   blur = torch separable Gaussian σ=2.0, ksize=13, reflect-pad.
#   aggregate = mean angle (deg) over valid pixels × train-views per scene.
#
# Drop rule per scene (LOCKED):
#   noise_early(s) = mean noise_score(s, iter ≤ EARLY_CUTOFF=2500)
#   noise_late(s)  = mean noise_score(s, iter ≥ LATE_CUTOFF =6000)
#   drop_rel(s)    = (early − late) / max(early, 1e-6)
#   class(s):  STRONG if drop_rel ≥ 0.30
#              WEAK   if 0.10 ≤ drop_rel < 0.30
#              NO     if drop_rel < 0.10 (or insufficient trajectory data)
#
# Aggregate verdict (LOCKED — single-config decision):
#   ≥ 5/8 STRONG ∧ median(elbow iter) ∈ [3000, 7000]
#       → substrate EXISTS → pilot N=24 at T = nearest of {5000, 7000} to median elbow
#   ≥ 5/8 ANY-drop ∧ < 5/8 STRONG
#       → substrate WEAK (ambiguous) → user quyết (NOT auto-pilot)
#   Else → substrate ABSENT → SKIP C1a-late → push Gate-phẳng / C1b
#
# Elbow per scene: iter where consecutive-iter drop (noise[i]−noise[i+1])
#   is largest. Median across STRONG-drop scenes only.
#
# Caveats (locked, mirror Tier 1):
#   (a) N=8 thiếu power → MAGNITUDE triage, NOT significance test.
#   (b) drop_rel measures within-scene trajectory, KHÔNG isolate noise-vs-
#       confound (geometry-complexity/DSINE-reliability could co-vary).
#   Causal test = pilot N=24 behavioral.
#   FORBID downstream: "p<0.05", "% variance explained", "noise = driver".
# ───────────────────────────────────────────────────────────────────────────
# ============================================================
"""[CRSGaussian Phase 17c — Tier 2] noise trajectory.

Server (no-train, ~10 min IF intermediate ckpts có sẵn):
    bash scripts/p17c_tier2_run.sh
  → logs/p17c/tier2_trajectory.txt + tier2_curves.png + tier2_drop_bar.png

Nếu insufficient: script in re-train command + exit. Re-train ~4 hr trên
2 GPU (mirror p13/p15 A3 baseline + --save_iterations 500 2000 5000 7000 10000).
"""

import os
import re
import sys
import math
import glob
import statistics
import numpy as np

sys.path.insert(0, ".")

DATA_ROOT   = os.environ.get("DATA_ROOT",   "data/nerf_llff_data")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "output/p13_lfcf")
A3_LOG      = os.environ.get("A3_LOG",      "logs/p13_lfcf")
C1_LOG      = os.environ.get("C1_LOG",      "logs/p17_c1")
SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
SEEDS    = os.environ.get("SEEDS",    "42 137 9999").split()
SEED_REF = os.environ.get("SEED_REF", "42")
OUT_DIR  = os.environ.get("OUT_DIR",  "logs/p17c")

EARLY_CUTOFF = int(os.environ.get("EARLY_CUTOFF", "2500"))
LATE_CUTOFF  = int(os.environ.get("LATE_CUTOFF",  "6000"))

PSNR_PAT = re.compile(r"Best test PSNR:\s*([\d\.eE\-\+]+)")
ITER_DIR_PAT = re.compile(r"iteration_(\d+)")

CLASS_STRONG_THR = 0.30
CLASS_WEAK_THR   = 0.10
ELBOW_LO         = 3000
ELBOW_HI         = 7000


# ── Metric (mirror Tier 1 noise_score for consistency) ──
def _gauss_kernel_1d(sigma=2.0, ksize=13):
    import torch
    x = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).cuda()


def gaussian_blur_2d(depth_HW, sigma=2.0, ksize=13):
    import torch
    import torch.nn.functional as F
    k = _gauss_kernel_1d(sigma, ksize)
    pad = ksize // 2
    d = depth_HW[None, None]
    d = F.pad(d, (pad, pad, 0, 0), mode='reflect')
    d = F.conv2d(d, k.view(1, 1, 1, -1))
    d = F.pad(d, (0, 0, pad, pad), mode='reflect')
    d = F.conv2d(d, k.view(1, 1, -1, 1))
    return d[0, 0]


def noise_score_per_pixel(depth_HW, fx, fy, cx, cy):
    import torch
    from utils.loss.c1_normal import _normal_from_depth_cam
    n_raw    = _normal_from_depth_cam(depth_HW, fx, fy, cx, cy)
    d_blur   = gaussian_blur_2d(depth_HW)
    n_smooth = _normal_from_depth_cam(d_blur, fx, fy, cx, cy)
    dot = (n_raw * n_smooth).sum(dim=-1).clamp(-1.0, 1.0)
    ang = torch.acos(dot) * (180.0 / math.pi)
    valid = torch.isfinite(depth_HW) & (depth_HW > 0)
    valid[:1, :] = False; valid[-1:, :] = False
    valid[:, :1] = False; valid[:, -1:] = False
    valid = valid & torch.isfinite(ang)
    return ang, valid


def _parse_psnr(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="ignore") as f:
        m = PSNR_PAT.search(f.read())
    return float(m.group(1)) if m else None


def per_scene_delta_c1a(scene):
    ds = []
    for sd in SEEDS:
        a = _parse_psnr(f"{A3_LOG}/A3_seed{sd}_{scene}.log")
        c = _parse_psnr(f"{C1_LOG}/C1L10_seed{sd}_{scene}.log")
        if a is not None and c is not None:
            ds.append(c - a)
    return statistics.fmean(ds) if ds else None


# ── Scan available intermediate ckpts ──
def scan_iters(scene):
    """Return sorted list of ITER ints available for this scene."""
    model_path = f"{OUTPUT_ROOT}/A3_seed{SEED_REF}_{scene}"
    pc_dir = os.path.join(model_path, "point_cloud")
    if not os.path.isdir(pc_dir):
        return []
    iters = []
    for d in os.listdir(pc_dir):
        m = ITER_DIR_PAT.match(d)
        if m:
            ply = os.path.join(pc_dir, d, "point_cloud.ply")
            if os.path.isfile(ply):
                iters.append(int(m.group(1)))
    return sorted(iters)


# ── Per-(scene, iter) noise measurement ──
def measure_one_ckpt(scene, iteration):
    """Return mean noise_score (deg) or None if missing/error."""
    import torch
    from argparse import ArgumentParser, Namespace
    from scene import Scene, GaussianModel
    from arguments import ModelParams, PipelineParams
    from gaussian_renderer import render

    model_path = f"{OUTPUT_ROOT}/A3_seed{SEED_REF}_{scene}"
    cfg_path = os.path.join(model_path, "cfg_args")
    ply = f"{model_path}/point_cloud/iteration_{iteration}/point_cloud.ply"
    if not os.path.isfile(cfg_path) or not os.path.isfile(ply):
        return None

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
    angs = []
    with torch.no_grad():
        gaussians = GaussianModel(args)
        scene_obj = Scene(args, gaussians, load_iteration=iteration,
                          shuffle=False)
        for c in scene_obj.getTrainCameras():
            rpkg = render(c, gaussians, pipe, bg)
            depth = (rpkg["depth"] / (rpkg["alpha"] + 1e-6)).squeeze(0)
            H, W = depth.shape
            fx = W / (2.0 * math.tan(c.FoVx * 0.5))
            fy = H / (2.0 * math.tan(c.FoVy * 0.5))
            ang, valid = noise_score_per_pixel(depth, fx, fy, W / 2.0, H / 2.0)
            angs.append(ang[valid].detach().cpu().numpy())
    del gaussians, scene_obj
    torch.cuda.empty_cache()
    if not angs:
        return None
    return float(np.concatenate(angs).mean())


def classify(drop_rel):
    if drop_rel is None or not math.isfinite(drop_rel):
        return "NO"
    if drop_rel >= CLASS_STRONG_THR:
        return "STRONG"
    if drop_rel >= CLASS_WEAK_THR:
        return "WEAK"
    return "NO"


def find_elbow(iters, noises):
    """Largest consecutive drop. Return iter of LATER point in that pair,
    or None if <2 points or no drop."""
    if len(iters) < 2:
        return None
    drops = [(iters[i + 1], noises[i] - noises[i + 1])
             for i in range(len(iters) - 1)]
    drops_pos = [(it, d) for (it, d) in drops if d > 0]
    if not drops_pos:
        return None
    return max(drops_pos, key=lambda x: x[1])[0]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 72)
    print("Phase 17c — Tier 2 ∇depth noise TRAJECTORY (pre-registered LOCKED)")
    print("=" * 72)
    print(f"SCENES        = {SCENES}")
    print(f"SEED_REF (ckpt) = {SEED_REF}")
    print(f"EARLY ≤ {EARLY_CUTOFF}  | LATE ≥ {LATE_CUTOFF}")
    print(f"Class thresholds: STRONG ≥ {CLASS_STRONG_THR}, "
          f"WEAK ≥ {CLASS_WEAK_THR}")
    print(f"Substrate-exists ⟺ ≥5/8 STRONG ∧ median elbow ∈ "
          f"[{ELBOW_LO},{ELBOW_HI}]")
    print()

    # ── Step 1: SCAN ──
    print("[Scan] Available intermediate ckpts per scene:")
    avail = {}
    for sc in SCENES:
        its = scan_iters(sc)
        avail[sc] = its
        print(f"  {sc:<9} iters: {its}")

    sufficient = sum(1 for sc in SCENES if len(avail[sc]) >= 3)
    print(f"\nScenes with ≥3 iter ckpts: {sufficient}/{len(SCENES)}")
    if sufficient < 5:
        print("\n❌ INSUFFICIENT trajectory data.")
        print("   Cần re-train A3 (seed 42) với intermediate saves.")
        print("   Command đề xuất (mirror p13/p15 A3 baseline + extra saves):")
        print("     # 1 train run/scene, ~30 min/scene; 2-GPU split tổng ~2h:")
        print("     bash scripts/p17c_tier2_a3_resave.sh")
        print("   (hoặc tôi viết script đó nếu bạn approve)")
        sys.exit(2)
    print("✅ Sufficient — proceed to measurement.")

    # ── Step 2: Measure per (scene, iter) ──
    print("\n[Measure] noise_score(scene, iter) (~5–10 min total)")
    data = {sc: [] for sc in SCENES}   # data[sc] = [(iter, noise), ...]
    for sc in SCENES:
        for it in avail[sc]:
            n = measure_one_ckpt(sc, it)
            if n is not None:
                data[sc].append((it, n))
                print(f"  {sc:<9} iter={it:<5} noise={n:7.3f}°")
            else:
                print(f"  {sc:<9} iter={it:<5} (load fail)")

    # ── Step 3: Per-scene drop + classify ──
    print("\n[Per-scene trajectory + classification]")
    print(f"  {'scene':<9} {'iters':<22} {'early':>7} {'late':>7} "
          f"{'drop_rel':>9} {'class':<8} {'elbow':>6} {'Δ_C1a':>8}")
    rows = []
    for sc in SCENES:
        traj = data[sc]
        if len(traj) < 2:
            print(f"  {sc:<9} (only {len(traj)} ckpts — skip)")
            rows.append((sc, traj, None, None, None, "NO", None, None))
            continue
        traj.sort()
        iters_s = [t[0] for t in traj]
        noises_s = [t[1] for t in traj]
        early_pts = [n for (it, n) in traj if it <= EARLY_CUTOFF]
        late_pts  = [n for (it, n) in traj if it >= LATE_CUTOFF]
        if not early_pts or not late_pts:
            ne = statistics.fmean(early_pts) if early_pts else None
            nl = statistics.fmean(late_pts) if late_pts else None
            drop = None; cls = "NO"
        else:
            ne = statistics.fmean(early_pts)
            nl = statistics.fmean(late_pts)
            drop = (ne - nl) / max(ne, 1e-6)
            cls = classify(drop)
        elbow = find_elbow(iters_s, noises_s)
        delta = per_scene_delta_c1a(sc)
        rows.append((sc, traj, ne, nl, drop, cls, elbow, delta))
        its_str = ",".join(str(x) for x in iters_s)
        ne_s = f"{ne:7.2f}" if ne is not None else "  ----"
        nl_s = f"{nl:7.2f}" if nl is not None else "  ----"
        dr_s = f"{drop:+9.3f}" if drop is not None else "    ----"
        el_s = f"{elbow}" if elbow is not None else "  --"
        d_s  = f"{delta:+8.3f}" if delta is not None else "    ----"
        print(f"  {sc:<9} {its_str:<22} {ne_s} {nl_s} {dr_s} {cls:<8} "
              f"{el_s:>6} {d_s}")

    # ── Step 4: Aggregate verdict ──
    counts = {"STRONG": 0, "WEAK": 0, "NO": 0}
    elbows_strong = []
    for (sc, _, _, _, _, cls, elb, _) in rows:
        counts[cls] += 1
        if cls == "STRONG" and elb is not None:
            elbows_strong.append(elb)
    n_strong = counts["STRONG"]
    n_any = counts["STRONG"] + counts["WEAK"]
    med_elbow = statistics.median(elbows_strong) if elbows_strong else None

    print("\n[Aggregate]")
    print(f"  STRONG = {n_strong}/8 | WEAK = {counts['WEAK']}/8 | "
          f"NO = {counts['NO']}/8 | any-drop = {n_any}/8")
    print(f"  median elbow across STRONG scenes: "
          f"{med_elbow if med_elbow is not None else '—'}")

    elbow_in_window = (med_elbow is not None
                       and ELBOW_LO <= med_elbow <= ELBOW_HI)
    if n_strong >= 5 and elbow_in_window:
        # Snap to nearest of {5000, 7000}
        T_proposed = 5000 if abs(med_elbow - 5000) <= abs(med_elbow - 7000) else 7000
        verdict = "SUBSTRATE_EXISTS"
        action = (f"pilot N=24 at T = {T_proposed} "
                  f"(nearest of {{5000,7000}} to median elbow {med_elbow})")
    elif n_any >= 5 and n_strong < 5:
        verdict = "SUBSTRATE_WEAK"
        action = ("ambiguous — user quyết (pilot at T=5000 with caveat / "
                  "skip / further diagnostic)")
    else:
        verdict = "SUBSTRATE_ABSENT"
        action = "SKIP C1a-late → push Gate-phẳng / C1b"

    # ── Step 5: Cross-tab noise drop × C1a Δ ──
    print("\n[Cross-tab] drop_rel × Δ_C1a per-scene (mechanistic sub-check)")
    print("  (Hypothesis sub-check: do planar winners drop noise more?")
    print("   If yes → defer helps winners (where help already exists); ")
    print("   foliage losers stay noisy → defer doesn't reach them.)")
    pairs = [(d, dl) for (_, _, _, _, d, _, _, dl) in rows
             if d is not None and dl is not None]
    if len(pairs) >= 5:
        xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
        mx = statistics.fmean(xs); my = statistics.fmean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        dy = math.sqrt(sum((y - my) ** 2 for y in ys))
        sub_r = num / (dx * dy) if (dx > 0 and dy > 0) else None
        if sub_r is not None:
            print(f"  Pearson r(drop_rel, Δ_C1a) = {sub_r:+.4f}  (N={len(pairs)})")
            print("  (informational — KHÔNG part of locked verdict)")

    print("\n" + "─" * 72)
    print(f"[Verdict] {verdict}")
    print(f"  Action: {action}")
    print("─" * 72)
    print("CAVEATS (locked, không slip):")
    print("  (a) N=8 → MAGNITUDE triage, NOT statistical significance.")
    print("  (b) drop_rel measures within-scene trajectory but does NOT")
    print("      isolate noise-vs-confound. Causal test = pilot N=24.")
    print("  FORBID: 'p<0.05', '% variance', 'noise = driver confirmed'.")
    print("─" * 72)

    # ── Step 6: Plots ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Curves per scene
        plt.figure(figsize=(8, 5))
        for (sc, traj, _, _, _, cls, _, _) in rows:
            if len(traj) < 2: continue
            xs = [t[0] for t in traj]; ys = [t[1] for t in traj]
            ls = "-" if cls == "STRONG" else ("--" if cls == "WEAK" else ":")
            plt.plot(xs, ys, ls, marker="o", label=f"{sc} ({cls})")
        plt.axvspan(ELBOW_LO, ELBOW_HI, color="lightgray", alpha=0.3,
                    label=f"elbow window [{ELBOW_LO},{ELBOW_HI}]")
        plt.xlabel("iteration")
        plt.ylabel("noise_score (deg)")
        plt.title(f"Tier 2 — noise trajectory per scene\nverdict={verdict}")
        plt.legend(fontsize=8, loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        fp1 = os.path.join(OUT_DIR, "tier2_curves.png")
        plt.savefig(fp1, dpi=120); plt.close()
        print(f"[Plot] curves  → {fp1}")
        # Bar of drop_rel
        plt.figure(figsize=(8, 4))
        labels = []; vals = []; colors = []
        cmap = {"STRONG": "C2", "WEAK": "C1", "NO": "C3"}
        for (sc, _, _, _, drop, cls, _, _) in rows:
            if drop is None: continue
            labels.append(sc); vals.append(drop * 100); colors.append(cmap[cls])
        plt.bar(labels, vals, color=colors)
        plt.axhline(CLASS_STRONG_THR * 100, color="gray", lw=0.5, ls="--")
        plt.axhline(CLASS_WEAK_THR * 100, color="gray", lw=0.5, ls="--")
        plt.ylabel("drop_rel (%, early→late)")
        plt.title("Tier 2 — per-scene noise drop")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fp2 = os.path.join(OUT_DIR, "tier2_drop_bar.png")
        plt.savefig(fp2, dpi=120); plt.close()
        print(f"[Plot] drop bar → {fp2}")
    except Exception as e:
        print(f"[Plot] SKIPPED (matplotlib err: {e})")

    print("\nDone.")


if __name__ == "__main__":
    main()
