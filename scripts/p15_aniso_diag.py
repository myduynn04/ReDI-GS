#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 15] Anisotropy substrate diagnostic (no-train).
# File: scripts/p15_aniso_diag.py  (TẠO MỚI)
# Mục đích: TRƯỚC khi implement L_aniso + tốn run, trả lời 3 câu
#   (như D3 đã đóng density-dropout, dense-init Gate):
#   Q1. Phân bố s_max/s_min của Gaussian A3 — pathological hay moderate?
#   Q2. High-aniso có TƯƠNG QUAN test-error không? (off-target → saturate)
#   Q3. High-aniso vs CRS + per-scene thin-label (context).
#   Q4 [DECISIVE, per-Gaussian — sửa gap granularity scene-level]:
#       high-aniso do s_min-sụp (surface-flat HỢP LỆ, kể cả intra-scene
#       cạnh tủ/khung cửa trong non-thin scene) hay s_max-phình
#       (overfit-stretch)? blunt L_aniso=s_max/s_min phạt CẢ HAI → nếu
#       flat-driven thống trị (kể cả non-thin) = lặp lỗi V1-structure
#       (không phân biệt legitimate vs overfit) → đóng.
#
# Verdict asymmetric (honest): moderate-dist HOẶC no-error-corr HOẶC
# thin-structure-dominated → đóng sạch (định lượng, không hand-wave).
# pathological + error-corr + low-CRS + KHÔNG thin-dominated → có đất
# (lúc đó mới bàn formulation/τ). KHÔNG kết luận "sẽ work".
#
# Verified-from-code: gaussian_model.get_scaling=exp(_scaling) (N,3,
#   world-unit, ratio bất biến activation), get_xyz, get_crs (property,
#   persisted ckpt :183/:201), get_opacity. Model-load + camera-proj
#   mirror VERIFIED scripts/p13_2_bottleneck_decompose.py:128-222,345-370.
# Standalone, no-train, KHÔNG đụng production (Gate discipline).
#
# PROXY caveat: Gaussian-center→test-pixel→error-lookup KHÔNG chứng minh
# nhân quả (occlusion confound, frontmost không xét). Coarse correlation:
# no-corr = đóng robust; có-corr = "đáng điều tra", KHÔNG = causal.
# ============================================================
"""[CRSGaussian Phase 15] Anisotropy substrate diagnostic (no-train)."""

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
# FULL-8 mặc định: reject-decision diagnostic (feedback_full_8scene).
SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
# Thin-structure scenes (foliage/branch — cần needle hợp lệ). Nếu các
# scene này thống trị high-aniso → blunt penalty hại chúng (risk #2).
THIN = set(os.environ.get("THIN_SCENES", "leaves fern trex flower").split())
RATIO_BINS = [3.0, 5.0, 10.0, 20.0, 50.0]
EPS = 1e-8


def _load(scene_name):
    """Load A3 ckpt; return (gaussians, test_cams, render, pipe, bg).
    Mirror bottleneck_decompose:128-183 (cfg_args merge verified)."""
    import torch
    from argparse import ArgumentParser, Namespace
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from arguments import ModelParams, PipelineParams

    mp = f"{OUTPUT_ROOT}/A3_seed{SEED}_{scene_name}"
    cfg = os.path.join(mp, "cfg_args")
    ply = f"{mp}/point_cloud/iteration_{ITERATION}/point_cloud.ply"
    if not (os.path.isfile(cfg) and os.path.isfile(ply)):
        print(f"  ERR missing ckpt {mp}")
        return None
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    _ = lp
    base = parser.parse_args([])
    with open(cfg) as f:
        cfgns = eval(f.read())
    merged = vars(base).copy()
    for k, v in vars(cfgns).items():
        merged[k] = v
    args = Namespace(**merged)
    args.model_path = mp
    args.iteration = ITERATION
    if not os.path.isdir(getattr(args, "source_path", "") or ""):
        args.source_path = os.path.join(DATA_ROOT, scene_name)
    pipe = pp.extract(args)
    g = GaussianModel(args)
    sc = Scene(args, g, load_iteration=ITERATION, shuffle=False)
    bg = __import__("torch").tensor([0., 0., 0.], dtype=__import__("torch").float32,
                                    device="cuda")
    return g, sc, render, pipe, bg


def _spearman(x, y):
    """Rank-correlation (robust, không cần linearity). x,y 1D np."""
    if len(x) < 10:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = (rx - rx.mean()) / (rx.std() + EPS)
    ry = (ry - ry.mean()) / (ry.std() + EPS)
    return float((rx * ry).mean())


def diag_scene(scene_name):
    import torch
    L = _load(scene_name)
    if L is None:
        return None
    g, scene, render, pipe, bg = L
    with torch.no_grad():
        scaling = g.get_scaling.detach().cpu().numpy()          # (N,3)
        xyz = g.get_xyz.detach().cpu().numpy()                  # (N,3)
        try:
            crs = g.get_crs.detach().cpu().numpy().reshape(-1)  # (N,)
        except Exception:
            crs = None
        N = scaling.shape[0]
        s_max = scaling.max(1)
        s_min = scaling.min(1)
        ratio = s_max / (s_min + EPS)
        log_ratio = np.log(ratio + EPS)

        # ── Q2: per-Gaussian local test-error qua center-projection ──
        ones = np.ones((N, 1), np.float64)
        xyz_h = np.concatenate([xyz.astype(np.float64), ones], 1)   # (N,4)
        err_sum = np.zeros(N)
        err_cnt = np.zeros(N)
        for c in scene.getTestCameras():
            pkg = render(c, g, pipe, bg)
            rd = pkg["render"].clamp(0, 1).detach().cpu().numpy()
            gt = c.original_image[:3].clamp(0, 1).detach().cpu().numpy()
            errmap = np.mean(np.abs(rd - gt), 0)                 # (H,W)
            H, W = errmap.shape
            wvt = c.world_view_transform.detach().cpu().numpy()  # W2C^T
            fx = W / (2.0 * math.tan(c.FoVx * 0.5))
            fy = H / (2.0 * math.tan(c.FoVy * 0.5))
            cx, cy = W / 2.0, H / 2.0
            pv = xyz_h @ wvt                                     # (N,4)
            z = pv[:, 2]
            valid = z > 1e-6
            u = np.full(N, -1.0)
            v = np.full(N, -1.0)
            u[valid] = fx * pv[valid, 0] / z[valid] + cx
            v[valid] = fy * pv[valid, 1] / z[valid] + cy
            inb = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            ui = np.clip(u[inb].astype(int), 0, W - 1)
            vi = np.clip(v[inb].astype(int), 0, H - 1)
            err_sum[inb] += errmap[vi, ui]
            err_cnt[inb] += 1.0
        seen = err_cnt > 0
        local_err = np.full(N, np.nan)
        local_err[seen] = err_sum[seen] / err_cnt[seen]

    del g, scene
    torch.cuda.empty_cache()

    # ── Q2 stats: corr(log_ratio, local_err) + decile error trend ──
    m = seen & np.isfinite(local_err)
    corr_err = _spearman(log_ratio[m], local_err[m]) if m.sum() > 50 else float("nan")
    # decile mean-error: error có tăng theo anisotropy không?
    dec = []
    if m.sum() > 100:
        q = np.quantile(ratio[m], np.linspace(0, 1, 11))
        for i in range(10):
            sel = m & (ratio >= q[i]) & (ratio <= q[i + 1])
            dec.append(float(np.mean(local_err[sel])) if sel.sum() else float("nan"))
    corr_crs = (_spearman(log_ratio, crs) if crs is not None and N > 50
                else float("nan"))

    # ── Q4 (per-Gaussian, self-calibrated): high-aniso do s_min-sụp
    # (flat hợp lệ) hay s_max-phình (overfit-stretch)? — granularity
    # đúng (per-Gaussian, KHÔNG scene-level), percentile-rank trong
    # scene → KHÔNG hằng số tuyệt đối bịa. Decisive hơn Q2 (no occ-confound).
    smin_rank = np.argsort(np.argsort(s_min)) / max(N - 1, 1)   # 0..1
    smax_rank = np.argsort(np.argsort(s_max)) / max(N - 1, 1)
    # high-aniso = top-decile ratio
    hi = ratio >= np.quantile(ratio, 0.90)
    nhi = int(hi.sum())
    q4 = dict(n_hi=nhi)
    if nhi >= 30:
        smin_lo = smin_rank < 0.20            # s_min rất nhỏ trong scene
        smax_hi = smax_rank > 0.80            # s_max rất lớn trong scene
        flat = hi & smin_lo & (~smax_hi)      # min sụp, max thường → flat hợp lệ
        strc = hi & smax_hi & (~smin_lo)      # max phình, min thường → stretch
        extr = hi & smin_lo & smax_hi         # cả hai cực đoan
        modr = hi & (~smin_lo) & (~smax_hi)   # ratio lớn từ mid-range
        q4.update(
            flat=float(flat.sum() / nhi),
            stretch=float(strc.sum() / nhi),
            extreme=float(extr.sum() / nhi),
            moderate=float(modr.sum() / nhi),
        )

    return dict(
        scene=scene_name, N=N,
        ratio_med=float(np.median(ratio)),
        ratio_p90=float(np.percentile(ratio, 90)),
        ratio_p99=float(np.percentile(ratio, 99)),
        ratio_max=float(ratio.max()),
        frac=[float((ratio > t).mean()) for t in RATIO_BINS],
        corr_err=corr_err, dec_err=dec, corr_crs=corr_crs,
        thin=(scene_name in THIN), q4=q4,
    )


def main():
    print("=== Phase 15 Anisotropy substrate diagnostic (no-train) ===")
    print(f"OUTPUT_ROOT={OUTPUT_ROOT} SEED={SEED} ITER={ITERATION}")
    print(f"SCENES={SCENES}  THIN={sorted(THIN)}\n")
    print("Q1 s_max/s_min dist | Q2 corr(aniso,test-err) | Q3 corr(aniso,CRS)+thin\n")

    rows = []
    for s in SCENES:
        print(f"──── {s} ────")
        r = diag_scene(s)
        if r is None:
            print(f"  SKIP\n"); continue
        rows.append(r)
        fr = "  ".join(f">{t:g}:{r['frac'][i]*100:.1f}%"
                       for i, t in enumerate(RATIO_BINS))
        print(f"  Q1 N={r['N']} ratio med={r['ratio_med']:.2f} "
              f"p90={r['ratio_p90']:.2f} p99={r['ratio_p99']:.2f} "
              f"max={r['ratio_max']:.1f}")
        print(f"     frac {fr}")
        print(f"  Q2 corr(loganiso,err)={r['corr_err']:+.3f}  "
              f"decile-err={[f'{x:.4f}' for x in r['dec_err']]}")
        print(f"  Q3 corr(loganiso,CRS)={r['corr_crs']:+.3f}  "
              f"thin-structure={r['thin']}")
        q4 = r['q4']
        if 'flat' in q4:
            print(f"  Q4 high-aniso(n={q4['n_hi']}): flat={q4['flat']*100:.0f}%"
                  f" stretch={q4['stretch']*100:.0f}%"
                  f" extreme={q4['extreme']*100:.0f}%"
                  f" moderate={q4['moderate']*100:.0f}%\n")
        else:
            print(f"  Q4: n_hi={q4['n_hi']} <30, skip\n")

    if not rows:
        print("NO scenes."); sys.exit(1)

    print("=== AGGREGATE ===")
    amed = statistics.fmean(r['ratio_med'] for r in rows)
    ap99 = statistics.fmean(r['ratio_p99'] for r in rows)
    af10 = statistics.fmean(r['frac'][2] for r in rows)   # >10
    af50 = statistics.fmean(r['frac'][4] for r in rows)   # >50
    ce = [r['corr_err'] for r in rows if not math.isnan(r['corr_err'])]
    ace = statistics.fmean(ce) if ce else float("nan")
    cc = [r['corr_crs'] for r in rows if not math.isnan(r['corr_crs'])]
    acc = statistics.fmean(cc) if cc else float("nan")
    print(f"  Q1: ratio med={amed:.2f}  p99={ap99:.2f}  "
          f">10={af10*100:.1f}%  >50={af50*100:.1f}%")
    print(f"  Q2: mean corr(loganiso, test-err) = {ace:+.3f}")
    print(f"  Q3: mean corr(loganiso, CRS)      = {acc:+.3f}")
    # ── Q4 aggregate (per-Gaussian, granularity ĐÚNG) ──
    q4r = [r['q4'] for r in rows if 'flat' in r['q4']]
    q4_nonthin = [r['q4'] for r in rows
                  if (not r['thin']) and 'flat' in r['q4']]
    aflat = statistics.fmean(q['flat'] for q in q4r) if q4r else float("nan")
    astrc = statistics.fmean(q['stretch'] for q in q4r) if q4r else float("nan")
    # DECISIVE: flat-driven NGAY CẢ trong non-thin scene (user critique:
    # scene-level che intra-scene legitimate-thin như khung cửa/cạnh tủ).
    aflat_nt = (statistics.fmean(q['flat'] for q in q4_nonthin)
                if q4_nonthin else float("nan"))
    print(f"  Q4: high-aniso flat-driven={aflat*100:.0f}%  "
          f"stretch-driven={astrc*100:.0f}%  "
          f"(flat trong NON-THIN scene={aflat_nt*100:.0f}%)")
    tt = [r['ratio_p90'] for r in rows if r['thin']]
    nt = [r['ratio_p90'] for r in rows if not r['thin']]
    if tt and nt:
        print(f"  Q3(context): p90 ratio thin={statistics.fmean(tt):.2f} "
              f"vs non-thin={statistics.fmean(nt):.2f}")

    print("\n=== VERDICT (asymmetric, honest — Q4 per-Gaussian quyết định) ===")
    pathological = (ap99 >= 10.0) and (af10 >= 0.05)
    err_corr = (not math.isnan(ace)) and ace >= 0.10
    # flat_dom = high-aniso chủ yếu do s_min-sụp (surface-flat HỢP LỆ),
    # tính CẢ trên non-thin scene → blunt L_aniso=s_max/s_min hại nền-tảng
    # biểu diễn surface ở MỌI scene (user critique granularity, decisive).
    flat_dom = ((not math.isnan(aflat)) and aflat >= 0.50) or \
               ((not math.isnan(aflat_nt)) and aflat_nt >= 0.50)
    stretch_dom = (not math.isnan(astrc)) and astrc >= 0.40
    crs_high = (not math.isnan(acc)) and acc >= 0.10   # high-aniso ↔ high-CRS

    if not pathological:
        print(f"  ⚪ Q1 NOT pathological (p99={ap99:.1f}<10, >10={af10*100:.0f}%<5%)")
        print("     → đa số Gaussian moderate-aniso → L_aniso off-target →")
        print("       predicted SATURATE (như L_consist trên geometry-solved).")
        print("     → ĐÓNG anisotropy, định lượng (không hand-wave).")
    elif not err_corr:
        print(f"  ⚪ Q1 pathological NHƯNG Q2 corr(aniso,err)={ace:+.2f}≈0")
        print("     → high-aniso KHÔNG drive test-error → off-target → ĐÓNG.")
    elif flat_dom:
        print(f"  ❌ pathological + err-corr NHƯNG Q4 flat-driven thống trị")
        print(f"     (flat={aflat*100:.0f}%, flat-trong-NON-THIN={aflat_nt*100:.0f}%)")
        print("     → high-aniso chủ yếu do s_min-sụp = surface-flat HỢP LỆ")
        print("       (per-Gaussian, CẢ non-thin scene: cạnh tủ/khung cửa).")
        print("       blunt L_aniso=s_max/s_min phạt nền-tảng biểu diễn")
        print("       surface ĐÚNG → predicted hại = lặp lỗi V1-structure.")
        print("     → ĐÓNG blunt-ratio. (Chỉ phần stretch-driven mới là")
        print("       target — formulation KHÁC: phạt s_max-excess-vs-scene,")
        print("       KHÔNG ratio; đó là cơ chế riêng, cần analysis riêng,")
        print("       KHÔNG auto-pursue. CRS×structure-fix ≈ Phase-3β reject.)")
    elif stretch_dom:
        tag = "high-CRS (elongated-but-geom-OK = đúng overfit-shape giả thuyết)" \
              if crs_high else "low-CRS (≈floater, có thể đã handled CRS-prune/freeze)"
        print(f"  🎯 pathological + err-corr + Q4 stretch-driven thống trị "
              f"(stretch={astrc*100:.0f}%, flat={aflat*100:.0f}% thấp).")
        print(f"     high-aniso ↔ {tag}. → CÓ ĐẤT (overfit-stretch thật).")
        print("     Bước tiếp: gated pilot — formulation NÊN là phạt")
        print("     s_max-excess-vs-scene-normal (KHÔNG s_max/s_min thuần,")
        print("     tránh đụng flat hợp lệ); τ + start>T_warmup; full-8")
        print("     single-seed; per-scene catastrophe-guard; reuse A3.")
        if not crs_high:
            print("     ⚠️ low-CRS → verify chưa bị CRS-prune/SH-freeze nuốt"
                  " (tránh redundant như Phase-3β).")
    else:
        print(f"  🟡 pathological + err-corr nhưng Q4 mixed "
              f"(flat={aflat*100:.0f}% stretch={astrc*100:.0f}% — không bên")
        print("     nào thống trị) → INCONCLUSIVE, lean-close: blunt-ratio")
        print("     vẫn đụng phần flat hợp lệ đáng kể. KHÔNG đủ cơ sở pilot")
        print("     blunt; chỉ s_max-excess-formulation đáng cân (riêng).")

    print("\n=== Caveats (đọc kèm — KHÔNG over-claim) ===")
    print("  - Q2 = PROXY: center→pixel→err KHÔNG causal (occlusion confound,")
    print("    frontmost không xét). no-corr = đóng robust; corr = đáng điều")
    print("    tra, KHÔNG = nhân quả.")
    print("  - Q4 = check QUYẾT ĐỊNH (per-Gaussian, granularity đúng — sửa")
    print("    gap scene-level che intra-scene legitimate-thin). 'flat' =")
    print("    s_min-sụp → suy diễn = surface-flat hợp lệ (mạnh: s_min≈độ")
    print("    dày mặt; nhưng vẫn là suy diễn, không nhãn ground-truth).")
    print("  - Single-seed checkpoint; ratio/s_min/s_max dùng get_scaling.")
    print("  - Verdict = 'đáng pilot' / 'đóng' / 'inconclusive', KHÔNG 'sẽ")
    print("    work'. Implement L_aniso CHỈ khi 🎯 (và formulation =")
    print("    s_max-excess, KHÔNG ratio thuần); nhánh khác = đóng/treo.")


if __name__ == "__main__":
    main()
