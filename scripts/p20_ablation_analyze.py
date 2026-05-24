#!/usr/bin/env python3
# ============================================================
# [CRSGaussian Phase 20] Ablation analyzer — bảng kiểu Binocular Table 4
# File: scripts/p20_ablation_analyze.py  (TẠO MỚI — keep local)
#
# Gom log mọi config (p20_ablation_run.sh) + reuse FULL/dense → in bảng:
#   config | PSNR | SSIM | LPIPS | N_gauss | Δ-vs-FULL (paired)
# Δ-vs-FULL của config LOO = ÂM → |Δ| = đóng góp BIÊN của block bị tắt.
#
# Reuse (KHÔNG train lại): full = logs/p13_lfcf (A3-MVS 21.330),
#                          full+denseinit = logs/p18_pilot (A3-dense).
#
# Cost = N_gaussians (đếm, tin được). KHÔNG dùng wall-clock — đã chứng
# minh cross-session noise 3× (decisions_log [2026-05-22]).
#
# Chạy (CPU, instant): python scripts/p20_ablation_analyze.py
# ============================================================
"""[CRSGaussian Phase 20] ablation analyzer — Table-4-style."""

import os
import re
import statistics

SCENES = os.environ.get(
    "SCENES", "fern flower fortress horns leaves orchids room trex").split()
SEEDS = os.environ.get("SEEDS", "42 137 9999").split()
ABL_ROOT = os.environ.get("ABL_ROOT", "logs/p20_ablation")

# (label, log_dir) — thứ tự in bảng. full / dense = reuse.
CONFIGS = [
    ("base",           f"{ABL_ROOT}/base"),
    ("no_depthcrs",    f"{ABL_ROOT}/no_depthcrs"),
    ("no_efa",         f"{ABL_ROOT}/no_efa"),
    ("no_drop",        f"{ABL_ROOT}/no_drop"),
    ("no_opacity",     f"{ABL_ROOT}/no_opacity"),
    ("no_dcycle",      f"{ABL_ROOT}/no_dcycle"),
    ("no_shcrs",       f"{ABL_ROOT}/no_shcrs"),
    ("no_crsprune",    f"{ABL_ROOT}/no_crsprune"),
    ("no_rvis",        f"{ABL_ROOT}/no_rvis"),
    ("no_crsprune_rvis", f"{ABL_ROOT}/no_crsprune_rvis"),
    ("no_prune_rvis",  f"{ABL_ROOT}/no_prune_rvis"),
    # ── A3-TRIM LOO trên dense (Phase 20 dense ablation) ──
    ("trim_full",         f"{ABL_ROOT}/trim_full"),
    ("trim_no_efa",       f"{ABL_ROOT}/trim_no_efa"),
    ("trim_no_drop",      f"{ABL_ROOT}/trim_no_drop"),
    ("trim_no_opacity",   f"{ABL_ROOT}/trim_no_opacity"),
    ("trim_no_dcycle",    f"{ABL_ROOT}/trim_no_dcycle"),
    ("trim_no_shcrs",     f"{ABL_ROOT}/trim_no_shcrs"),
    ("trim_no_depthcrs",  f"{ABL_ROOT}/trim_no_depthcrs"),
    ("FULL (A3-MVS)",  os.environ.get("FULL_LOG", "logs/p13_lfcf")),
    ("FULL+denseinit", os.environ.get("DENSE_LOG", "logs/p18_pilot")),
]
REF = os.environ.get("REF_LABEL", "FULL (A3-MVS)")   # mốc Δ; dense ablation dùng REF_LABEL=trim_full

PSNR_PAT = re.compile(r"Best test PSNR:\s*([\d.eE+\-]+)")
NG_PAT = re.compile(r"Final #Gaussians:\s*(\d+)")
# Summary table test row: "<iter> | test | <psnr> | <ssim> | <lpips> | ..."
ROW_PAT = re.compile(
    r"\|\s*test\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|")


def parse_log(path):
    """Trả (psnr, ssim, lpips, n_gauss) — None nếu thiếu."""
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    mp = PSNR_PAT.search(txt)
    psnr = float(mp.group(1)) if mp else None
    mn = NG_PAT.search(txt)
    ng = int(mn.group(1)) if mn else None
    rows = ROW_PAT.findall(txt)            # mọi test row trong summary
    ssim = lpips = None
    if rows:
        ssim, lpips = float(rows[-1][1]), float(rows[-1][2])  # row cuối
    return psnr, ssim, lpips, ng


def collect(log_dir):
    """Trả {(scene,seed): (psnr,ssim,lpips,ng)} cho mọi run tìm thấy."""
    out = {}
    for sc in SCENES:
        for sd in SEEDS:
            r = parse_log(f"{log_dir}/A3_seed{sd}_{sc}.log")
            if r and r[0] is not None:
                out[(sc, sd)] = r
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def main():
    print("=" * 88)
    print("Phase 20 — Ablation study (LOO trên A3-FULL), kiểu Binocular Table 4")
    print(f"scenes={len(SCENES)}  seeds={SEEDS}  (mỗi config tối đa "
          f"{len(SCENES)*len(SEEDS)} run)")
    print("=" * 88)

    data = {label: collect(d) for label, d in CONFIGS}
    ref = data.get(REF, {})

    hdr = (f"{'config':<16} {'N_run':>6} {'PSNR':>8} {'SSIM':>7} "
           f"{'LPIPS':>7} {'N_gauss':>10} {'Δ-vs-FULL':>10}")
    print(hdr)
    print("-" * len(hdr))
    for label, _ in CONFIGS:
        runs = data.get(label, {})
        if not runs:
            print(f"{label:<16} {'0':>6}   (chưa có log)")
            continue
        psnr = _mean([v[0] for v in runs.values()])
        ssim = _mean([v[1] for v in runs.values()])
        lpips = _mean([v[2] for v in runs.values()])
        ng = _mean([v[3] for v in runs.values()])
        # paired Δ vs FULL — chỉ trên (scene,seed) có ở CẢ hai
        common = [k for k in runs if k in ref]
        dvf = (statistics.fmean([runs[k][0] - ref[k][0] for k in common])
               if common and label != REF else None)
        print(f"{label:<16} {len(runs):>6} {psnr:>8.4f} "
              f"{(f'{ssim:.4f}' if ssim is not None else '-'):>7} "
              f"{(f'{lpips:.4f}' if lpips is not None else '-'):>7} "
              f"{(f'{ng:.0f}' if ng is not None else '-'):>10} "
              f"{(f'{dvf:+.4f}' if dvf is not None else '—'):>10}")

    print()
    print("ĐỌC:")
    print("  Δ-vs-FULL (paired) của config `no_X` = ÂM → |Δ| = đóng góp BIÊN")
    print("    của block X (tắt X thì A3 tụt bấy nhiêu).")
    print("  `base` Δ rất âm = tổng hợp lực mọi module. `no_depthcrs` =")
    print("    tắt cả depth+CRS (cascade) → đóng góp cả khối CRS.")
    print("  ⚠️ LOO có synergy (Phase 9) → các |Δ| KHÔNG cộng đúng = base→FULL.")
    print("  ⚠️ Cost = N_gauss (đếm). Wall-clock bỏ — cross-session noise 3×.")


if __name__ == "__main__":
    main()
