#!/usr/bin/env python
"""
[B6 thesis figure] Generate Pareto plot Quality (PSNR) vs Speed (FPS).

Mục đích: Figure visualization trade-off cho thesis section 4.6 "Practical Integration".
3 method × 8 scene LLFF 3-view → 1 figure compact.

Output: outputs/figures/fig_pareto_quality_speed.png (300 DPI, ready paste LaTeX)

Cách dùng (chạy trong nerfstudio env):
    cd /home/aidev/workspace/representation-3d/duyen/nerfstudio
    conda activate nerfstudio
    python crsgaussian_plugin/scripts/gen_pareto_figure.py

Dependencies: matplotlib, torch, json (đều có sẵn trong nerfstudio env).
"""

import json
import torch
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    # ── Config ──
    SCENES = ["fern", "horns", "fortress", "flower", "leaves", "orchids", "room", "trex"]

    # (display_name, output_subdir, method_name, eval_prefix, color, marker, base_marker_size)
    CONFIGS = [
        ("ReDI-GS (ours)",       "phase22_b4",              "crsgaussian",              "eval_phase22_b4_",        "darkred",   "*", 400),
        ("Splatfacto",           "splatfacto_3view",        "splatfacto-sparse",        "eval_splatfacto3_",       "steelblue", "o", 200),
        ("Pure-3DGS (no AbsGS)", "splatfacto_3view_noabs",  "splatfacto-sparse-noabs",  "eval_splatfacto3_noabs_", "gray",      "^", 200),
    ]

    EVAL_DIR = Path("outputs/eval_results")
    BASE_OUTPUTS = Path("outputs")

    # ── Gather data ──
    fig, ax = plt.subplots(figsize=(9, 6.5))

    summary = []
    for name, out_dir, mname, prefix, color, marker, base_size in CONFIGS:
        psnrs, fpss, n_gausss = [], [], []
        for scene in SCENES:
            # PSNR + FPS from official eval JSON
            json_path = EVAL_DIR / f"{prefix}{scene}.json"
            if json_path.exists():
                try:
                    d = json.load(open(json_path))["results"]
                    psnrs.append(d["psnr"])
                    fpss.append(d.get("fps", 0))
                except Exception as e:
                    print(f"  ⚠️ Failed parse {json_path}: {e}")

            # N_gauss from ckpt
            ckpt_candidates = list((BASE_OUTPUTS / out_dir / scene / mname).glob("*/nerfstudio_models/step-*.ckpt"))
            if ckpt_candidates:
                try:
                    c = torch.load(str(ckpt_candidates[-1]), map_location="cpu", weights_only=False)
                    if mname == "crsgaussian":
                        n = c["pipeline"]["_model._extra_state"][1].shape[0]
                    else:
                        n = c["pipeline"]["_model.gauss_params.means"].shape[0]
                    n_gausss.append(n)
                except Exception as e:
                    print(f"  ⚠️ Failed load ckpt {ckpt_candidates[-1]}: {e}")

        if not psnrs or not fpss:
            print(f"  ❌ No data for {name} — skip")
            continue

        avg_psnr = sum(psnrs) / len(psnrs)
        avg_fps = sum(fpss) / len(fpss)
        avg_n = sum(n_gausss) / len(n_gausss) if n_gausss else 50000

        print(f"  {name:<28} | PSNR {avg_psnr:5.2f} | FPS {avg_fps:5.1f} | N_gauss {avg_n:>8,.0f}")
        summary.append((name, avg_psnr, avg_fps, avg_n))

        # Marker size proportional to N_gauss
        marker_size = base_size * (avg_n / 50000) ** 0.5  # sqrt scale cho visual

        ax.scatter(
            avg_fps, avg_psnr,
            s=marker_size,
            c=color, marker=marker,
            edgecolors='black', linewidths=1.5,
            label=f"{name}  ({avg_n/1000:.0f}K Gaussians)",
            alpha=0.85,
            zorder=3,
        )
        # Annotation PSNR value next to marker
        ax.annotate(
            f"{avg_psnr:.2f} dB",
            (avg_fps, avg_psnr),
            xytext=(12, -4),
            textcoords='offset points',
            fontsize=10,
            color='black',
        )

    # ── Axis + title ──
    ax.set_xlabel("Render speed (FPS) — higher is faster", fontsize=12)
    ax.set_ylabel("Quality (PSNR ↑, dB)", fontsize=12)
    ax.set_title("Quality vs Speed Trade-off\n(average over 8 LLFF scenes, 3-view sparse, 1/8 resolution)",
                 fontsize=12, pad=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9)

    # ── Annotation cho ReDI-GS Pareto-optimal corner ──
    if summary:
        # Find ReDI-GS coords
        for name, psnr, fps, n in summary:
            if "ReDI" in name:
                ax.annotate(
                    'Best quality\n(slower render)',
                    xy=(fps, psnr),
                    xytext=(fps + 8, psnr + 0.7),
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
                    fontsize=11, ha='center',
                    color='darkred', fontweight='bold',
                )
                break

    # ── Save ──
    plt.tight_layout()
    Path("outputs/figures").mkdir(exist_ok=True)
    out_path = "outputs/figures/fig_pareto_quality_speed.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {out_path}")
    print(f"   Use trong LaTeX: \\includegraphics[width=0.85\\textwidth]{{fig_pareto_quality_speed.png}}")


if __name__ == "__main__":
    main()
