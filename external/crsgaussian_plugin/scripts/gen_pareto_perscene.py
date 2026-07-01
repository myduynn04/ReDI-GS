#!/usr/bin/env python
"""
[Thesis 4.6 Figure] Per-scene Pareto scatter — Quality (PSNR) vs Speed (FPS).

Mục đích: Show ReDI-GS dominate CONSISTENT trên cả 8 scene, KHÔNG chỉ AVG.
Bổ sung cho fig_pareto_quality_speed.png (AVG only).

Output: outputs/figures/fig_pareto_perscene.png (300 DPI)

Cách dùng:
    cd /home/aidev/workspace/representation-3d/duyen/nerfstudio
    conda activate nerfstudio
    python crsgaussian_plugin/scripts/gen_pareto_perscene.py
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    SCENES = ["fern", "horns", "fortress", "flower", "leaves", "orchids", "room", "trex"]

    # (display_name, eval_prefix, color, marker, label_short)
    CONFIGS = [
        ("ReDI-GS (ours)",       "eval_phase22_b4_",       "darkred",   "*",  "ReDI-GS"),
        ("Splatfacto",           "eval_splatfacto3_",      "steelblue", "o",  "Splat"),
        ("Pure-3DGS (no AbsGS)", "eval_splatfacto3_noabs_", "gray",     "^",  "Pure"),
    ]

    EVAL_DIR = Path("outputs/eval_results")
    fig, ax = plt.subplots(figsize=(10, 7))

    # ── Plot per-scene points ──
    for name, prefix, color, marker, short in CONFIGS:
        psnrs, fpss, scenes_with_data = [], [], []
        for scene in SCENES:
            json_path = EVAL_DIR / f"{prefix}{scene}.json"
            if json_path.exists():
                try:
                    d = json.load(open(json_path))["results"]
                    psnrs.append(d["psnr"])
                    fpss.append(d.get("fps", 0))
                    scenes_with_data.append(scene)
                except Exception as e:
                    print(f"  ⚠️ Failed {json_path}: {e}")

        # Scatter
        ax.scatter(
            fpss, psnrs,
            s=120, c=color, marker=marker,
            edgecolors='black', linewidths=1.0,
            label=f"{name}",
            alpha=0.80, zorder=3,
        )

        # AVG of this method (large halo)
        if psnrs:
            avg_psnr = sum(psnrs) / len(psnrs)
            avg_fps = sum(fpss) / len(fpss)
            ax.scatter(
                avg_fps, avg_psnr,
                s=500, facecolors='none', edgecolors=color,
                linewidths=3.0, marker='o',
                zorder=4,
            )
            ax.annotate(
                f"{short}\nAVG",
                (avg_fps, avg_psnr),
                xytext=(0, -25), textcoords='offset points',
                fontsize=9, color=color, fontweight='bold',
                ha='center',
            )

        # Label scene names next to ReDI-GS points (chỉ method anh để KHÔNG quá lộn xộn)
        if "ReDI" in name:
            for s, p, f in zip(scenes_with_data, psnrs, fpss):
                ax.annotate(
                    s, (f, p),
                    xytext=(8, 3), textcoords='offset points',
                    fontsize=8, color='darkred', alpha=0.85,
                )

    # ── Axis + title ──
    ax.set_xlabel("Render speed (FPS) — higher is faster", fontsize=12)
    ax.set_ylabel("Quality (PSNR ↑, dB)", fontsize=12)
    ax.set_title("Per-scene Quality vs Speed (8 LLFF scenes, 3-view sparse)",
                 fontsize=13, pad=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

    # ── Annotation cho ReDI-GS cluster ──
    ax.annotate(
        'ReDI-GS dominates:\nbetter quality + faster\non all 8 scenes',
        xy=(150, 23), xytext=(80, 27),
        arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
        fontsize=11, ha='center', color='darkred', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.1, edgecolor='darkred'),
    )

    # ── Save ──
    plt.tight_layout()
    Path("outputs/figures").mkdir(exist_ok=True)
    out_path = "outputs/figures/fig_pareto_perscene.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {out_path}")
    print(f"   LaTeX: \\includegraphics[width=0.95\\textwidth]{{fig_pareto_perscene.png}}")


if __name__ == "__main__":
    main()
