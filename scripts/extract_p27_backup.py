"""
Extract ALL Phase 27 data from logs before deleting output/p27_pillar/.

Usage:
    cd ~/workspace/representation-3d/duyen/CoR-GS
    python scripts/extract_p27_backup.py

Output:
    docs/27_phase27_full_backup.md   (human-readable)
    docs/27_phase27_full_backup.json (machine-readable)

After running, safe to: rm -rf output/p27_pillar/
"""

import re
import json
from pathlib import Path
from collections import defaultdict

# ============================================================
# Config
# ============================================================
SCENES = ['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex']
LOG_DIR = Path("logs/p27_pillar")
OUTPUT_DIR = Path("docs")

# Regex để parse Best test PSNR/SSIM/LPIPS line từ training log
RX_METRICS = re.compile(
    r'10000\s*\|\s*test\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*[\d.]+\s*\|\s*(\d+)'
)
RX_BEST = re.compile(r'Best test PSNR:\s*([\d.]+)\s*at iter\s*(\d+)\s*\(N=(\d+)\)')
RX_TIME = re.compile(r'Total training:\s*([\d.]+)s')


def extract_log(log_path):
    """Extract metrics + meta from a single log file."""
    if not log_path.exists():
        return None

    text = log_path.read_text(errors='ignore')
    result = {}

    # Metrics @ iter 10000
    m = RX_METRICS.search(text)
    if m:
        result['psnr'] = float(m.group(1))
        result['ssim'] = float(m.group(2))
        result['lpips'] = float(m.group(3))
        result['ngauss'] = int(m.group(4))

    # Best test (final)
    m = RX_BEST.search(text)
    if m:
        result['best_psnr'] = float(m.group(1))
        result['best_iter'] = int(m.group(2))
        result['best_ngauss'] = int(m.group(3))

    # Training time
    m = RX_TIME.search(text)
    if m:
        result['training_time_s'] = float(m.group(1))

    return result if result else None


def discover_cells():
    """Find all cell folders in logs/p27_pillar/."""
    if not LOG_DIR.exists():
        print(f"⚠️ {LOG_DIR} not found!")
        return []

    # Cell folders (contain log files)
    cells = set()
    for log_file in LOG_DIR.rglob("A3_seed*_*.log"):
        # log_file = logs/p27_pillar/<cell>/A3_seed42_fern.log
        cell = log_file.parent.name
        cells.add(cell)
    return sorted(cells)


def discover_seeds(cell):
    """Find all seeds present for a cell."""
    cell_dir = LOG_DIR / cell
    if not cell_dir.exists():
        return []
    seeds = set()
    for log_file in cell_dir.glob("A3_seed*_*.log"):
        # filename = A3_seed{SEED}_{scene}.log
        match = re.match(r'A3_seed(\d+)_', log_file.name)
        if match:
            seeds.add(int(match.group(1)))
    return sorted(seeds)


def main():
    print("=" * 80)
    print("  Phase 27 Backup Extractor")
    print("=" * 80)

    # Discover all cells + seeds
    cells = discover_cells()
    print(f"\nFound {len(cells)} cells:")
    for c in cells:
        seeds = discover_seeds(c)
        print(f"  {c}: seeds={seeds}")

    if not cells:
        print("\n❌ No cells found in logs/p27_pillar/. Exit.")
        return

    # ============================================================
    # Extract all data
    # ============================================================
    print(f"\nExtracting metrics from {len(cells)} cells × {len(SCENES)} scenes × seeds...")
    all_data = {}  # {cell: {seed: {scene: {metrics}}}}

    for cell in cells:
        all_data[cell] = {}
        seeds = discover_seeds(cell)
        for seed in seeds:
            all_data[cell][seed] = {}
            for scene in SCENES:
                log_path = LOG_DIR / cell / f"A3_seed{seed}_{scene}.log"
                metrics = extract_log(log_path)
                if metrics:
                    all_data[cell][seed][scene] = metrics

    # ============================================================
    # Save JSON (machine-readable)
    # ============================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "27_phase27_full_backup.json"
    with json_path.open('w') as f:
        json.dump(all_data, f, indent=2, sort_keys=True)
    print(f"\n✓ JSON backup saved: {json_path}")

    # ============================================================
    # Save Markdown (human-readable)
    # ============================================================
    md_path = OUTPUT_DIR / "27_phase27_full_backup.md"
    with md_path.open('w') as f:
        f.write("# Phase 27 — Full Data Backup\n\n")
        f.write("Auto-generated from logs/p27_pillar/. Source of truth before output deletion.\n\n")
        f.write(f"**Extracted from**: {LOG_DIR}\n")
        f.write(f"**Cells**: {len(cells)}\n")
        f.write(f"**Scenes**: {len(SCENES)}\n\n")

        f.write("## Cell list\n\n")
        for cell in cells:
            seeds = discover_seeds(cell)
            n_logs = sum(len(all_data[cell][s]) for s in seeds)
            f.write(f"- `{cell}`: seeds {seeds}, {n_logs} scene-seed logs\n")
        f.write("\n")

        # === Per-cell tables ===
        for cell in cells:
            f.write(f"## Cell: `{cell}`\n\n")
            seeds = discover_seeds(cell)

            for seed in seeds:
                f.write(f"### Seed {seed}\n\n")
                f.write("| Scene | PSNR | SSIM | LPIPS | N_gauss | Best PSNR | Time (s) |\n")
                f.write("|-------|------|------|-------|---------|-----------|----------|\n")

                def fmt(v, prec=4, default='N/A'):
                    if v is None: return default
                    if prec == 0: return str(int(v))
                    return f"{v:.{prec}f}"

                psnrs, ssims, lpipss, ngauss = [], [], [], []
                for scene in SCENES:
                    if scene in all_data[cell][seed]:
                        m = all_data[cell][seed][scene]
                        psnr = m.get('psnr')
                        ssim = m.get('ssim')
                        lpips = m.get('lpips')
                        ng = m.get('ngauss')
                        best = m.get('best_psnr')
                        tt = m.get('training_time_s')

                        f.write(f"| {scene} | {fmt(psnr,4)} | {fmt(ssim,4)} | "
                                f"{fmt(lpips,4)} | {fmt(ng,0)} | "
                                f"{fmt(best,4)} | {fmt(tt,1)} |\n")

                        if psnr is not None: psnrs.append(psnr)
                        if ssim is not None: ssims.append(ssim)
                        if lpips is not None: lpipss.append(lpips)
                        if ng is not None: ngauss.append(ng)
                    else:
                        f.write(f"| {scene} | N/A | N/A | N/A | N/A | N/A | N/A |\n")

                if psnrs:
                    f.write(f"| **MEAN** | "
                            f"**{sum(psnrs)/len(psnrs):.4f}** | "
                            f"**{sum(ssims)/len(ssims):.4f}** | "
                            f"**{sum(lpipss)/len(lpipss):.4f}** | "
                            f"**{int(sum(ngauss)/len(ngauss))}** | — | — |\n")

                f.write("\n")

        # === Cross-cell summary table ===
        f.write("## Cross-cell summary (PSNR mean per cell, seed 42)\n\n")
        f.write("| Cell | PSNR mean | SSIM mean | LPIPS mean | N_gauss mean | N scenes |\n")
        f.write("|------|-----------|-----------|------------|--------------|----------|\n")

        for cell in cells:
            seed = 42 if 42 in discover_seeds(cell) else (discover_seeds(cell)[0] if discover_seeds(cell) else None)
            if seed is None:
                continue
            cell_data = all_data[cell].get(seed, {})
            psnrs = [v['psnr'] for v in cell_data.values() if v.get('psnr')]
            ssims = [v['ssim'] for v in cell_data.values() if v.get('ssim')]
            lpipss = [v['lpips'] for v in cell_data.values() if v.get('lpips')]
            ngauss = [v['ngauss'] for v in cell_data.values() if v.get('ngauss')]

            if psnrs:
                f.write(f"| `{cell}` | "
                        f"{sum(psnrs)/len(psnrs):.4f} | "
                        f"{sum(ssims)/len(ssims):.4f} | "
                        f"{sum(lpipss)/len(lpipss):.4f} | "
                        f"{int(sum(ngauss)/len(ngauss))} | "
                        f"{len(psnrs)}/8 |\n")

        # === Disk usage info ===
        f.write("\n## Notes\n\n")
        f.write("- All data extracted from text logs (logs/p27_pillar/ kept)\n")
        f.write("- Output directory (output/p27_pillar/) safe to delete after this backup\n")
        f.write("- Includes: PSNR, SSIM, LPIPS, N_gauss, Best PSNR, Training time\n")
        f.write("- Format: JSON (machine) + Markdown (human)\n")

    print(f"✓ Markdown backup saved: {md_path}")

    # ============================================================
    # Summary stats
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    total_logs = sum(
        len(all_data[c][s])
        for c in cells
        for s in all_data[c]
    )
    print(f"  Total cells:    {len(cells)}")
    print(f"  Total scene-seed logs: {total_logs}")
    print(f"  Expected (cells × 8 scenes × seeds): {len(cells) * 8 * 1}+")
    print()
    print(f"  Backup files:")
    print(f"    {json_path}")
    print(f"    {md_path}")
    print()
    print(f"  Safe to delete: output/p27_pillar/")
    print(f"  Command: rm -rf output/p27_pillar/")


if __name__ == "__main__":
    main()
