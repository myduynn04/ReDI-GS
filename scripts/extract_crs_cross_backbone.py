"""
Extract no_CRS composite contribution across 3 init regimes:
  - MVS (Phase 20)
  - PDCNet+ dense (Phase 20 dense)
  - RoMa v1 (Phase 23)
  - RoMa v2: NOT AVAILABLE (no specific ablation script)

Cell of interest:
  trim_full         = full recipe (with CRS)
  trim_no_depthcrs  = depth + CRS removed (composite)
  trim_no_shcrs     = SH-CRS only removed
  trim_no_dcycle    = D_cycle only removed

Output: cross-backbone comparison table for thesis 4.5.3 Discussion.
"""

import re
from pathlib import Path

RX = re.compile(r'10000\s*\|\s*test\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*[\d.]+\s*\|\s*(\d+)')
SCENES = ['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex']

# Phase → log root path
PHASES = {
    'MVS (Phase 20)':       'logs/p20_ablation',
    'PDCNet+ (Phase 20 dense)': 'logs/p20_ablation_dense',
    'RoMa v1 (Phase 23)':   'logs/p23_v1_ablation',
}

CELLS = ['trim_full', 'trim_no_depthcrs', 'trim_no_shcrs', 'trim_no_dcycle']
CELL_LABELS = {
    'trim_full':        'Full recipe',
    'trim_no_depthcrs': 'w/o depth+CRS (composite)',
    'trim_no_shcrs':    'w/o SH-CRS only',
    'trim_no_dcycle':   'w/o D_cycle only',
}


def extract(log_path):
    p = Path(log_path)
    if not p.exists():
        return None
    m = RX.search(p.read_text(errors='ignore'))
    if not m:
        return None
    return {'psnr': float(m.group(1)), 'ssim': float(m.group(2)),
            'lpips': float(m.group(3)), 'ngauss': int(m.group(4))}


def cell_metrics(log_root, cell, seed=42):
    """Get mean metrics for a cell across 8 scenes."""
    data = {}
    for sc in SCENES:
        log_path = f"{log_root}/{cell}/A3_seed{seed}_{sc}.log"
        r = extract(log_path)
        if r:
            data[sc] = r
    if not data:
        return None
    psnrs = [v['psnr'] for v in data.values()]
    ssims = [v['ssim'] for v in data.values()]
    lpipss = [v['lpips'] for v in data.values()]
    ngauss = [v['ngauss'] for v in data.values()]
    return {
        'psnr': sum(psnrs)/len(psnrs),
        'ssim': sum(ssims)/len(ssims),
        'lpips': sum(lpipss)/len(lpipss),
        'ngauss': int(sum(ngauss)/len(ngauss)),
        'n_scenes': len(data),
    }


def main():
    print(f"\n{'='*100}")
    print(f"  CRS framework cross-backbone comparison")
    print(f"  (Phase 20 + Phase 23 LOO ablation data)")
    print(f"{'='*100}\n")

    # Extract all cells × all phases
    results = {}
    for phase_name, log_root in PHASES.items():
        results[phase_name] = {}
        for cell in CELLS:
            metrics = cell_metrics(log_root, cell)
            results[phase_name][cell] = metrics

    # === Cell-level table for each phase ===
    for phase_name in PHASES:
        print(f"\n--- {phase_name} ---")
        print(f"{'Cell':<28} {'PSNR':>9} {'SSIM':>9} {'LPIPS':>9} {'N_gauss':>10} {'N_scenes':>10}")
        print("-" * 80)
        for cell in CELLS:
            m = results[phase_name].get(cell)
            label = CELL_LABELS.get(cell, cell)
            if m:
                print(f"{label:<28} {m['psnr']:>9.4f} {m['ssim']:>9.4f} "
                      f"{m['lpips']:>9.4f} {m['ngauss']:>10} {m['n_scenes']:>5}/8")
            else:
                print(f"{label:<28} {'(no data)':>9}")

    # === Composite CRS contribution table ===
    print(f"\n{'='*100}")
    print(f"  COMPOSITE CRS CONTRIBUTION (depth + CRS removed vs full)")
    print(f"{'='*100}\n")
    print(f"{'Init regime':<28} {'trim_full PSNR':>14} {'trim_no_depthcrs':>17} {'Δ':>10}")
    print("-" * 75)
    for phase_name in PHASES:
        full = results[phase_name].get('trim_full', {})
        nocrs = results[phase_name].get('trim_no_depthcrs', {})
        if full and nocrs:
            f_psnr = full['psnr']
            nc_psnr = nocrs['psnr']
            d = f_psnr - nc_psnr
            print(f"{phase_name:<28} {f_psnr:>14.4f} {nc_psnr:>17.4f} {d:>+10.4f}")
        else:
            print(f"{phase_name:<28} {'(incomplete data)':>14}")

    # === Per-component CRS LOO ===
    print(f"\n{'='*100}")
    print(f"  PER-COMPONENT CRS LOO (vs full)")
    print(f"{'='*100}\n")
    print(f"{'Init regime':<28} {'SH-CRS LOO':>11} {'D_cycle LOO':>12} {'Composite LOO':>14}")
    print("-" * 75)
    for phase_name in PHASES:
        full = results[phase_name].get('trim_full', {})
        sh = results[phase_name].get('trim_no_shcrs', {})
        dc = results[phase_name].get('trim_no_dcycle', {})
        comp = results[phase_name].get('trim_no_depthcrs', {})
        if full:
            f_psnr = full['psnr']
            sh_d = (sh['psnr'] - f_psnr) if sh else None
            dc_d = (dc['psnr'] - f_psnr) if dc else None
            comp_d = (comp['psnr'] - f_psnr) if comp else None
            sh_str = f"{sh_d:>+11.4f}" if sh_d is not None else f"{'n/a':>11}"
            dc_str = f"{dc_d:>+12.4f}" if dc_d is not None else f"{'n/a':>12}"
            comp_str = f"{comp_d:>+14.4f}" if comp_d is not None else f"{'n/a':>14}"
            print(f"{phase_name:<28} {sh_str} {dc_str} {comp_str}")

    # === Thesis-ready format ===
    print(f"\n{'='*100}")
    print(f"  THESIS-READY TABLE (markdown format)")
    print(f"{'='*100}\n")
    print(f"| Init regime         | Full PSNR | no_CRS PSNR | $\\Delta_{{\\text{{CRS}}}}$ |")
    print(f"|---------------------|-----------|-------------|---------|")
    for phase_name in PHASES:
        full = results[phase_name].get('trim_full', {})
        comp = results[phase_name].get('trim_no_depthcrs', {})
        if full and comp:
            f_psnr = full['psnr']
            c_psnr = comp['psnr']
            d = f_psnr - c_psnr
            print(f"| {phase_name:<19} | {f_psnr:.4f}    | {c_psnr:.4f}      | $+{d:.4f}$ |")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
