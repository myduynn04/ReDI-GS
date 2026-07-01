"""
Extract LOO ablation table với PSNR + SSIM + LPIPS + N_gauss.

Usage: cd ~/workspace/representation-3d/duyen/CoR-GS
       python scripts/extract_loo_ablation.py
"""

import json
import re
from pathlib import Path

SCENES = ['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex']

# ============================================================
# Load Phase 27 backup
# ============================================================
backup_path = Path("docs/27_phase27_full_backup.json")
if backup_path.exists():
    with backup_path.open() as f:
        p27 = json.load(f)
    print(f"✓ Loaded Phase 27 backup ({len(p27)} cells)")
else:
    print(f"⚠️ No Phase 27 backup at {backup_path}, will skip")
    p27 = {}


# ============================================================
# Extract Phase 28 tau65 reference (existing logs)
# ============================================================
RX = re.compile(r'10000\s*\|\s*test\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*[\d.]+\s*\|\s*(\d+)')


def extract_log(log_path):
    p = Path(log_path)
    if not p.exists():
        return None
    m = RX.search(p.read_text(errors='ignore'))
    if not m:
        return None
    return {'psnr': float(m.group(1)), 'ssim': float(m.group(2)),
            'lpips': float(m.group(3)), 'ngauss': int(m.group(4))}


def cell_means(cell_data):
    """Compute aggregate means from per-scene dict."""
    if not cell_data:
        return None
    psnrs = [v['psnr'] for v in cell_data.values() if v]
    ssims = [v['ssim'] for v in cell_data.values() if v]
    lpipss = [v['lpips'] for v in cell_data.values() if v]
    ngauss = [v['ngauss'] for v in cell_data.values() if v]
    if not psnrs:
        return None
    return {
        'psnr': sum(psnrs) / len(psnrs),
        'ssim': sum(ssims) / len(ssims),
        'lpips': sum(lpipss) / len(lpipss),
        'ngauss': int(sum(ngauss) / len(ngauss)),
        'n_scenes': len(psnrs),
    }


# ============================================================
# Phase 28 tau65 (Phase 28 logs)
# ============================================================
tau65_data = {sc: extract_log(f"logs/p28_crs_boost/tau65/A3_seed42_{sc}.log") for sc in SCENES}
tau65 = cell_means({k: v for k, v in tau65_data.items() if v})


# ============================================================
# Phase 27 LOO cells (from backup JSON)
# ============================================================
def get_p27_cell(cell_name, seed='42'):
    """Get cell data from Phase 27 backup."""
    if cell_name not in p27:
        return None
    if seed not in p27[cell_name]:
        return None
    return cell_means(p27[cell_name][seed])


cells_p27 = {
    'trim_full':         get_p27_cell('trim_full'),
    'loo_no_ldepth':     get_p27_cell('loo_no_ldepth'),
    'loo_no_ldepth_crs': get_p27_cell('loo_no_ldepth_crs'),
    'loo_no_drop':       get_p27_cell('loo_no_drop'),
    'loo_no_opa':        get_p27_cell('loo_no_opa'),
    'loo_no_lfcf':       get_p27_cell('loo_no_lfcf'),
    'loo_no_absgs':      get_p27_cell('loo_no_absgs'),
    'loo_no_crs':        get_p27_cell('loo_no_crs'),
}


# ============================================================
# Print ablation table — Markdown format for thesis
# ============================================================
print(f"\n{'='*90}")
print(f"  LOO ABLATION TABLE — Full metrics")
print(f"  Reference: Phase 28 tau65")
print(f"{'='*90}\n")

# Header
print(f"| Cell                       | PSNR    | SSIM   | LPIPS  | N_gauss | N_scenes |")
print(f"|----------------------------|---------|--------|--------|---------|----------|")

# Reference tau65
if tau65:
    print(f"| **tau65 (full recipe)**    | "
          f"{tau65['psnr']:.4f} | {tau65['ssim']:.4f} | {tau65['lpips']:.4f} | "
          f"{tau65['ngauss']} | {tau65['n_scenes']}/8 |")

# Phase 27 cells
for cell_name, data in cells_p27.items():
    if data:
        print(f"| {cell_name:<26} | "
              f"{data['psnr']:.4f} | {data['ssim']:.4f} | {data['lpips']:.4f} | "
              f"{data['ngauss']} | {data['n_scenes']}/8 |")
    else:
        print(f"| {cell_name:<26} | (no data) |  |  |  |  |")

# ============================================================
# Δ table (vs tau65)
# ============================================================
print(f"\n{'='*90}")
print(f"  Δ ABLATION (vs tau65) — Contribution per pillar")
print(f"{'='*90}\n")

print(f"| Pillar                       | ΔPSNR    | ΔSSIM   | ΔLPIPS  | Note |")
print(f"|------------------------------|----------|---------|---------|------|")

if tau65:
    # Mapping LOO cell → pillar name
    pillar_map = [
        ('L_depth (FSGS-adopted)',           'loo_no_ldepth',     None),
        ('L_depth + CRS combined',           'loo_no_ldepth_crs', 'Joint loss component'),
        ('DropAnSH (borrowed)',              'loo_no_drop',       None),
        ('Opacity decay (ours)',             'loo_no_opa',        None),
        ('EFA-GS LFCF (ours)',               'loo_no_lfcf',       None),
        ('EFA-GS AbsGS (ours)',              'loo_no_absgs',      None),
        ('CRS framework (ours)',             'loo_no_crs',        '3-way verified +0.06'),
    ]
    for pillar, cell_name, note in pillar_map:
        loo = cells_p27.get(cell_name)
        if loo:
            dpsnr = tau65['psnr'] - loo['psnr']
            dssim = tau65['ssim'] - loo['ssim']
            dlpips = loo['lpips'] - tau65['lpips']  # LPIPS: lower better → flip sign
            note_str = note if note else '-'
            print(f"| {pillar:<28} | +{dpsnr:.4f} | +{dssim:.4f} | +{dlpips:.4f} | {note_str} |")
        else:
            print(f"| {pillar:<28} | (no data) |  |  |  |")

# ============================================================
# CRS framework — 3-way verified detail
# ============================================================
print(f"\n{'='*70}")
print(f"  CRS framework 3-way verification")
print(f"{'='*70}\n")

# Phase 29-RP no_crs (R ON, fresh) — from logs
no_crs_rp = {sc: extract_log(f"logs/p29_nocrs_verify/no_crs_{sc}.log") for sc in SCENES}
no_crs_strict_rp = {sc: extract_log(f"logs/p29_nocrs_verify/no_crs_strict_{sc}.log") for sc in SCENES}

no_crs_rp_m = cell_means({k: v for k, v in no_crs_rp.items() if v})
no_crs_strict_rp_m = cell_means({k: v for k, v in no_crs_strict_rp.items() if v})

print(f"| Reference                                | PSNR    | SSIM   | LPIPS  | Δ vs tau65 |")
print(f"|------------------------------------------|---------|--------|--------|------------|")
if tau65:
    print(f"| **tau65 (Phase 28 best)**                | "
          f"{tau65['psnr']:.4f} | {tau65['ssim']:.4f} | {tau65['lpips']:.4f} | (ref) |")
if cells_p27.get('loo_no_crs'):
    d = cells_p27['loo_no_crs']
    print(f"| loo_no_crs (Phase 27, R ON, original)    | "
          f"{d['psnr']:.4f} | {d['ssim']:.4f} | {d['lpips']:.4f} | "
          f"+{tau65['psnr']-d['psnr']:.4f} |")
if no_crs_rp_m:
    print(f"| no_crs (Phase 29-RP, R ON, fresh)        | "
          f"{no_crs_rp_m['psnr']:.4f} | {no_crs_rp_m['ssim']:.4f} | {no_crs_rp_m['lpips']:.4f} | "
          f"+{tau65['psnr']-no_crs_rp_m['psnr']:.4f} |")
if no_crs_strict_rp_m:
    print(f"| no_crs_strict (Phase 29-RP, R OFF)       | "
          f"{no_crs_strict_rp_m['psnr']:.4f} | {no_crs_strict_rp_m['ssim']:.4f} | {no_crs_strict_rp_m['lpips']:.4f} | "
          f"+{tau65['psnr']-no_crs_strict_rp_m['psnr']:.4f} |")

# Adopted value
print(f"\n→ **Adopted CRS framework contribution: +0.06 dB** (average of 3 baselines)")
print(f"→ Δ tau65 vs no_crs (3-way average): +{(tau65['psnr'] - (cells_p27['loo_no_crs']['psnr'] + no_crs_rp_m['psnr'] + no_crs_strict_rp_m['psnr'])/3) if tau65 and cells_p27.get('loo_no_crs') and no_crs_rp_m and no_crs_strict_rp_m else '?':.4f}")

# ============================================================
# Per-scene breakdown — tau65 reference
# ============================================================
print(f"\n{'='*90}")
print(f"  PER-SCENE — tau65 (Phase 28 best)")
print(f"{'='*90}\n")

print(f"| Scene    | PSNR    | SSIM   | LPIPS  | N_gauss |")
print(f"|----------|---------|--------|--------|---------|")
for sc in SCENES:
    v = tau65_data.get(sc)
    if v:
        print(f"| {sc:<8} | {v['psnr']:.4f} | {v['ssim']:.4f} | {v['lpips']:.4f} | {v['ngauss']} |")

# ============================================================
# Save to file for thesis
# ============================================================
output_path = Path("docs/27_loo_ablation_table.md")
print(f"\n💾 To save table to file:")
print(f"   python scripts/extract_loo_ablation.py > {output_path}")
