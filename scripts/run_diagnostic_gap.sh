#!/bin/bash
# ============================================================
# [CRSGaussian Diagnostic] Run gap analysis on 4 representative scenes
# File: scripts/run_diagnostic_gap.sh
#
# Scenes: fern (easy), flower (texture), orchids (hard), fortress (specular)
#
# USAGE:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_diagnostic_gap.sh
#
# Single scene:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_diagnostic_gap.sh fern
#
# All 8 scenes:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_diagnostic_gap.sh all
# ============================================================

SCENE_ARG=${1:-default}

DATA_ROOT="data/nerf_llff_data"
DEFAULT_SCENES="fern flower orchids fortress"
ALL_SCENES="fern flower fortress horns leaves orchids room trex"

case ${SCENE_ARG} in
    all|ALL)     SCENES="${ALL_SCENES}" ;;
    default)     SCENES="${DEFAULT_SCENES}" ;;
    *)           SCENES="${SCENE_ARG}" ;;
esac

LOGDIR="logs/diagnostic_gap"
mkdir -p ${LOGDIR}

# ── Tìm checkpoint path (fallback logic) ──
find_model_path() {
    local scene=$1
    local p1="output/ablation_track_b/B1b_${scene}"
    local p2="output/ablation_b1_start/B1_s1000_${scene}"

    if [ -f "${p1}/point_cloud/iteration_10000/point_cloud.ply" ]; then
        echo "$p1"
    elif [ -f "${p2}/point_cloud/iteration_10000/point_cloud.ply" ]; then
        echo "$p2"
    else
        echo ""
    fi
}

echo ""
echo "============================================================"
echo "  Diagnostic Gap Analysis — ${SCENES}"
echo "============================================================"

# Verify all checkpoints exist
for s in ${SCENES}; do
    mp=$(find_model_path "$s")
    if [ -z "$mp" ]; then
        echo "[ERROR] No checkpoint for scene=${s}"
        echo "  Checked: output/ablation_track_b/B1b_${s}/"
        echo "           output/ablation_b1_start/B1_s1000_${s}/"
        exit 1
    fi
    echo "  ${s} → ${mp}"
done

echo ""
echo "Running 4 tests per scene..."
echo ""

for s in ${SCENES}; do
    mp=$(find_model_path "$s")
    log="${LOGDIR}/${s}.log"

    echo "================================================================"
    echo " Scene: ${s} | model: ${mp}"
    echo "================================================================"

    python -u scripts/diagnostic_gap_analysis.py \
        --model_path "${mp}" \
        --source_path "${DATA_ROOT}/${s}" \
        --n_views 3 \
        --iteration 10000 \
        --eval \
        --save_json \
        2>&1 | tee "${log}"

    echo ""
done

# ── Aggregate summary ──
echo ""
echo "================================================================"
echo "  AGGREGATE SUMMARY (${SCENES})"
echo "================================================================"

# Parse JSON results
python3 -c "
import json, os, glob
scenes = '${SCENES}'.split()
results = {}
for s in scenes:
    p = 'logs/diagnostic_gap/{}.json'.format(s)
    if os.path.exists(p):
        with open(p) as f:
            results[s] = json.load(f)

if not results:
    print('  No JSON results found.')
    exit()

# Test 4: DC vs Rest
print('\n  Test 4 — DC vs Rest:')
print('    {:>10s} | {:>8s} | {:>8s} | {:>8s} | {:>14s}'.format(
    'Scene', 'Full', 'DC-only', 'Delta', 'Verdict'))
for s, r in results.items():
    t = r.get('test4', {})
    print('    {:>10s} | {:>8.3f} | {:>8.3f} | {:>+8.3f} | {:>14s}'.format(
        s, t.get('full_psnr',0), t.get('dc_only_psnr',0),
        t.get('delta',0), t.get('verdict','?')))

# Test 5: Angular distance
print('\n  Test 5 — Angular distance:')
print('    {:>10s} | {:>10s} | {:>10s} | {:>14s}'.format(
    'Scene', 'Mean(deg)', 'Max(deg)', 'Verdict'))
for s, r in results.items():
    t = r.get('test5', {})
    print('    {:>10s} | {:>10.1f} | {:>10.1f} | {:>14s}'.format(
        s, t.get('test_nearest_mean',0), t.get('test_nearest_max',0),
        t.get('verdict','?')))

# Test 2: SH divergence
print('\n  Test 2 — SH divergence:')
print('    {:>10s} | {:>10s} | {:>10s} | {:>14s}'.format(
    'Scene', 'test/train', 'rand/train', 'Verdict'))
for s, r in results.items():
    t = r.get('test2', {})
    print('    {:>10s} | {:>9.2f}x | {:>9.2f}x | {:>14s}'.format(
        s, t.get('ratio_test_train',0), t.get('ratio_rand_train',0),
        t.get('verdict','?')))

# Test 6: Error concentration
print('\n  Test 6 — Error concentration:')
print('    {:>10s} | {:>8s} | {:>8s} | {:>8s} | {:>14s}'.format(
    'Scene', 'Top5%', 'Top10%', 'Top20%', 'Verdict'))
for s, r in results.items():
    t = r.get('test6', {})
    print('    {:>10s} | {:>7.1f}% | {:>7.1f}% | {:>7.1f}% | {:>14s}'.format(
        s, t.get('top_5pct',0), t.get('top_10pct',0), t.get('top_20pct',0),
        t.get('verdict','?')))

print()
" 2>&1

echo "================================================================"
echo "  JSON saved: logs/diagnostic_gap/<scene>.json"
echo "  Logs saved: logs/diagnostic_gap/<scene>.log"
echo "================================================================"
