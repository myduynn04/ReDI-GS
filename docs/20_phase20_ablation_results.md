# Phase 20 — Ablation Results (Table-4 style)

> **Date**: 2026-05-23 → 2026-05-24
> **Status**: MVS bảng + TRIM verify N=24 done. Dense bảng 1-seed first-look done.
> **Output dirs DELETED 2026-05-24** (user cleanup). Số liệu raw lưu file này.

---

## 1. Mục đích

Unified attribution-table kiểu **Binocular3DGS Table 4** thay 6+ partial-attribution rải rác Phase 8/9/13 (mỗi phase backbone khác + methodology không đồng nhất). Đo đóng góp BIÊN mỗi block trên A3-FULL bằng cùng 1 setup.

**Methodology LOO 7-block** (không full-factorial 2¹¹ vì bất khả + dependency graph: `use_depth_prior` = prerequisite của CRS/D_cycle/SH-freeze/R_vis → cascade khi tắt). Cost-axis = N_gaussians (đếm, tin được). KHÔNG dùng wall-clock — cross-session noise 3×.

---

## 2. Configurations

11 module gốc trong A3-FULL → block hóa 7 blocks cho LOO khả thi:

| Block | Module(s) | Flag |
|---|---|---|
| Depth+CRS cascade | depth-prior + CRS framework (informed_init + D_cycle + R_vis + SH-CRS + S_stab + CRS-prune) | `--use_depth_prior` (prerequisite) |
| CRS-prune | CRS-prune + informed_crs_init | `--use_crs_pruning` + `--informed_crs_init` |
| D_cycle | D_cycle signal | `--use_d_cycle` |
| SH-CRS | SH-modulated freeze + S_stability | `--use_crs_modulated_sh_freeze` + `--use_sh_reliability` |
| DropAnSH | dropout regularizer (anchor + SH) | `--use_dropansh` |
| Opacity-decay | continuous opacity ×0.999/iter | `--use_opacity_decay` |
| EFA | LFCF + AbsGS (bundle, frequency-axis densify) | `--use_lfcf --absdensify` |
| R_visible | (sub-block of CRS) visibility-aware R_i | `--use_r_visible` |
| Dense init | (init-axis riêng) PDCNet+ dense pcd | `fused.ply` swap |

---

## 3. MVS bảng — 1-seed first-look

**Setup**: 8 scene × seed 42 = N=8 paired. Baseline = `FULL (A3-MVS)` từ `logs/p13_lfcf` seed 42. Sàn nhiễu paired SE ≈ ±0.46.

| Config | Module tắt | PSNR | SSIM | LPIPS | N_gauss | Δ paired |
|---|---|---|---|---|---|---|
| base | tất cả (3DGS thuần) | 20.833 (N=1, OOM 7/8) | 0.6745 | 0.2376 | 894653 | −2.526 |
| **no_depthcrs** | depth + cả CRS cascade | 20.469 | 0.7001 | 0.2082 | 233007 | **−0.862** ✅ |
| **no_drop** | DropAnSH | 20.768 | 0.7033 | 0.1988 | 93632 | **−0.563** ✅ |
| no_shcrs | SH-modulated freeze + S_stab | 21.034 | 0.7231 | 0.1915 | 87860 | −0.296 ⚠️ |
| no_opacity | opacity-decay | 21.171 | 0.7289 | 0.1858 | 92111 | −0.160 ⚠️ |
| no_dcycle | D_cycle | 21.180 | 0.7284 | 0.1869 | 87789 | −0.151 ⚠️ |
| no_efa | LFCF + AbsGS (bundle) | 21.216 | 0.7277 | 0.1894 | 89400 | −0.115 ⚠️ |
| no_crsprune | CRS-prune + informed_init | 21.333 | 0.7340 | 0.1833 | 94216 | **+0.002** ❌ |
| no_rvis | R_visible | 21.381 | 0.7341 | 0.1840 | 90477 | **+0.051** ❌ |
| **FULL (A3-MVS)** | — | **21.330** | **0.7314** | **0.1857** | **89762** | **0 REF** |
| FULL+denseinit | thêm dense-init | 21.595 | 0.7571 | 0.1699 | 213386 | +0.265 ✅ |

**Synergy 4× overlap**: Sum |Δ_LOO| ≈ 2.15 vs (FULL − base) = 0.498 → mỗi LOO Δ thổi phồng ~4× vì modules trùng job (anti-overfit). KHÔNG cộng |Δ| được. Bảng trả lời "bỏ thì mất bao nhiêu", KHÔNG trả lời "thêm vào đóng góp bao nhiêu".

---

## 4. MVS TRIM verify — N=24 decisive

**Setup**: 2 config × 8 scene × 3 seed (42, 137, 9999) = N=24 paired vs FULL N=24. Sàn paired SE ≈ ±0.10.

**Decision rule (pre-locked TRƯỚC khi chạy)**:
- Δ ≥ +0.10 + CI excludes 0 → ✅ trim improves
- Δ ∈ [−0.10, +0.10] → ✅ trim no-harm (bỏ được, recipe gọn hơn không mất gì)
- Δ ≤ −0.10 → ❌ giữ (synergy lộ ra)

**Kết quả**:

| Config | Δ paired N=24 | N_gauss | Verdict |
|---|---|---|---|
| `no_crsprune_rvis` (tắt 3: informed_init + prune + rvis) | **+0.0042** | 93374 | ✅ trim no-harm |
| `no_prune_rvis` (tắt 2 giữ init) | **−0.0247** | 92467 | ✅ trim no-harm |
| **FULL (A3-MVS)** | 0 REF | **89834** | 21.330 baseline |

**VERDICT LOCKED**: bỏ 3 module verified-dead trên MVS:
- ❌ `--informed_crs_init` (+ weight flags)
- ❌ `--use_crs_pruning`
- ❌ `--use_r_visible`

→ Recipe **11 → 8 module** (= A3-TRIM). PSNR không đổi đo được. N_gauss +3-4% (mất kênh xóa isolated qua CRS).

**Pattern**: drop-3 (+0.004) hơi nhỉnh drop-2 (−0.025) → informed_init không pay-off riêng khi mất CRS-prune consumer. Bỏ sạch sạch hơn.

---

## 5. Dense bảng — 1-seed first-look

**Setup**: 8 scene × seed 42 = N=8 paired. Backbone = PDCNet+ dense init (Phase 18). Baseline = `FULL+denseinit` (= A3-FULL on dense, reuse `logs/p18_pilot` seed 42 = **21.595**, Phase 18 N=24 SIG +0.270).

| Config | Module tắt (so trim_full) | PSNR | SSIM | LPIPS | N_gauss | Δ vs FULL-dense |
|---|---|---|---|---|---|---|
| **FULL on dense** (= Phase 18 reuse) | — | **21.595** | **0.7571** | **0.1699** | **213386** | **0 REF** |
| trim_full (A3-TRIM on dense) | 3 dead modules đã verify | 21.525 | 0.7574 | 0.1699 | 222378 | −0.070 |
| trim_no_efa | + EFA (LFCF + AbsGS) | 21.571 | 0.7556 | 0.1712 | 218921 | **−0.025** |
| **trim_no_drop** | + DropAnSH | 20.954 | 0.7280 | 0.1941 | 221154 | **−0.641** ✅ |
| trim_no_opacity | + opacity-decay | 21.355 | 0.7506 | 0.1733 | 298547 | −0.241 ⚠️ |
| trim_no_dcycle | + D_cycle | 21.586 | 0.7580 | 0.1698 | 222651 | **−0.009** |
| trim_no_shcrs | + SH-CRS freeze | 21.553 | 0.7576 | 0.1698 | 221510 | **−0.043** |
| **trim_no_depthcrs** | + depth+CRS cascade | 20.979 | 0.7339 | 0.1898 | 413858 | **−0.617** ✅ |
| base | tất cả (OOM 4/8) | 21.195 (N=4) | 0.7060 | 0.2163 | 435622 | — |

**OOM note**: `base/leaves` OOM trên dense (distCUDA2 KNN), tương tự MVS bảng `base/flower` OOM. Expected — base không có anti-overfit, N_gauss bùng nổ. Không ảnh hưởng verdict.

---

## 6. Cross-backbone comparison

| Module | MVS Δ vs A3-FULL | Dense Δ vs A3-FULL-dense | Shift |
|---|---|---|---|
| DropAnSH | −0.563 ✅ | −0.641 ✅ | stable trụ |
| depth+CRS cascade | −0.862 ✅ | −0.617 ✅ | yếu nhẹ trên dense |
| opacity-decay | −0.160 ⚠️ | −0.241 ✅ | rõ hơn trên dense |
| EFA (LFCF+AbsGS) | −0.115 ⚠️ | **−0.025** | **dispensable trên dense** |
| D_cycle | −0.151 ⚠️ | **−0.009** | **dispensable trên dense** |
| SH-CRS (hero MVS) | −0.296 ⚠️ | **−0.043** | **dispensable trên dense** |

**3 module CRS-axis (EFA, D_cycle, SH-CRS) shift từ "small contribution" trên MVS → "near zero" trên dense.** Mechanistic: dense init thêm thông tin ngoài → CRS framework (trích thông tin nội bộ) overlap → marginal value giảm.

---

## 7. Findings tổng

1. **MVS recipe**: 11 module → **8 module sau trim** (verified N=24 no-harm). Bỏ:
   - `informed_crs_init`, `use_crs_pruning`, `use_r_visible`

2. **3 trụ cross-backbone** (vượt/sát sàn trên cả 2 backbone):
   - **DropAnSH** (mượn Co-Adapt) — trụ thứ 2 toàn recipe
   - **depth+CRS cascade** (depth mượn LoopSparseGS+Chung; CRS framework là của bạn)
   - **opacity-decay** (mượn Binocular3DGS)

3. **3 module dispensable trên dense** (suggestive 1-seed, cần N=24 confirm):
   - EFA (LFCF+AbsGS) — mượn EFA-GS + AbsGS
   - D_cycle — của bạn
   - SH-CRS modulated freeze + S_stability — của bạn (Phase 8 hero MVS)

4. **Phase 18 dense init replicate**: 1-seed +0.265 ≈ Phase 18 N=24 +0.270 — stable.

5. **Synergy 4× overlap** (Phase 9 confirmed quantitatively): modules trùng job. KHÔNG cộng |Δ| được.

6. **CRS-axis của user**: 2/4 channel pay-off rõ trên MVS (SH-CRS, D_cycle), 2/4 dead (CRS-prune, R_visible) + 1/1 init dead (informed_init). Trên dense, **3 channel pay-off cũng giảm về ~0** — dense init ăn vào cùng "lỗ hổng info" mà CRS đang trám.

---

## 8. Caveats

- **MVS 1-seed bảng**: sàn ±0.46 → mọi |Δ| < 0.46 trong noise → cần multi-seed mới chốt sub-sàn modules
- **MVS TRIM N=24**: sàn ±0.10 → 2 config trong [−0.10, +0.10] = verified no-harm (decisive)
- **Dense 1-seed bảng**: sàn ±0.46 — **3 module dispensable cùng dấu (−0.025, −0.009, −0.043) trong noise** → pattern suggestive nhưng cần multi-seed để chốt
- **Dense trim_full Δ=−0.07 vs FULL-dense**: trong noise, nhưng pattern khác MVS (where trim no-harm exact)
- **Base config OOM trên cả 2 backbone**: 3DGS thuần → N_gauss bùng nổ → distCUDA2 KNN OOM một số scene (MVS: flower; Dense: leaves). Expected
- **Phase 18 backbone NOT pre-registered GO**: C3 criterion fail (horns/trex regress, thin-structure capacity ceiling) — dense path không thay được MVS path phổ thông

---

## 9. Files & scripts (kept-local)

### Scripts
- `scripts/p20_ablation_run.sh` — MVS LOO runner (10 configs)
- `scripts/p20_ablation_dense_run.sh` — Dense LOO standalone runner (8 configs)
- `scripts/p20_ablation_analyze.py` — Analyzer (REF_LABEL env override)

### Logs preserved (NOT deleted)
- `logs/p20_ablation/` — MVS LOO logs (per config / per scene-seed)
- `logs/p20_ablation_dense/` — Dense LOO logs

### Output dirs DELETED 2026-05-24 by user
- `output/p20_ablation/` — MVS ckpts/renders
- `output/p20_ablation_dense/` — Dense ckpts/renders

→ **Số liệu PSNR/SSIM/LPIPS/N_gauss đã preserved trong bảng trên + logs còn nguyên**. Re-run ablation if needed = full pipeline available via scripts.

### Baselines reused (NOT touched)
- `logs/p13_lfcf/` — A3-MVS baseline N=24 (Phase 13 committed)
- `logs/p18_pilot/` — A3-PDCNet+ baseline N=24 (Phase 18 GIỮ MỞ)

---

## 10. Next pending

- **(a)** Multi-seed N=24 cho dense ablation (3 dispensable modules + trim_full) — chốt evaporate trên dense hay không. ~30-40h / 2 GPU.
- **(b)** Multi-seed N=24 cho sub-sàn LOO configs trên MVS (no_efa, no_opacity, no_dcycle, no_shcrs) — paper-quality bảng. ~18-24h.
- **(c)** Pivot research axis mới (Phase 19 SH-curriculum đã DROPPED 2026-05-24, see [[phase19-curriculum-dropped]] memory).

Decision pending. Bảng hiện tại đủ cho narrative high-level (3 trụ cross-backbone, 2-3 dispensable). N=24 chỉ cần nếu muốn rigorous attribution numbers.
