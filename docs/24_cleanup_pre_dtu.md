# Phase 24 — Cleanup (Pre-DTU Scaling) — UMBRELLA

**Date drafted:** 2026-05-29
**Date updated:** 2026-05-29
**Status:** 🟢 24_0 + 24_1 + 24_3 DONE | 24_2 + 24_4 pending

> **Convention**: Phase 24 là umbrella cho toàn bộ pre-DTU cleanup. Sub-phase notation `24_x` cho từng nhóm việc. **KHÔNG nhảy sang Phase 25** — Phase 25 reserved cho việc lớn hơn (DTU scaling / evaluation).

## Sub-phase index

| Sub-phase | Nội dung | Status | Evidence |
|---|---|---|---|
| **24_0** | Trim verify N=24 (test 3 flag trên v1 RoMa) | ✅ DONE 2026-05-29 | `logs/p24_trim_add_v1/` + memory `project_phase24_trim_verify_v1.md` + decisions_log [2026-05-29] |
| **24_1** | NHÓM 1 cleanup (Phase 20 TRIM physical removal) | ✅ DONE 2026-05-29 | inline `[CRSGaussian Phase 24 cleanup 2026-05-29]` comments in arguments/train.py/crs_module.py + `rm` crs_init.py + smoke verify PASS (drift −0.068 in noise floor) |
| **24_2** | NHÓM 2 truly-dead files (sh_curriculum + shape_reg + binocular_consistency) | ⏳ PENDING | (next, ~300 LOC removed, zero risk) |
| **24_3** | Paper defense Q&A docs | ✅ DONE 2026-05-29 | `docs/24_3_paper_defense_removed_features.md` (canonical defense for removed features) |
| **24_4** | NHÓM 3-5 (backups + dead flags Nhóm 4 + reject scripts Nhóm 5) | ⏳ PENDING | TBD |

## 24_0 result (anchor for 24_1 justification)

**Phase 24 trim-verify N=24** (2026-05-29): N=24 paired vs Phase 22 anchor 21.918:
- 3 individual flags WASH (Δ ≈ 0 ±0.10)
- Stack-3 = ❌ HURT −0.796 (anti-synergy NEW finding)
- → 24_1 NHÓM 1 SAFE to remove physical
- → Plus: paper contribution #5 "legacy CRS stack incompatible with dense init"

Detail: `docs/04_decisions_log.md` [2026-05-29] + `memory project_phase24_trim_verify_v1.md`.
**Context:** Phase 22 RoMa v1 + A3-TRIM 8-module recipe = project best **21.918 PSNR**. Phase 23 ablation confirms 4 contributions. Trước khi mở rộng sang DTU 3-view, cần dọn dead code từ các phase đã reject để (a) tránh bug khi scaling, (b) làm paper code release sạch, (c) giảm noise grep/review.

> ⚠️ **CRITICAL CONSTRAINT** — Không được đụng chạm 8-module A3-TRIM recipe.
> Sau cleanup PHẢI verify: smoke 1-scene 1-seed Phase 22 recipe → PSNR ≈ 21.918 region.
> Nếu khác → revert toàn bộ cleanup, debug.

---

## 1. Phase 22 recipe — VERIFIED INTACT (do NOT touch)

8 module sau wire đúng trong `train.py` — cleanup KHÔNG được động:

| Flag | train.py line | Module file | Status |
|---|---|---|---|
| `use_depth_prior` | 122, 439, 547+ | `utils/depth/depth_alignment.py` + `utils/depth/depth_model.py` | KEEP |
| `use_d_cycle` | 555, 563, 599, 601 | `utils/crs/d_cycle.py` | KEEP |
| `use_crs_modulated_sh_freeze` | 811 | `utils/crs/sh_freeze.py` | KEEP |
| `use_sh_reliability` | 576, 832 | `utils/crs/sh_stability.py` | KEEP |
| `use_dropansh` | 288, 299 (renderer) | `utils/regularizer/dropansh.py` | KEEP |
| `use_opacity_decay` | 848 | inline trong train.py | KEEP |
| `use_lfcf` | 713 | `utils/densify/lfcf.py` | KEEP |
| `absdensify` | direct `opt.absdensify` | scene/gaussian_model.py | KEEP |

Core dependencies cũng KEEP:
- `utils/crs/crs_module.py` (core CRS update, always active when use_depth_prior=True)
- `utils/depth/depth_warping.py` (used by D_cycle)
- `scene/gaussian_model.py` (model với CRS attribute)
- `gaussian_renderer/__init__.py` (renderer với DropAnSH gate)

---

## 2. CLEANUP CATEGORIES — 5 nhóm theo risk

### **NHÓM 1 — Phase 20 TRIM completion** (HIGH PRIORITY, ZERO RISK)

CLAUDE.md ghi rõ Phase 20 TRIM N=24 verified: bỏ `informed_crs_init` + `use_crs_pruning` + `use_r_visible` (recipe 11→8 module). NHƯNG code vẫn còn 3 flag này dangling. Phase 22 KHÔNG set chúng → default OFF → các nhánh code skip → REMOVE = byte-identical kết quả 21.918.

**Risk: ZERO** (nếu làm coordinated cross-file)

| # | File | Lines | Action | Lý do |
|---|---|---|---|---|
| 1.1 | `arguments/__init__.py` | 89-112 | Delete `informed_crs_init` + 6 sub-flags (`crs_init_use_reproj/depth/view`, `crs_init_w_reproj/depth/view`, `crs_init_tau_r/gamma/eta`, `crs_densify_inherit`) | Phase 20 TRIM removed (verified N=24 ±0.10) |
| 1.2 | `arguments/__init__.py` | 247 | Delete `use_pos_constraint = False` | Phase 4 disabled vĩnh viễn (PSNR −3dB, CLAUDE.md line 268-271) |
| 1.3 | `arguments/__init__.py` | 248 | Delete `use_crs_pruning = False` | Phase 20 TRIM removed |
| 1.4 | `arguments/__init__.py` | 304-306 | Delete `use_r_visible` + 2 sub-flags (`r_visible_occlusion_tolerance`, `r_visible_min_views`) | Phase 20 TRIM removed |
| 1.5 | `train.py` | 173-198 (approx) | Delete `informed_crs_init` import + gating block | Phase 20 TRIM removed |
| 1.6 | `train.py` | ~572 | Remove `use_r_visible=opt.use_r_visible` param từ `update_crs(...)` call | Phase 20 TRIM removed |
| 1.7 | `train.py` | 694-702 | Delete `pos_constraint` + `crs_pruning` gate block (revert `densify_and_prune` call về signature gốc) | Phase 4 + Phase 20 dead |
| 1.8 | `utils/crs/crs_init.py` | TOÀN FILE | Delete (356 lines) | Sole consumer = informed_crs_init flag (Phase 20 trimmed) |
| 1.9 | `utils/crs/crs_module.py` | check `use_r_visible` parameter signature | Verify trước khi remove param + r_visible logic | Phase 20 TRIM removed — cần đọc trước |
| 1.10 | `scene/gaussian_model.py` | check `densify_and_prune` signature | Có thể có `_crs_dict` param từ Phase 4 — verify | Phase 4/20 dead |

**Prerequisite trước khi remove 1.6/1.9**: Đọc `utils/crs/crs_module.py` để hiểu signature `update_crs()` và xác nhận remove `use_r_visible` param + r_visible compute branch không break logic chính (D_cycle + S_stability).

---

### **NHÓM 2 — Dead utility files** (LOW RISK)

Các file dưới đã verify **không được import bất kỳ đâu** trong production code (train.py + renderer). Cleanup an toàn nếu re-verify grep zero-import lần cuối.

**Risk: LOW** (re-grep before delete)

| # | File | Phase | Verdict source | Action |
|---|---|---|---|---|
| 2.1 | `utils/crs/sh_curriculum.py` | Phase 19 | DROPPED (CLAUDE.md line 1945-1951) | Delete |
| 2.2 | `utils/regularizer/shape_reg.py` | Phase 15 | REJECTED (CLAUDE.md line 1747) | Delete |
| 2.3 | `utils/regularizer/density_voxel.py` | Phase 2d | REJECTED (memory `density_aware_dropout_idea`) | Delete |
| 2.4 | `utils/regularizer/density_covariance.py` | Phase 2d | REJECTED | Delete |
| 2.5 | `utils/regularizer/sh_dropout.py` | Track B B3 mode | use_dropout system dead (Nhóm 4) | Delete (sau cleanup use_dropout) |
| 2.6 | `utils/loss/binocular_consistency.py` | Phase 14 | REJECTED — **NHƯNG** Explore note "kept for Phase 18 reference" | VERIFY trước, có thể KEEP |
| 2.7 | `utils/loss/covisibility_depth.py` | Phase 11 Step 3 | Pre-flight killed (CLAUDE.md line 1605) | Delete sau verify zero-import |

**Verification command (chạy trước mỗi delete)**:
```bash
grep -rn "from utils.crs.sh_curriculum\|import.*sh_curriculum" --include="*.py" .
# Expect: 0 lines (zero import) → safe delete
```

---

### **NHÓM 3 — Backup/orphan artifacts** (TRIVIAL)

**Risk: TRIVIAL**

| # | File | Lý do |
|---|---|---|
| 3.1 | `utils/crs/crs_module.py.pre_cleanup_20260511` | Backup từ 2026-05-11, không còn cần (git là single source of truth) |
| 3.2 | `utils/regularizer/__init__.py.pre_cleanup_20260511` | Như trên |
| 3.3 | `logs/p13_2_gdags_gate.log` | GDAGS REJECTED, log orphan |

---

### **NHÓM 4 — Dead flags from rejected/superseded phases** (LOW-MEDIUM RISK)

Default OFF flags từ phases đã reject hoặc superseded. Code path vẫn còn nhưng không bao giờ trigger. Remove theo Quy tắc 13 + giảm noise grep.

**Risk: LOW-MEDIUM** (cần remove block code tương ứng trong train.py)

| # | Flag(s) | File:Line | Phase | Memory verdict | Action |
|---|---|---|---|---|---|
| 4.1 | `use_loss_reweight`, `lossw_gamma`, `lossw_render_freq` | arguments:296-298 + train.py:309/316/599/607 | Phase 7 | Superseded by Phase 8 SH-freeze hero (+0.189) | Delete flag + remove gating blocks |
| 4.2 | `use_rnrc` + 7 sub-flags (`rnrc_mode/beta/warmup/per_gauss_warmup/floor/norm_mode/tau`) | arguments:261-268 + train.py:778 | Hướng D MVP | Abandoned (no committed phase), proxy RC superseded | Delete flag + remove gating block |
| 4.3 | `use_coreliability_reweight` + 4 sub | arguments:137-141 + train.py:157/353 | Phase 11 Step 1 | MARGINAL +0.014 "keep OFF" (CLAUDE.md line 151-152) | Delete flag + remove gating blocks |
| 4.4 | `dropansh_density_method` | arguments:212 | Phase 2d | REJECTED (memory) | Delete (logic chỉ dùng khi != "uniform") |
| 4.5 | `dropansh_crs_anchor`, `dropansh_crs_sh` | arguments:201-202 | Phase 3 | Deferred (memory `crs_guided_anchor_idea`) | KEEP (Phase 3 marked future work) |
| 4.6 | `freeze_sh_after` | arguments:276 + train.py:866 | DIAG E1 | Superseded by `use_crs_modulated_sh_freeze` | Delete + remove gate |
| 4.7 | `freeze_dc_only`, `freeze_dc_start_iter` | arguments:281-282 + train.py:875 | DIAG A2 | Diagnostic only, unused | Delete + remove gate |
| 4.8 | `disable_r_signal` | arguments:329 + train.py:581 | Phase 9 | Phase 9 D-only test, superseded by Phase 20 TRIM | Delete + remove gate |
| 4.9 | `disable_global_sh_freeze` | arguments:330 + train.py:866 | Phase 9 | Phase 9 cross-test, superseded | Delete + remove gate |
| 4.10 | `use_dropout` + 6 sub (`dropout_mode/base/w_crs/w_sh/max/start_iter`) | arguments:180-186 + train.py:288/298 + renderer | Track B | Superseded by `use_dropansh` (Track B B1 = DropAnSH wrapper) | VERIFY (có thể vẫn dùng B1 mode) trước delete |
| 4.11 | `tier_a_diag`, `tier_a_diag_iters` | arguments:126-127 + train.py:60-61 (import) + 637 | Phase 6 | Diagnostic suite, unused trong recipe | KEEP (diagnostic, useful for paper figures) |

**Prerequisite cho 4.10**: Đọc `gaussian_renderer/__init__.py` để verify `use_dropout` (Track B B1) có là alias cho `use_dropansh` mask hay không. Nếu Track B = wrapper cũ, DropAnSH = wrapper mới same logic → delete an toàn.

---

### **NHÓM 5 — Reject-phase scripts** (per Quy tắc 13)

Theo Quy tắc 13: REJECTED phase → script .sh/.py wrapper xóa, design doc trong `docs/` giữ.

**Risk: LOW** (scripts không ảnh hưởng training code)

| # | Files | Phase | Verdict | Action |
|---|---|---|---|---|
| 5.1 | `scripts/p10a_master.sh`, `p10a_analyze.py`, `p10a_save_renders.sh`, `p10a_diagnostic.sh`, `precompute_dust3r.py` | Phase 10A | DUSt3R AXIS DEAD (CLAUDE.md 62-69) | Delete (5 files) |
| 5.2 | `scripts/p11s{1,2,3,4,5,12_stack}_*` (10+ files) | Phase 11 | 6/6 loss-axis REJECTED | Delete |
| 5.3 | `scripts/p12_crs_pull_multiseed.sh` + analyze | Phase 12 | 3/3 CRS-pull REJECTED | Delete |
| 5.4 | `scripts/p13_2_bottleneck_decompose.py` | Phase 13.2 | Diagnostic post-mortem | KEEP (useful for bottleneck analysis figure) |
| 5.5 | `scripts/p14_lconsist_*` (4 files) | Phase 14 | L_consist REJECTED | Delete |
| 5.6 | `scripts/p15_*` (3 files: `aniso_diag.py`, `shape_pilot.sh`, `shape_analyze.py`) | Phase 15 | shape-reg/blunt-aniso REJECTED | Delete |
| 5.7 | `scripts/p18b_*` (6 files) | Phase 18b | 5/5 fix-hypotheses REFUTED | Delete |
| 5.8 | `scripts/p19_curriculum_*` (2 files) | Phase 19 | SH-curriculum DROPPED | Delete |
| 5.9 | `scripts/p16_normal_redundancy_gate.py` | Phase 16 | Gate verdict completed | KEEP (gate protocol reference) |
| 5.10 | `scripts/p17c_*` (5 files: tier1/tier2 noise studies, a3_resave) | Phase 17c | Status unclear — verify | VERIFY trước delete |
| 5.11 | `scripts/p17e_*` (3 files: step_a/b/c gate simulation) | Phase 17e | Status unclear — verify | VERIFY trước delete |

**KEEP scripts (active hoặc committed recipes)**:
- `scripts/p8_*`, `p9_*` (Phase 8 lock + Phase 9 simplification)
- `scripts/p13_lfcf_multiseed*`, `p13_1_sweep*`, `p13_overfit_analyze.py` (Phase 13 committed)
- `scripts/p17_c1_*` + `dsine_pkg/` (Phase 17 C1 active)
- `scripts/p18_pilot*`, `p18_gate*` (Phase 18 PDCNet+ KEPT OPEN)
- `scripts/p20_ablation_*` (Phase 20 TRIM verify)
- `scripts/p21_*` (Phase 21 RoMa v2)
- `scripts/p22_*` (Phase 22 RoMa v1 PROJECT BEST)
- `scripts/p23_*` (Phase 23 ablation)
- `scripts/t2min_final_*` (final reference)

---

## 3. RECOMMENDED EXECUTION ORDER

Làm theo thứ tự dưới để minimize risk + dễ verify từng bước:

### Bước 1 — Backup safety (5 phút)
```bash
# Tạo branch backup hiện tại (nếu git repo)
git status
git checkout -b backup/pre-cleanup-20260529
git checkout -  # quay lại working branch

# Snapshot disk
ls -la utils/crs/ utils/regularizer/ utils/loss/ utils/densify/ > /tmp/pre_cleanup_files.txt
```

### Bước 2 — NHÓM 3 trước (TRIVIAL, làm warm-up)
- Delete 3 backup/orphan files
- Smoke verify: not needed (không động code)

### Bước 3 — NHÓM 1 (Phase 20 TRIM)
- Đọc `utils/crs/crs_module.py` để hiểu `update_crs()` signature + r_visible logic
- Coordinated edit: arguments + train.py + crs_module.py (cùng 1 PR/commit)
- Delete `utils/crs/crs_init.py`
- **Smoke verify**: chạy 1 scene (fern) 1 seed với recipe Phase 22 → PSNR phải match 21.918 region (tolerance ±0.10 do single-scene noise)

### Bước 4 — NHÓM 2 (Dead utility files)
- Re-grep mỗi file (xem command box ở Nhóm 2)
- Delete file by file
- **Smoke verify after each batch**: không cần, vì file không được import

### Bước 5 — NHÓM 4 (Dead flags)
- Xử lý theo subgroup, mỗi subgroup commit riêng:
  - 4.1 (loss_reweight) — commit
  - 4.2 (rnrc) — commit
  - 4.3 (coreliability) — commit
  - 4.4 (density_method) — commit
  - 4.6+4.7 (freeze diagnostic) — commit
  - 4.8+4.9 (Phase 9 disable flags) — commit
  - 4.10 (use_dropout) — VERIFY first, possibly defer
- **Smoke verify after 4.1-4.3**: chạy lại Phase 22 smoke 1 scene

### Bước 6 — NHÓM 5 (Reject-phase scripts)
- Delete batch theo memory verdict
- Không cần smoke (không ảnh hưởng code)

### Bước 7 — FINAL VERIFICATION
```bash
# Smoke verify Phase 22 recipe 1 scene
CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --source_path data/nerf_llff_data/fern \
    -m /tmp/post_cleanup_smoke \
    --eval -r 8 --n_views 3 --random_background \
    --iterations 10000 --densify_until_iter 5000 \
    --densify_grad_threshold 0.0005 --gaussiansN 1 \
    --sample_pseudo_interval 1 --start_sample_pseudo 500 \
    --test_iterations 10000 \
    --seed 42 \
    --use_depth_prior --dav2_path ../Depth-Anything-V2 \
    --crs_ema_decay 0.3 --crs_update_interval 100 \
    --use_d_cycle --d_cycle_warmup 1000 --d_cycle_sigma 5.0 --d_cycle_update_freq 100 \
    --use_crs_modulated_sh_freeze --crs_freeze_start 1000 --crs_freeze_tau 0.5 \
    --use_sh_reliability --sh_stability_warmup 1000 --sh_stability_ema_beta 0.95 --crs_w_s 0.33 \
    --use_dropansh --dropansh_pa 0.02 --dropansh_psh 0.2 \
    --use_opacity_decay --opacity_decay_factor 0.999 \
    --use_lfcf --lfcf_init_scaling_max 1.5 --lfcf_init_scaling_min 1.0 \
    --lfcf_last_scaling_max 1.0 --lfcf_pow 1.0 --lfcf_splitting_ub 1.0 \
    --lfcf_interval_times 2 --lfcf_tolerance 1e-5 --lfcf_diffscale True \
    --absdensify

# Expected: "Best test PSNR" ~ 23.8 region (fern Phase 22 = 23.84)
# Tolerance: ±1.3 dB single-scene atomicAdd noise → ±0.5 acceptable
```

Nếu fern PSNR < 23.0 sau cleanup → **REVERT** + debug.

---

## 4. ESTIMATED IMPACT

| Metric | Trước cleanup | Sau cleanup (ước tính) |
|---|---|---|
| `train.py` line count | ~1100 | ~950 (−150) |
| `arguments/__init__.py` line count | ~370 | ~280 (−90) |
| Files in `utils/` (dead) | 7 dead | 0 dead |
| Files in `scripts/` (REJECTED phases) | ~35 | ~5 (keep diagnostic refs) |
| Backup `.pre_cleanup_*` files | 2 | 0 |
| Orphan logs | 1 | 0 |
| **Total LOC reduction** | — | **~500-700 LOC** |

**Compute savings:** N/A — cleanup không thay đổi training speed (code dead đã không trigger).

**Maintenance benefit:**
- Grep `[CRSGaussian` chỉ trả về code còn relevant
- DTU scaling không gặp dangling flag confusion
- Paper code release sạch (reviewer không hỏi "tại sao có file này không dùng?")

---

## 5. ITEMS REQUIRING USER DECISION

Một số mục tôi đánh dấu "VERIFY" hoặc "có thể KEEP" — cần bạn quyết:

| # | Item | Tùy chọn |
|---|---|---|
| Q1 | Nhóm 2.6 `binocular_consistency.py` | (a) Delete (b) Keep cho Phase 18 reference (c) Tôi verify thêm trước khi quyết |
| Q2 | Nhóm 4.5 `dropansh_crs_anchor/sh` (Phase 3 deferred) | (a) Keep (memory says future work) (b) Delete (no concrete plan to use) |
| Q3 | Nhóm 4.10 `use_dropout` (Track B B1) | (a) Verify alias relationship rồi quyết (b) Keep both (no harm) |
| Q4 | Nhóm 4.11 `tier_a_diag` | (a) Keep (diagnostic for paper figures) (b) Delete (clean cleanup) |
| Q5 | Nhóm 5.10/5.11 `p17c_*`, `p17e_*` scripts | (a) Tôi verify status từ decisions_log (b) Skip — keep all (c) Delete all |
| Q6 | Backup branch trước cleanup | (a) Create `backup/pre-cleanup-20260529` (b) Skip — git history đủ |

---

## 6. WHAT I WILL NOT DO without explicit confirmation

- ❌ Sửa bất kỳ file production nào trong `arguments/`, `train.py`, `utils/crs/`, `utils/regularizer/`, `utils/loss/`, `gaussian_renderer/`, `scene/`
- ❌ Xóa file nào trong `utils/`, `scripts/`, `logs/`
- ❌ Xóa branch hoặc commit history
- ❌ Modify `docs/04_decisions_log.md` hoặc `docs/03_task_queue.md` để ghi cleanup result (sẽ làm SAU khi cleanup done)
- ❌ Push lên remote / sync server

---

## 7. NEXT STEPS

1. **Bạn review document này** + trả lời Q1-Q6 + approve/reject từng nhóm
2. Tôi áp dụng theo thứ tự execution (Bước 1-7), commit per subgroup
3. Sau mỗi subgroup → smoke verify nếu áp dụng
4. Final smoke verify Phase 22 recipe fern → confirm PSNR intact
5. Update `docs/04_decisions_log.md` với entry `[2026-05-XX] Phase 24 cleanup completed`
6. → Proceed to DTU scaling (Phase 25)

---

**Status:** ⏳ AWAITING APPROVAL
**Approver:** User (bkt.resinet@gmail.com)
**Estimated total work:** 2-3 ngày coding + 1 ngày verify
