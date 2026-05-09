# CRS Diagnostic & Redesign Arc (Phase 6 → 7)

> **Arc timeline**: 2026-04-24 (paper crisis trigger) → 2026-05-04 (Tier 2-min decision)
> **Status**: Phase 7 (Tier 2-min) pending execution
> **Single source of truth** cho diagnostic methodology + mechanism exhaustion + signal redesign decision.

---

## 1. Trigger event — Paper crisis (2026-04-24)

### 1.1 CRS isolation ablation

Bối cảnh: Phase 2c locked D1-O999 (DropAnSH + opacity decay 0.999) là backbone với +0.19 dB.
Cần verify CRS pruning có thực sự contribute trên top of opacity decay, hay redundant.

| Config | AVG PSNR | Δ vs B0 | Note |
|--------|----------|---------|------|
| D1-O999 (CRS pruning ON, opacity decay ON) | 21.13 dB | +0.19 | Phase 2c result |
| **D1-noCRS-O999 (CRS pruning OFF, opacity decay ON)** | **21.21 dB** | **+0.27** | CRS removed → +0.08 dB BETTER |

**Kết luận**: CRS pruning REDUNDANT (or marginally harmful) khi opacity decay đang chạy.

### 1.2 Pattern history ngược

3 prior tests đã warning về CRS-mechanism redundancy:
- **Track B B2** (uniform dropout × (1-CRS)): marginal, không vượt B1β uniform
- **Phase 3α** (CRS-guided anchor sampling): MIXED, AVG +0.075 trong noise
- **Phase 2d Stage A** (density-aware dropout): NEGATIVE, signal structure inadequate

→ Pattern: "CRS weighting selection mechanism → marginal/fail" repeated 4 lần.

### 1.3 Crisis implications

- Paper centered around CRS-as-pruning narrative → CRS không add value → narrative collapse
- Cần evidence định lượng để biết: signal failure? mechanism failure? hay cả 2?
- Cần exhaustive test để defendable claim trong paper

---

## 2. Tier A Diagnostic Suite (Phase 6.1)

### 2.1 Methodology

**Mục tiêu**: Quantitative characterization of CRS signal quality TRƯỚC khi thử thêm mechanism.

4 tests trên 8 LLFF scenes × 4 iter checkpoints (1100, 3000, 5000, 10000):

| Test | Đo gì | Pass criterion |
|------|-------|----------------|
| **A1 BC bimodality** (Sarle's coefficient) | D, R có 2 modes (floater vs surface)? | BC > 0.555 |
| **A2 D-R correlation** | D, R có capture different info? | |corr| < 0.3 |
| **A3 synthetic floater** | Perturb high-CRS Gaussians 0.3×depth_range, đo CRS response | Δ_CRS < -0.1 |
| **A4 occlusion contamination** | Count pairs (Gauss, view) với gauss_z > render_z × 1.05 | Report contamination % |

Implementation: `utils/crs/tier_a_diag.py`

### 2.2 Findings

| Test | Result | Interpretation |
|------|--------|----------------|
| **A1 D bimodality** | BC > 0.555 in 6/8 scenes | ✅ D có 2 modes, signal shape OK |
| **A1 R bimodality** | BC > 0.555 in 5/8 scenes | ✅ R bimodal mostly |
| **A2 D-R correlation** | |corr| < 0.3 in all scenes | ✅ Orthogonal, capture different info |
| **A3 floater discrimination** | Median Δ_CRS = -0.18 after perturbation | ✅ CRS DOES drop on floater |
| **A4 occlusion contamination** | **36.5% R_i samples contaminated** | ❌ R noisy, R component compromised |

### 2.3 Conclusion

**Signal có discriminative power nhưng R contaminated.**
- D component: clean signal
- R component: 36.5% occlusion contamination
- 2 components orthogonal → R isn't fully redundant với D

→ Mechanism failure CANNOT be fully explained by signal failure alone. Cần test mechanism axis trực tiếp.

---

## 3. Six CRS Mechanism Variants — Exhaustive Ceiling Test (Phase 6.2)

### 3.1 Variants tested

| # | Variant | Mechanism | AVG ΔPSNR | Verdict |
|---|---------|-----------|-----------|---------|
| 1 | 6 CRS pruning ablations (tau ∈ {0.20-0.45}) | gate prune | ≤ +0.05 | dead |
| 2 | C1 occlusion-aware R (depth-test) | gate prune | −0.064 | worse |
| 3 | C1.5 depth_range-relative tolerance | gate prune | best −0.038 | dead |
| 4 | F-invisible visibility-streak counter | gate prune | marginal | dead (1.3% catch) |
| 5 | D1G CRS-gated densification | gate densify | +0.028 | REDUNDANT vs DECAY |
| 6 | Stack D1G + DECAY 0.999 | dual gate | +0.082 | sub-additive (synergy −0.132) |
| 7 | hC Hybrid RC × D fusion | gate prune | **−0.079** | worse than B0 |
| 8 | RNRC L3 differentiable α-coupling | continuous opacity multiplier | +0.074 | STACK +0.163 < DECAY +0.186 |

### 3.2 Pattern recognition

**Δ ceiling +0.07 ± 0.10 dB across mọi mechanism dùng D+R signal.**

| Mechanism class | Variants | Result |
|-----------------|----------|--------|
| Gate prune | 6 ablations + C1 + F-inv + hC | all marginal/dead |
| Gate densify | D1G | marginal, redundant |
| Differentiable continuous | RNRC L3 | marginal, sub-DECAY |

→ **Bottleneck là SIGNAL, không phải mechanism class.**

### 3.3 4 Structural Gaps Identified

Through Tier A + 6 mechanism failures, 4 fundamental gaps in current CRS:

| Gap | Description | Evidence |
|-----|-------------|----------|
| **#1** | CRS đo CONSISTENCY, không đo CONTRIBUTION | RNRC L3 với proxy RC fail confirm |
| **#2** | R contaminated 36.5% bởi occluder | Tier A4 quantitative |
| **#3** | Static signal trên dynamic training | EMA snapshot kill late-bloomers |
| **#4** | Post-hoc gate, không enter gradient | Gate variants all fail vs differentiable RNRC L3 marginal |

Plus 2 derived gaps:
- **#5** D phụ thuộc DAV2 (single-view monodepth bias)
- **#6** View aggregation uniform (không pairwise robust)

---

## 4. Stack Synergy Tests

### 4.1 Hướng 2 — D1G + DECAY stack

| Config | AVG | Per-scene synergy verdict |
|--------|-----|---------------------------|
| B0 | 20.797 | — |
| D1G alone | 20.825 (+0.028) | marginal |
| DECAY alone | 20.983 (+0.186) | reference |
| **D1G + DECAY** | **20.879 (+0.082)** | **REDUNDANT** (synergy −0.132) |

→ D1G subsumed by DECAY. Both work on opacity → 1 DOF.

### 4.2 Hướng C — Hybrid CRS

| Config | AVG | Verdict |
|--------|-----|---------|
| B0 | 20.797 | — |
| Hybrid hC alone | 20.718 (−0.079) | worse |
| Hybrid + DECAY | 20.674 (−0.122) | worse than B0! |

→ Proxy RC × D fusion không cứu được. Synergy −0.230 (largest negative observed).

### 4.3 Hướng D — RNRC L3

| Config | AVG | Verdict |
|--------|-----|---------|
| B0 | 20.797 | — |
| FULL (Layer 3 differentiable) | 20.871 (+0.074) | borderline |
| FULL + DECAY | 20.960 (+0.163) | < DECAY alone +0.186 |

Per-scene synergy excluding room outlier: +0.051 (5/7 positive). Mechanism evidence của differentiable axis khác gate (+0.074 vs hC −0.079 với same proxy signal). Nhưng ceiling vẫn không break DECAY +0.186.

---

## 5. Literature Survey & Breakthrough Analysis (Phase 6.3)

### 5.1 SOTA snapshot (LLFF 3-view)

| Method | PSNR | Year | Mechanism |
|--------|------|------|-----------|
| ICO-GS | **22.20** | 2026 | Cycle-depth + feature-MPC virtual view loss |
| BinocularGS | 21.44 | 2024 | Stereo-pair pseudo-views + decay 0.995 |
| CuriGS | 21.10 | 2025 | Curriculum student views + IQA promotion |
| CoR-GS+CoMapGS | 21.105 | 2025 | MASt3R covisibility + per-pixel loss reweight |
| DropAnSH-GS | 20.68 | 2026 | Random anchor + SH dropout |
| **CRSGaussian (current best, no CRS)** | **21.21** | — | Track A1 + B1β + DECAY 0.999 |
| DOC-GS reported | 21.38 | — | |

### 5.2 Architectural axes — saturated vs unexplored

| Axis | Status | Evidence |
|------|--------|----------|
| Opacity decay | ✅ saturated by DECAY +0.186 | DECAY subsumes D1G, RNRC L3 |
| Densification gate | ✅ saturated | D1G fail |
| Pruning gate | ✅ saturated | 6 variants fail |
| **Loss-path** (per-pixel weight) | ❌ **unexplored cho CRS** | **CoMapGS +0.65 dB precedent với weaker signal** |
| **Cross-view feature MPC** | ❌ unexplored cho CRS | ICO-GS +0.7+ dB precedent |
| **SH-degree gating per-Gaussian** | ❌ unexplored | DropAnSH +0.42 random precedent |
| Pseudo-view α attenuation | ⚠️ likely redundant | Same DOF as DECAY |

### 5.3 4 Breakthrough proposals

| Proposal | Mechanism | Signal | Cost | Probability | Top? |
|----------|-----------|--------|------|-------------|------|
| **A: Per-pixel loss reweighter** | Render CRS map → weight L_recon | Existing CRS or upgraded | 1 ngày, no CUDA | ~30-40% | ✅ Recommend |
| B: SH-degree gating | Per-Gaussian sh_max ∝ CRS | CRS | 2 ngày + CUDA | ~20-30% | Risk B1β saturation |
| C: CRS-soft feature-MPC | DINO/VGG MPC weighted by CRS | CRS + foundation | 3-5 ngày, infra heavy | ~25-35% | Re-implements 70% ICO-GS |
| D: Pseudo-view α attenuation | α_eff = α × CRS chỉ ở pseudo views | CRS | 4 giờ | ~15-20% | Likely redundant với DECAY |

→ **Top recommendation**: Proposal A (loss reweighter) — orthogonal axis với DECAY, strong precedent (CoMapGS +0.65), cheap falsification.

---

## 6. Tier 2-min Plan — D_cycle + Loss Reweighter (Phase 7, current)

### 6.1 Dual upgrade rationale

**Single-axis upgrade insufficient**:
- Mechanism A (loss reweighter) với D+R cũ → expected ~+0.05-0.10 (Vanilla A)
- Signal D_cycle với gate prune cũ → cùng cap ceiling vì gate failed for 6 variants
- **Both upgrade simultaneously** → address signal AND mechanism gaps cùng lúc

### 6.2 Signal change: D_cycle replace D_DAV2

```
D_DAV2 (old):
  D_i = 1 - |d_proj_i - d_DAV2_i| / depth_range

D_cycle (new):
  for each pair (a, b) of training views:
    P_a    = project(Gaussian → cam_a)
    d_a    = rendered_depth(P_a)               # internal, không cần DAV2
    P_b    = unproject_then_project(P_a, d_a, cam_a → cam_b)
    d_b    = rendered_depth(P_b)
    P_a'   = project_back(P_b, d_b, cam_b → cam_a)
    cycle_error_pair = ||P_a - P_a'||
  D_cycle_i = exp(-mean_pairs(cycle_error_i) / sigma)         sigma ≈ 5.0 px
```

**Gap addressing**:
| Gap | D_DAV2 | D_cycle |
|-----|--------|---------|
| #5 monodepth bias | ❌ | ✅ no DAV2 |
| #2 occlusion (D side) | ❌ | ✅ cycle break |
| #3 static | ❌ | ✅ rendered depth dynamic |
| #6 view aggregation | ❌ uniform | ✅ pairwise cycle |

Literature precedent: ICO-GS cycle-depth filter +0.7+ dB.

### 6.3 Mechanism change: Loss reweighter replace gate prune

```
Step 1 — Render CRS map (color-swap trick, no CUDA):
  Tạm replace gaussian._features_dc bằng CRS_i (broadcast to RGB)
  Forward render → output là alpha-composited CRS_pix per pixel
  Restore _features_dc

  CRS_pix(p) = Σᵢ Tᵢ(p) · αᵢ(p) · CRSᵢ            (alpha-composite)

Step 2 — Apply weight on reconstruction loss:
  w(p) = clamp(γ + (1-γ) · CRS_pix(p), γ, 1.0)    γ ≈ 0.5
  L_recon = Σ_p w(p) · |I_render(p) - I_GT(p)|

Step 3 — Stop-gradient on CRS_pix to prevent degenerate cycle.
```

**Gap addressing**:
| Gap | Gate prune | Loss reweighter |
|-----|-----------|-----------------|
| #4 non-differentiable | ❌ discrete | ✅ gradient flow qua w(p) |
| #3 kill late-bloomers | ❌ irreversible | ✅ soft, recoverable |

Literature precedent: CoMapGS per-pixel covisibility weight +0.65 dB.

### 6.4 Combined gap coverage

Tier 2-min addresses **4.5/6 gaps**:
| Gap | Tier 2-min |
|-----|-----------|
| #1 contribution measurement | ❌ vẫn consistency-based (cần Tier 1 với foundation models) |
| #2 occlusion contamination | ✅ partial (D_cycle break, R chưa fix) |
| #3 static signal | ✅ D_cycle dynamic + loss reweighter recoverable |
| #4 non-differentiable | ✅ loss reweighter |
| #5 monodepth bias | ✅ D_cycle no DAV2 |
| #6 view aggregation | ✅ pairwise cycle |

So với 1.5/6 với D+R + gate. Hơn 3 gaps.

### 6.5 Compute analysis (per memory rule "always measure compute cost")

| Metric | Baseline | Tier 2-min | Note |
|--------|----------|-----------|------|
| Training time / scene (10k iter) | ~10 phút | ~11 phút (+10%) | D_cycle compute mỗi 100 iters |
| Peak GPU memory | baseline | **+0 GB** | No foundation model |
| Final N_gaussians | ~150-300k | TBD | Report per scene |
| Render FPS (inference) | baseline | **baseline** | CRS không touch inference |
| Model storage (PLY) | baseline | **baseline** | CRS không saved |
| Setup time | 0 | 0 | No model download |

→ Chỉ tăng training time, không động deployment metrics. Paper-friendly.

### 6.6 Ablation matrix

| Tag | use_d_cycle | use_loss_reweight | Mục đích |
|-----|-------------|-------------------|----------|
| B0 | (REUSE) | (REUSE) | baseline |
| DCYCLE | ON | OFF | signal upgrade alone |
| LWEIGHT | OFF | ON | mechanism upgrade alone |
| **TIER2MIN** | **ON** | **ON** | **full Tier 2-min** |

24 NEW runs (3 configs × 8 scenes), ~1.5-2h trên 2 GPU.

**Attribution diagnostic**:
- Δ_DC alone → signal contribution
- Δ_LW alone → mechanism contribution
- Synergy = Δ_T2M − (Δ_DC + Δ_LW) → bổ trợ vs redundant

### 6.7 Commitment criteria (BINARY decision)

| Verdict | Δ_T2M | Action |
|---------|-------|--------|
| 🟢🟢 BREAKTHROUGH | ≥ +0.20 | Defendable contribution. Upgrade Tier 1 (DINOv2 + MASt3R) hoặc consolidate paper với Tier 2-min |
| 🟢 Solid | +0.10 ~ +0.20 | Ship hoặc Tier 1 (user chọn) |
| 🟡 Marginal | 0 ~ +0.10 | Pivot recipe paper (consolidate Tier1 best 21.21 + Tier A diagnostic methodology) |
| 🔴 Dead | ≤ 0 | Pivot decisively. **KHÔNG đề xuất variant #8** |

---

## 7. Pivot Plan (if Tier 2-min fails)

### 7.1 Recipe + Methodology paper outline

**Title tentative**: *"Diagnosing and Bypassing Reliability Bottlenecks in Sparse-View 3D Gaussian Splatting"*

**Contributions**:
1. **Tier A Diagnostic Suite** — 4 quantitative tests cho reliability signal (BC bimodality, A2 orthogonality, A3 synthetic floater, A4 occlusion contamination). Novel cho 3DGS field.
2. **Empirical Analysis: 4 Structural Gaps of Consistency-Based CRS** — systematic failure characterization across 7 mechanism variants.
3. **Working Recipe (Without CRS)** — Track A1 (sh_degree=1 + freeze_sh) + B1β (uniform dropout 0.2) + DECAY 0.999 = **21.21 dB AVG** (close to DOC-GS 21.38).

**Negative results documentation**:
- 6 CRS mechanism variants fail
- CRS-as-pruning fundamentally redundant với opacity regularization
- D+R consistency signal có ceiling +0.07 dB
- Lessons for future reliability-aware sparse-view 3DGS work

### 7.2 Compute baseline competitive

So với SOTA:
- Training: ~10 phút / scene 10k iter (vs 30+ min cho 30k baseline)
- GPU memory: ~5 GB (consumer GPU friendly)
- Render FPS: baseline 3DGS speed
- Model storage: baseline PLY size

### 7.3 Defensibility

- 6 architectural variants × 8 scenes = 48 negative data points
- 4 quantitative diagnostics × 8 scenes = 32 diagnostic data points
- Hierarchical decomposition (signal vs mechanism vs combined) shows systematic exhaustion
- Reviewer counter "did you try X?" → answer with diagnostic evidence

---

## 8. References

- **Decision log**: docs/04_decisions_log.md (chronological)
- **Task queue**: docs/03_task_queue.md (Phase 6, Phase 7)
- **Track A/B history**: docs/10_track_a_b_results.md
- **Tier A implementation**: utils/crs/tier_a_diag.py
- **Memory entries**:
  - `feedback_measure_compute_cost.md` — compute reporting requirement
  - `project_sh_reliability_crs_idea.md` — Phase 4 SH reliability (deferred)
  - `project_budget_scaling_rule.md` — opacity decay budget-scaling

---

*Last updated: 2026-05-08. Phase 9 execution pending.*

---

## 7. Phase 7 Tier 2-min RESULTS

### 7.1 Phase 7 Original — Phase 5 weak backbone (B0 = 20.234)

Implementation: `utils/crs/d_cycle.py` (cycle-depth helpers), `render_crs_map` qua color-swap
trick (override_color path), train.py loss reweighter block. Python only, no CUDA.

**Result (8 LLFF scenes):**

| Config | AVG | Δ vs B0 | Note |
|--------|-----|---------|------|
| B0 | 20.234 | 0 | reference |
| **DCYCLE alone** | **20.359** | **+0.125** ⭐ | **FIRST CRS variant với clean positive across 9 attempts** |
| LWEIGHT alone | 20.228 | −0.007 | mechanism neutral |
| TIER2MIN combined | 20.269 | +0.035 | synergy −0.083 (redundant) |

**Compute:** +0.1-0.2% slowdown across configs, GPU memory unchanged.

**Key findings:**
1. D_cycle là first signal upgrade clean positive — beats 8 prior CRS variants ceiling +0.07
2. Loss reweighter mechanism alone neutral
3. Combined TIER2MIN destructive — synergy −0.083
4. Per-scene heterogeneity high: D_cycle wins floater-prone (trex +0.345, horns +0.327),
   fails on detail/photometric (room −0.067 with LW catastrophic −0.699)

**Verdict:** 🔴 STOP per +0.15 commitment threshold trên backbone này. Nhưng **D_cycle is real signal**.

### 7.2 Phase 7 Stage 1 — D1-O999 strong backbone

Goal: verify D_cycle scale từ weak backbone (Phase 5) sang strong backbone (D1-O999 = current best).

**3-config attribution (ablation control raised by execution session):**

| Config | AVG | Δ vs DAV2_GATE | Note |
|--------|-----|----------------|------|
| TIER1_DAV2_GATE | 21.178 | 0 | reference (matches Phase 6 D1-O999 ~21.13 + run variance) |
| TIER1_DC_GATE | 21.163 | **−0.015** | **D_cycle FLIPS NEGATIVE on strong backbone** |
| TIER1_DC_LW | 21.228 | +0.050 | Combined slightly positive |
| vs No-CRS (21.21) | — | +0.018 | within noise |

**Critical finding — D_cycle effect FLIPS sign across backbones:**
- Phase 5 weak: +0.125 (positive)
- D1-O999 strong: −0.015 (negative)

**Hypothesis:** D_cycle fills regularization gap on weak backbone, REDUNDANT với DropAnSH+DECAY
trên strong backbone. **R contamination 36.5% may be diluting D_cycle gain trên strong backbone too.**

**Per-scene pattern:** D_cycle helps floater-prone (horns +0.106, room +0.139), hurts detail
(fern −0.070, leaves −0.138).

**Compute:** DC_GATE +2.7%, DC_LW +4.8% slowdown — reasonable.

**Verdict:** 🟡 H_both_dead — signal + mechanism saturated với D+R formula intact.
**→ Phase 8 hypothesis: fix R first, then add S signal.**

---

## 8. Phase 8 — Formula Redesign + SH Path 🎉 BREAKTHROUGH

### 8.1 Motivation

Phase 7 Stage 1 raised concern: **stack S on broken D+R risky** (R contamination 36.5%
propagates noise). Must fix R formula trước.

3 sub-phases simultaneously với 5-config attribution control:
- **8a: R_visible** (visibility-aware reprojection, fix occlusion contamination)
- **8b: S_stability** (SH coefficient EMA variance, multi-dim CRS adding color path)
- **8c: CRS-modulated SH freeze** (per-Gaussian targeted freeze replace global Track A1)

### 8.2 Implementation

| File | Action |
|------|--------|
| `utils/crs/sh_stability.py` (NEW) | EMA variance tracking per-Gaussian |
| `utils/crs/sh_freeze.py` (NEW) | Per-Gaussian gradient zeroing on _features_rest |
| `utils/crs/crs_module.py` (modify) | R_visible logic + multi-component CRS formula với auto-norm |
| `train.py` (modify) | Hooks for S update + CRS-mod SH freeze (after backward, before optimizer.step) |
| `scripts/p8_master.sh` (NEW) | 5-config ablation runner |
| `scripts/p8_analyze.py` (NEW) | Attribution analyzer |

**Formula upgrade:**
```
CRS_new = sigmoid(scale × (w_d·D_cycle + w_r·R_visible + w_s·S_stability − threshold))
```

3 components address 3 known gaps simultaneously.

### 8.3 Result (5 configs × 8 scenes = 40 runs)

| Config | Components changed | AVG | Δ vs OLD |
|--------|-------------------|-----|----------|
| OLD | reference | 21.178 | 0 |
| FIX_R_DAV2 | + R_visible (R fix only) | 21.068 | **−0.111** ❌ R alone HURTS |
| FIX_R_DC | + D_cycle (with clean R) | 21.169 | +0.102 ✅ D vindicated |
| FIX_RS | + S_stability | 21.146 | −0.023 ⚪ S adds nothing |
| **FULL** | + CRS-mod SH freeze | **21.335** | **+0.156** |

**vs No-CRS reference (21.21):** FULL = **+0.125 dB** 🎉 (first CRS variant beat no-CRS by meaningful margin)

**Per-scene FULL vs OLD (5/8 wins):**
- Big wins: room +0.544, horns +0.361, fortress +0.344
- Small wins: flower +0.103, leaves +0.019
- Marginal loss: fern −0.075, orchids −0.040, trex −0.005

**Compute:** FULL +3.7% slowdown, GPU memory unchanged, render FPS unchanged.

### 8.4 Attribution insights

| Component | Δ alone | Verdict |
|-----------|---------|---------|
| Δ_R (R_visible) | **−0.111** | ❌ Surprise: R alone hurts! Visibility filter aggressive → data loss > noise reduction |
| Δ_D (D_cycle in clean R) | **+0.102** | ✅ Vindicated: works when R cleaned (Stage 1 R noise was diluting) |
| Δ_S (S_stability) | −0.023 | ⚪ EMA variance không add useful info |
| **Δ_M (CRS-mod SH freeze)** | **+0.189** | ✅ **BIGGEST WINNER** — per-Gaussian targeted > global freeze |

### 8.5 KEY MILESTONE

🎉 **Phase 8 = first CRS variant beat no-CRS recipe by meaningful margin** sau 9 prior CRS attempts
ceiling at +0.07 dB.

**Reference targets check:**
- Phase 8 FULL: 21.335
- DOC-GS reported: 21.380 (gap **−0.045**, within noise)
- BinocularGS: 21.440 (gap −0.105)
- ICO-GS SOTA: 22.200 (gap −0.865)

→ Phase 8 close DOC-GS, still cần lever khác để break SOTA.

### 8.6 Hypotheses cho Phase 9

1. **H1**: Drop R hoàn toàn (D-only formula) → cleaner signal, không R noise
2. **H2**: Drop S → simplify recipe, không lose performance
3. **H3**: CRS-mod SH freeze universal mechanism → test on A1+B1β backbone (replace Track A1 global freeze)

---

## 9. Phase 9 — Simplification + Cross-backbone (DONE)

### 9.1 Test 1 — D1-O999 simplification

| Tag | Components | Hypothesis |
|-----|-----------|------------|
| FULL (reuse Phase 8) | D + R + S + CRS-mod-freeze | reference 21.335 |
| FULL_NoS | D + R + CRS-mod-freeze | H2: drop S |
| D_ONLY_FREEZE | D + CRS-mod-freeze (no R, no S) | H1: drop R |
| D_ONLY_GATE | D + CRS prune (no R, no S, no SH freeze) | isolate D signal alone |

### 9.2 Test 2 — A1+B1β cross-backbone

| Tag | Backbone | Components | Hypothesis |
|-----|----------|-----------|------------|
| A1B1_BASELINE | A1+B1β | (no Phase 8 additions) | reference (~20.96) |
| A1B1_BEST | A1+B1β (NO global freeze) | + R_visible + D_cycle + CRS-mod freeze | H3: SH freeze universal |

### 9.3 Implementation needs

- `--disable_r_signal` flag: skip R compute, w_d=1 in formula
- `--disable_global_sh_freeze` flag: ignore Track A1 freeze when CRS-mod active

### 9.4 Cost & verdict tree

**Cost:** 32 NEW runs (~100 phút 2 GPU). FULL reuse từ Phase 8 logs/p8/.

| Outcome | Action |
|---------|--------|
| H1 confirmed (drop R win ≥ +0.10) | Simplify recipe to D-only |
| H2 confirmed (drop S neutral ±0.05) | Drop S, cleaner formula |
| H3 confirmed (A1+B1β cross-backbone +0.15) | SH freeze universal mechanism |
| Best variant > 21.45 | Close BinocularGS, push toward SOTA |

### 9.5 Phase 9 RESULTS

**Test 1 — D1-O999 simplification (8 LLFF scenes):**

| Config | AVG | Δ vs FULL | Note |
|--------|-----|-----------|------|
| **FULL (Phase 8 ref)** | **21.335** | 0 | **BEST — locked** |
| D_ONLY_GATE | 21.242 | −0.093 | drop SH freeze hurts |
| FULL_NoS | 21.200 | −0.135 | drop S hurts MORE than expected |
| D_ONLY_FREEZE | 21.159 | −0.176 | drop both R+S worst |

**Test 2 — A1+B1β cross-backbone:**

| Config | AVG | Δ |
|--------|-----|---|
| A1B1_BASELINE | 20.932 | reference (~Track A+B 20.96 ✓) |
| A1B1_BEST | 20.983 | +0.051 |

### 9.6 Verdict — All hypotheses non-trivial

| H | Status |
|---|--------|
| H1 (drop R helps) | ❌ REJECTED — R contributes |
| H2 (drop S neutral) | ❌ REJECTED — S synergize with mechanism |
| H3 (SH freeze universal) | 🟡 PARTIAL — works on A1+B1β but smaller (+0.051 vs +0.189 on D1-O999) |

### 9.7 KEY INSIGHT — Sequential vs Leave-one-out attribution

| Component | Phase 8 sequential Δ | Phase 9 leave-one-out Δ |
|-----------|-----------------------|--------------------------|
| R | −0.111 (alone hurts) | −0.041 (synergize when combined) |
| **S** | **−0.023 (looks neutral)** | **−0.135 (synergize strongly!)** |

→ **Sequential delta misleading**. Components có non-trivial synergy. **Combination > sum of parts.**

→ All 4 components (D_cycle + R_visible + S_stability + CRS-mod SH freeze) phải present trong FULL recipe.

### 9.8 Compute insight

| Config | AVG train (s) | Backbone |
|--------|---------------|----------|
| A1B1_BEST | 192.0 ⚡ | A1+B1β (no DropAnSH overhead) |
| A1B1_BASELINE | 205.2 | A1+B1β |
| D1-O999 FULL | 360.7 | D1-O999 (DropAnSH overhead) |

A1+B1β backbone ~2× faster nhưng PSNR thấp 0.35 dB. **D1-O999 đáng overhead cho gain.**

### 9.9 Phase 10 plan — SOTA gap closure (CRS axis exhausted)

**State:** Phase 8 FULL = 21.335 dB locked. CRS axis maxed out.
**Gap to SOTA:** ICO-GS 22.20 → still −0.865 dB.

Cần orthogonal lever:

| Lever | Expected | Cost | Note |
|-------|----------|------|------|
| **10A: Dense init (DUSt3R/MASt3R)** | +1.0-3.0 dB | 1-2 ngày | Biggest lever, init-time |
| 10B: Feature MPC (DINO loss) | +0.3-0.7 dB | 2-3 ngày | ICO-GS path, training-time |
| 10AB: Stack | combined +1.5-3.5 dB | 3-4 ngày | Max ambition |

**Trade-off**: dense init/MPC orthogonal với CRS — paper claim "CRS contribution" loãng nếu lever này dominate.
