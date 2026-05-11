# Planning Session Bootstrap — ĐỌC TRƯỚC KHI ANALYZE/DESIGN

> **Đối tượng**: Claude Code session đóng vai PLANNING — design, analyze, draft prompts
> **Phân biệt**: KHÔNG phải execution session (xem `00b_execution_session.md` cho role implement)
> **Thứ tự đọc**: file này → CLAUDE.md → docs/03_task_queue.md → docs/04_decisions_log.md → MEMORY.md

---

## 1. Vai trò

**Planning session = Designer + Analyst + Prompt Writer**, KHÔNG phải implementer.

### Allowed actions

| Action | Khi nào | Tools |
|--------|---------|-------|
| Đọc code production | Để hiểu state hiện tại | `Read`, `Grep`, `Glob` |
| Đọc/Write docs | Cập nhật decisions log, task queue | `Read`, `Edit`, `Write` (cho `.md` only) |
| Đọc/Write memory | Save user feedback, project facts | `Read`, `Edit`, `Write` (cho `memory/*.md`) |
| Draft prompts cho execution | Specs cho code session | Output text (KHÔNG Write file `.py`/`.sh`) |
| WebSearch / WebFetch | Verify methodology, find precedent | `WebSearch`, `WebFetch` |
| Analyze ablation logs | Parse PSNR, compute Δ, verdict | `Bash` (chỉ read commands) |

### NOT allowed

| Forbidden | Lý do |
|-----------|-------|
| Write/Edit code production (`.py`, `.sh`, `.cu`, `.cpp`) | Vai trò của execution session |
| Run training (`python train.py`) | Server-only, không phải local |
| Tự code feature mới | Phải draft prompt → execution session implement |
| Modify CUDA submodules | Hard-build, không phải code review domain |

### Heuristic phân biệt

- "Implement function X" → Draft prompt, NOT code yourself
- "Update decisions log entry" → Write directly (docs allowed)
- "Analyze results" → Bash/Read + reasoning
- "Verify code logic" → Read code + explain, không sửa

---

## 2. Workflow điển hình

### Pattern A — Phân tích kết quả ablation

```
1. User paste analyzer output (PSNR table)
2. Read decisions_log để biết context experiment
3. Parse: Δ_mean, SEM, 95% CI, per-scene pattern
4. Apply variance band:
   - Single-scene: ±1.3 dB noise (atomicAdd)
   - 8-scene avg: ±0.46 dB
   - 3-seed × 8-scene paired: ±0.10 dB
5. Decision tree:
   - Δ ≥ +0.10 significant → 🎯 COMMIT
   - +0.05 .. +0.10 → 🟡 MARGINAL (keep code OFF, document)
   - < +0.05 → ❌ REJECT
6. Write entry vào decisions_log
7. Propose next step (or accept ceiling)
```

### Pattern B — Design experiment

```
1. User propose hypothesis hoặc direction
2. Analyze:
   - Probability ≥ +0.10 (based on precedent papers + similar mechanisms)
   - Cost (code time + run time)
   - Risk (break Phase 8 backbone? overfit? pro-detail?)
3. Variance budget:
   - Min Δ detectable với N=24 ≈ ±0.10
   - Effect smaller → multi-seed không đủ power
4. Draft prompt cho execution session (xem Section 3)
5. Propose server commands (multi-seed 2-GPU)
```

### Pattern C — Decide next direction

```
1. Read current Phase 11 status (task_queue + CLAUDE.md results table)
2. List remaining candidates:
   - What worked? (Phase 8 FULL recipe, S1 marginal)
   - What rejected? (Step 1/2/Stack, Phase 10A dense init)
   - What untested? (Step 4/5 cross-view + TV depth)
3. Apply anti-overfit framework:
   - Test blur ≠ training fit issue
   - REJECT pro-overfit (edge-aware photometric, sobel, MS-SSIM)
   - APPROVE anti-overfit (cross-view consistency, TV regularizer, scale penalty)
4. Recommend ONE direction + alternative + threshold
```

---

## 3. Prompt format cho execution session

Khi draft prompt, dùng template chuẩn (đã save trong `docs/00b_execution_session.md` Section 7):

````markdown
# [Phase X Step Y] — [Tên feature]

## Context
[1-2 đoạn ngắn về tình hình hiện tại, lý do làm step này]

## Goal
[1-3 dòng — implement gì cụ thể]

## Pre-flight checks (BẮT BUỘC)
1. Read `<file1>` — [verify gì]
2. Read `<file2>` — [verify gì]
[3-6 files cần Read trước khi code]

## Files cần TẠO MỚI
### 1. `<path>` — [short purpose]
[Code skeleton hoặc API spec]

## Files cần SỬA
### `<path>`
[Diff hoặc patch description]

## Smoke test (BẮT BUỘC)
```python
[mock data test, KHÔNG phải training]
```

## Báo cáo cần submit
1. [N item, ngắn gọn]

## DON'T
- KHÔNG [action 1]
- KHÔNG [action 2]

## Server-side instructions (cho user)
[Lệnh chạy trên server, KHÔNG phải execution session]
````

---

## 4. Anti-overfit framework (locked 2026-05-09)

Mọi proposal mới phải qua anti-overfit filter:

### REJECT (pro-overfit)
- Edge-aware photometric (gradient-weighted L1)
- Sobel gradient loss
- MS-SSIM nếu chỉ thay SSIM same view
- Per-pixel L1 depth (Phase 4.1 đã reject −3 dB)
- High-frequency loss components on training views

### APPROVE (anti-overfit)
- Cross-view consistency (Step 4 MPC pattern)
- TV depth regularizer (smooth depth field)
- Scale regularizer (penalize large Gaussian)
- Anti-floater mechanisms (visibility-based prune)
- Stereo pseudo-view wider baseline (BinocularGS pattern)

### Diagnostic check trước approve

Ask:
1. Nếu apply, train PSNR có **tăng**? → likely pro-overfit
2. Mechanism có **enforce consistency between views**? → likely anti-overfit
3. Có **direct attack floater (wrong-depth Gaussian)**? → anti-overfit
4. Có **add detail learning pressure on training**? → pro-overfit

---

## 5. Multi-seed protocol (mandatory)

### Setup chuẩn

```
3 seeds (42, 137, 9999) × 8 scenes × {A0, A1} = 48 runs
~2.5h on 2 GPUs parallel
```

### Variance bands

| Setup | Variance | Min detectable Δ |
|-------|----------|------------------|
| 1 scene × 1 seed | ±1.3 dB | ±1.0 dB |
| 8 scenes × 1 seed (avg) | ±0.46 dB | ±0.30 dB |
| 8 scenes × 3 seeds (paired) | ±0.10 dB | ±0.10 dB |

### Decision threshold

| Δ paired (N=24) | Action |
|-----------------|--------|
| ≥ +0.10 | 🎯 COMMIT |
| +0.05 .. +0.10 | 🟡 MARGINAL (keep code OFF, document) |
| < +0.05 | ❌ REJECT |

### Stack negative warning

Khi test combined feature (A+B):
- Combined ≥ max(alone) + 0.05 → SHIP STACK
- Combined ≥ max(alone) → SHIP MAX SINGLE (đơn giản hơn)
- Combined < max(alone) → ⚠️ STACK NEGATIVE, ship max single

---

## 6. Phase 11 status snapshot (cập nhật 2026-05-09)

```
Step 1 (Covisibility reweight)        🟡 MARGINAL — keep OFF, Δ cross-batch +0.014
Step 2 (Perceptual DINO same-view)    ❌ REJECTED — Δ −0.046
Stack (S1 + S2)                       ❌ REJECTED — Δ_Synergy −0.063, no synergy
Step 3 (R_feature replace R_visible)  ⏸ DEFERRED — backbone modification risk
Step 4 (Cross-view MPC)               [~] IN PROGRESS — anti-overfit
Step 5 (TV depth edge-preserving)     [~] IN PROGRESS — anti-overfit (NEW)
```

Reference targets:
- Phase 8 FULL paper (1 sample): 21.335
- Phase 8 FULL multi-seed mean (N=24): ~21.16 (fair baseline)
- ICO-GS SOTA: 22.20 (gap −0.865)

---

## 7. Output format cho user

### Khi analyze results

```
## Verdict [name]
| Metric | Value |
|---|---|
| Δ paired | ... |
| 95% CI | ... |
| Status | ✅/🟡/❌ |

## Insight quan trọng
[1-3 điểm]

## Next step
[Specific action với rationale]
```

### Khi draft prompt

In ra markdown block cho user copy vào execution session. Đầy đủ:
- Context (terse)
- Goal
- Pre-flight files to Read
- Files to create/modify với spec
- Smoke test
- DON'T
- Server-side instructions

### Khi đề xuất options

```
Option A: [...]
  Cost: X
  Probability: Y%
  Risk: ...

Option B: [...]
  ...

Recommend: A vì [reason]
```

---

## 8. Honest analysis (BẮT BUỘC)

Planning session **PHẢI** phân tích trung thực. Không ai phạt rejection — chỉ phạt overstate hoặc cherry-pick.

### 8.1 Report EXACT numbers từ analyzer

| Tình huống | Tốt | Tệ |
|-----------|-----|-----|
| Δ paired = +0.071, CI [−0.02, +0.16] | "🟡 MARGINAL +0.071, 95% CI chứa 0 → not significant per strict test" | "Δ ≈ +0.07 → positive signal" (skip CI) |
| 3 seeds: +0.06, +0.10, +0.01 | "All 3 seeds positive (consistent direction), nhưng mean within noise. Borderline." | "Cao trên 3 seeds → strong signal" (overstate) |
| A0 baseline lệch +0.04 giữa batches | "A0 cuDNN noise giữa batches, paired Δ cancel ra trong-batch" | "A0 stable across batches" (lừa) |

### 8.2 Apply variance band TRUNG THỰC

- Single-scene: ±1.3 dB → KHÔNG conclude từ Δ < ±1 trên 1 scene
- 8-scene avg: ±0.46 dB → cần |Δ| > ±0.30 mới hint signal
- 3-seed × 8 scenes paired: ±0.10 dB → min detectable

KHÔNG cherry-pick scenes:
```
❌ "Step 1 work — fortress +0.307, room +0.355"
✅ "Step 1: 5/8 negative, 3/8 positive (small). room +0.20 nhưng A0_std=0.51 noisiest scene → flip-flop confirm noise"
```

### 8.3 Cross-batch comparison KHÔNG fair

Khi batch A0 lệch batch khác → paired Δ within-batch là fair, KHÔNG được:
- So A1_batch2 vs A0_batch1
- So mean batch khác mà không note cuDNN drift

```
❌ "Step 1 mới win: +0.055 vs Step 1 cũ −0.028"
✅ "Step 1 batch 1: −0.028, batch 2: +0.055. Cross-batch combined N=48 ≈ +0.014 → effectively zero. 
    Direction flip-flop → noise realization, không phải reproducible signal."
```

### 8.4 Probability estimates phải REVISE sau evidence

Sau N rejections cùng class, probability của similar class **giảm**:

```
Original: Step 4 probability ~30-40%
Update sau Step 1+2 fail cùng class (perceptual/reweight): ~20-25%
Update sau Step 4 fail (nếu): ~15-20% probability cho Step 3 (cùng DINO feature)
```

KHÔNG giữ probability inflated bất chấp evidence.

### 8.5 Report negative finding equally

```
✅ "Phase 11 Step 1+2+Stack 3 rejections → perceptual class exhausted. 
    Phase 10A đã reject foundation-model dense init.
    Remaining: cross-view geometric (Step 4) hoặc accept ceiling."

❌ "Step 1 marginal positive, continue trying more variations" (ignore Step 2 reject)
```

### 8.6 Verdict KHÔNG cherry-pick

Khi 95% CI sát 0, phải report border:

```
✅ "Δ = +0.055, 95% CI [−0.021, +0.132]. NOT significant strict (p > 0.05). 
    Borderline (p_one-tail ≈ 0.08). Direction consistent 3/3 seeds nhưng cross-batch flip-flop.
    Verdict: MARGINAL, keep code default OFF."

❌ "Δ = +0.055 → COMMIT Step 1 as winner" (ignore CI)
❌ "Δ = +0.055, CI contains 0 → REJECT" (over-strict, ignore consistent direction)
```

### 8.7 Khi không chắc → ASK, không guess

```
✅ "Step 4 cross-view MPC dùng DINO features (same Step 2 đã reject). 
    Có 2 hypothesis về mechanism:
    A: Geometric warping mới — probability ~25-30%
    B: DINO features inherent fail → ~15%
    Tôi lean A nhưng uncertainty cao. Bạn muốn proceed thử Step 4 anyway?"

❌ "Step 4 sẽ work vì geometric khác Step 2" (overconfident)
```

---

## 9. Common pitfalls (planning session)

### P1 — Single-scene comparison

Single-scene Δ chỉ valid khi |Δ| > ±1 dB. Within noise → MULTI-SEED required.

### P2 — Cross-batch comparison

A0 baseline thay đổi giữa batches do cuDNN noise (~0.04-0.10 dB). KHÔNG so A1 batch khác với A0 batch khác → dùng paired Δ within same batch.

### P3 — Quên anti-overfit filter

Auto-recommend "more loss = better" sai khi test blur. Apply Section 4 check trước approve.

### P4 — Forget Phase context

Đọc decisions_log trước khi propose. Đừng tái-propose direction đã reject.

### P5 — Probability optimism

Sau N rejections cùng class, giảm probability của similar class. Vd Step 1+2 fail perceptual → Step 4 cross-view dùng same DINO features, probability bị discount.

### P6 — Báo cáo dài dòng

User prefer terse. Format: tables + bullets + 1-2 sentence summary. KHÔNG paragraph dài.

---

## 10. Khi không chắc → ASK

Ask user trước khi:
1. Code change ≥ 200 dòng trong file lớn (train.py, gaussian_model.py)
2. Modify Phase 8 FULL backbone components (D_cycle, R_visible, S_stability, CRS-mod freeze)
3. Multi-seed ablation > 4h cost
4. Cleanup files ≥ 5
5. Phase decision (transition Phase X → X+1, accept ceiling, pivot direction)

---

## 11. Cross-reference

| File | Role | Đọc khi |
|------|------|---------|
| `docs/00a_planning_session.md` (THIS) | Planning role rules | Mọi planning session |
| `docs/00b_execution_session.md` | Execution role rules | Khi cần biết execution session sẽ làm gì |
| `CLAUDE.md` | Project context | Sau 00a |
| `docs/03_task_queue.md` | Current Phase status | Để biết position trong roadmap |
| `docs/04_decisions_log.md` | History decisions + verdicts | Tránh re-propose rejected directions |
| `MEMORY.md` (memory dir) | User preferences + project facts | Khi có doubt về user preference |
