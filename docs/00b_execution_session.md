# Execution Session Bootstrap — ĐỌC TRƯỚC KHI CODE

> **Đối tượng**: Claude Code session đóng vai EXECUTION — implement, edit, smoke test
> **Phân biệt**: KHÔNG phải planning session (xem `00a_planning_session.md`)
> **Thứ tự đọc**: file này → CLAUDE.md → docs/03_task_queue.md (task hiện tại)

---

## 1. Vai trò

**Execution session = Implementer**. Nhận prompt từ planning session → code → smoke test → report.

### Allowed actions

| Action | Tools |
|--------|-------|
| Read code production | `Read`, `Grep`, `Glob` |
| Write/Edit code production | `Edit`, `Write` (`.py`, `.sh`, `.cu`, `.cpp`) |
| Smoke test (mock data) | `Bash` (limited to verify imports + shape) |
| Tạo scripts | `Write` (`.sh`, `.py` trong `scripts/`) |

### NOT allowed

| Forbidden | Lý do |
|-----------|-------|
| Run training (`python train.py ...`) | Server-only, không local |
| Design new features | Vai trò của planning session |
| Modify decisions log | Vai trò của planning session (allowed nhưng nên skip) |
| Auto-cleanup khi không được prompt | Rule 13 yêu cầu user approve |

---

## 2. Workflow điển hình

### Khi nhận prompt từ planning session

```
1. Pre-flight: Read 3-6 files được chỉ định
   → Verify API existing, không đoán
2. Báo lại 1 dòng confirm pre-flight (camera convention, function signature, etc.)
3. Implement:
   - Tạo file mới với marker [CRSGaussian Phase X Step Y]
   - Edit file cũ với gating block (default OFF)
4. Smoke test mock data (verify import + shape + grad flow)
5. Báo cáo theo format prompt yêu cầu
```

---

## 3. Coding rules (BẮT BUỘC)

### Rule 1 — Marker `[CRSGaussian Phase X]`

Mọi function/section thêm/sửa **PHẢI** có comment marker để grep ngược tìm được:

```python
# ============================================================
# [CRSGaussian Phase 11 Step 5] Task: TV depth regularizer
# File: utils/regularizer/tv_depth.py
# Mục đích: edge-preserving TV smooth depth field
# ============================================================

# Inline cũng được:
# ── [CRSGaussian Phase 11 Step 5] TV depth hook ──
```

→ `grep -r "[CRSGaussian Phase 11]"` phải lấy được mọi đoạn code Phase 11.

### Rule 2 — Master switch pattern (default OFF)

Mọi feature mới:

1. **Flag** trong `arguments/__init__.py` → `self.use_<feature> = False`
2. **Gating** trong code → `if dataset.use_<feature> and <conditions>:`
3. **Default OFF** đảm bảo behavior trước-feature unchanged
4. **Verify**: chạy 1 config without flag → metric phải match log baseline

### Rule 3 — Tách file (KHÔNG nhồi vào train.py)

- Logic mới > 30 dòng → tạo file riêng tại `utils/<dir>/<feature>.py`
- `train.py` chỉ chứa: `import + gate block + 1-3 dòng gọi` (interface clean)
- Subdir: `utils/depth/`, `utils/crs/`, `utils/loss/`, `utils/init/`, `utils/regularizer/`, `utils/feature/`

### Rule 4 — Read trước khi edit, KHÔNG đoán API

- LUÔN `Read` file gốc trước khi sửa
- Khi tích hợp library ngoài: `Read` source code library
- Nếu không tìm thấy: dùng `Grep` / `Glob` chứ KHÔNG fabricate

### Rule 5 — KHÔNG sửa BASE folders chỉ-đọc

**CHỈ ĐỌC**:
- `FSGS/`, `DNGaussian/`, `LoopSparseGS/`, `DepthRegularizedGS/`, `SCGaussian/`
- `external/dust3r/` (foundation models)
- `submodules/diff-gaussian-rasterization-confidence/`, `submodules/simple-knn/` (CUDA)

**Sửa được**: `CoR-GS/` root files, `utils/`, `scene/`, `arguments/`, `gaussian_renderer/`, `scripts/`, `tests/`.

### Rule 6 — KHÔNG run training để verify

- Smoke test chỉ: mock data, no GPU training
- Import test: `python -c "from utils.x import f; print('OK')"`
- Verify default OFF: explain bằng lời, không cần chạy

### Rule 7 — Comment style

- Header block đầu function/class mới (Việc 1)
- Inline comment tại logic không-hiển-nhiên (Việc 2)
- KHÔNG comment code self-explanatory
- Comment explain **WHY**, không phải WHAT

---

## 4. Verify-before-done checklist

Trước khi báo "done" cho planning session:

- [ ] Mọi file đã tạo có header marker `[CRSGaussian Phase X]`
- [ ] Master switch flag default = `False`
- [ ] Logic nằm trong file riêng, không nhồi vào train.py
- [ ] Smoke test (numpy/torch mock) PASS
- [ ] Verify default OFF: explain bằng lời tại sao behavior unchanged khi flag tắt
- [ ] List rõ file đã tạo/sửa cho planning verify
- [ ] Cleanup imports không dùng
- [ ] KHÔNG chạy training command nào

---

## 5. Smoke test patterns

### Pattern 1 — Verify import + grad flow

```bash
python -c "
import torch, sys
sys.path.insert(0, '.')
from utils.<module> import <function>

# Mock inputs
input_tensor = torch.rand(1, 3, 200, 200, device='cuda', requires_grad=True)
output = <function>(input_tensor)
output.backward() if hasattr(output, 'backward') else None
print(f'Output shape: {output.shape}')
print(f'Grad flow: {input_tensor.grad is not None}')
"
```

### Pattern 2 — Verify edge-case behavior

```bash
python -c "
from utils.<module> import <function>
import torch

# Edge case 1: empty input
# Edge case 2: extreme values
# Edge case 3: synthetic ground truth (verify mechanism)
"
```

### Pattern 3 — Verify default OFF

KHÔNG chạy training, explain bằng lời:

```
Default OFF behavior:
- `dataset.use_<feature> = False` → if block skip
- Imports lazy → không load module khi flag OFF
- → Identical với baseline (Phase 8 FULL)
```

---

## 6. Common pitfalls (execution session)

### P1 — Hardcode path

❌ `dust3r_path = "/home/aidev/workspace/.../external/dust3r"`
✅ `dust3r_path = os.path.join(os.path.dirname(__file__), "..", "external", "dust3r")`
✅ Hoặc CLI arg `--dust3r_path`

### P2 — Không thread args xuống dataset_readers

Khi feature cần đọc flag từ `args`:
- Sửa signature `readColmapSceneInfo(..., args=None)`
- Sửa `scene/__init__.py` truyền `args=args` vào `sceneLoadTypeCallbacks`
- Gate bằng `if args is not None and getattr(args, "<flag>", False):`

### P3 — Quên LLFF train/test split logic

LLFF eval: `train_idx = idx % llffhold != 0` (llffhold=8 default)
n_views=3 subsample: `np.linspace(0, len(train)-1, 3).round()`

### P4 — COLMAP coordinate convention

- COLMAP `images.bin` lưu **world→cam**
- Many libraries (DUSt3R, NeRF) dùng **cam→world**
- Convert: `C2W = inv([[R_w2c, T_w2c], [0,1]])`

### P5 — Quên dependencies có version constraint

- xformers force-upgrade torch 2.1.0 → 2.4.1 (đã gặp 2026-05-08)
- Luôn `pip install --dry-run <package>` để verify trước
- Phải acceptable: chỉ install new package, KHÔNG upgrade existing core deps

### P6 — Mismatch package name

- `submodules/diff-gaussian-rasterization-confidence/` (folder)
- Import: `from diff_gaussian_rasterization import GaussianRasterizer` (package name)
- Folder name ≠ Python module name → check `setup.py` cho `name=...`

### P7 — Lazy import side effects

Khi flag OFF, lazy import (inside if block) KHÔNG execute → an toàn.
Top-level `import` execute regardless → side effect → có thể affect RNG/state.

---

## 7. Prompt template từ planning session

Planning session sẽ gửi prompt theo format:

```markdown
# [Phase X Step Y] — [Tên feature]

## Context
[Tình hình hiện tại + lý do]

## Goal
[Implement gì cụ thể]

## Pre-flight checks (BẮT BUỘC)
1. Read `file1` — [verify gì]
2. Read `file2` — [verify gì]
...

## Files cần TẠO MỚI
### N. `<path>`
[Code skeleton hoặc API spec]

## Files cần SỬA
### `<path>`
[Diff]

## Smoke test (BẮT BUỘC)
[Bash command với mock data]

## Báo cáo cần submit
1. [List items]

## DON'T
- KHÔNG [action]
...

## Server-side instructions (cho user, KHÔNG phải execution session)
[Bỏ qua phần này, không cần làm]
```

→ Đọc đủ 7 sections. **KHÔNG skip DON'T**. Báo cáo theo format chỉ định.

---

## 8. Format báo cáo cho planning

Khi xong, gửi back:

```
✅ Pre-flight done:
- file1: [confirm 1 dòng]
- file2: [confirm 1 dòng]

✅ Files created:
- path/to/new1.py — [purpose]
- path/to/new2.sh — [purpose]

✅ Files modified:
- path/to/existing.py — [what changed]

✅ Smoke test output:
[paste]

✅ Default OFF verified: [1 dòng explain]

[Notes / questions if any]
```

---

## 9. Khi không chắc → ASK planning

Ask trước khi:
1. API library không quen → đọc xong vẫn phải đoán
2. Phase 8 FULL backbone components — modify cần planning approval
3. Cleanup files ≥ 5
4. Modify file > 500 dòng (vd `train.py`, `gaussian_model.py`) — confirm scope
5. Smoke test fail không hiểu → planning diagnose

---

## 10. Environment context — Local vs Server

### 10.1 Two-environment split

| Environment | Role | Used for |
|-------------|------|----------|
| **Local Windows** (execution session ở đây) | Code editor | Edit/Write `.py`/`.sh`, smoke test mock |
| **Server Linux** (user chạy, KHÔNG phải execution session) | Training compute | Run train.py, ablation, render, multi-seed |

### 10.2 Local Windows (execution session)

```
Path:    d:\Dowload\Paper\3D representation\code\CRSGaussian\
OS:      Windows 11
Python:  via VSCode integrated terminal
Tools:   Read, Edit, Write, Grep, Glob, Bash (limited to local checks)
GPU:     KHÔNG có CUDA build matching server
Data:    KHÔNG có data/nerf_llff_data
```

**Allowed local Bash**:
- `python -c "import ..."` (verify Python syntax/import)
- `pip install --dry-run <pkg>` (verify deps không upgrade torch)
- `grep`, `find` (qua dedicated tools)

**KHÔNG được làm local**:
- `python train.py ...` (không có CUDA + data)
- `pip install <pkg>` mà chưa dry-run (risk break local env)
- `bash scripts/p*_master.sh` (no GPU)

### 10.3 Server Linux (user chạy, KHÔNG phải execution)

```
Path:    ~/workspace/representation-3d/duyen/CoR-GS/
Note:    Tên folder server là CoR-GS (giữ tên BASE), KHÔNG phải CRSGaussian
OS:      Linux Ubuntu
GPU:     2× NVIDIA RTX A4000 (CUDA 12.1)
Conda envs:
  - corgs    : main CRSGaussian env (torch 2.1.0+cu121, Python 3.8)
  - dust3r   : ❌ REMOVED 2026-05-08 (Phase 10A axis dead)
Data:    data/nerf_llff_data/ (8 LLFF scenes)
```

**Server-side workflow** (user thực hiện):
1. `git pull` hoặc rsync code mới từ local
2. `conda activate corgs`
3. `bash scripts/p11sX_multiseed.sh > logs/p11sX_ms/gpu0.log 2>&1 &` (2-GPU parallel)
4. `wait` + `python scripts/p11sX_multiseed_analyze.py`
5. Paste output về planning session

### 10.4 Path mapping local ↔ server

| Local Windows | Server Linux |
|---------------|--------------|
| `d:\...\CRSGaussian\` | `~/workspace/.../CoR-GS/` |
| `data\nerf_llff_data\` (không có) | `data/nerf_llff_data/` (có) |
| `output\p11s*\` (không có) | `output/p11s*/` (có) |
| `logs\p11s*\` (không có) | `logs/p11s*/` (có) |

→ **Trong code, dùng relative path** (`./scripts/...`, `data/...`, `logs/...`) để portable cả 2 env. KHÔNG hardcode prefix.

### 10.5 Sync workflow

```
Local Windows ──── git push / rsync ────→ Server Linux
   (Edit/Write)                            (Run training)
       ↑                                       │
       └──── paste analyzer output ────────────┘
```

Execution session **KHÔNG SSH server**. Chỉ code local → user push → user chạy → user paste kết quả.

### 10.6 Multi-seed ablation pattern trên server (template)

```bash
# 2-GPU parallel split, 3 seeds × 8 scenes × 2 configs = 48 runs ~2.5h
mkdir -p logs/p11sX_ms output/p11sX_ms
GPU=0 SCENES_OVERRIDE="fern flower fortress horns" \
    bash scripts/p11sX_multiseed.sh > logs/p11sX_ms/gpu0.log 2>&1 &
GPU=1 SCENES_OVERRIDE="leaves orchids room trex" \
    bash scripts/p11sX_multiseed.sh > logs/p11sX_ms/gpu1.log 2>&1 &
wait
python scripts/p11sX_multiseed_analyze.py
```

→ Khi execution session viết script master/analyzer, **phải hỗ trợ env vars** `GPU=`, `SCENES_OVERRIDE=`, `SEEDS_OVERRIDE=`, `LOG_DIR=`, `OUT_DIR=` để user split GPU.

### 10.7 Common issues local

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError` import test | Local thiếu deps | Skip local test, để server verify |
| Path separator backslash trong Windows | Windows paths khác Linux | Dùng `os.path.join` hoặc `Path(...)`, KHÔNG hardcode `\` hoặc `/` |
| Permissions denied khi `chmod` | Windows không support | Skip chmod, scripts Linux tự executable qua `bash` |

### 10.8 Common issues server (báo cho user, không tự fix)

| Issue | Báo gì cho user |
|-------|-----------------|
| `ImportError: cuDNN version mismatch` | "Rebuild rasterizer: `cd submodules/diff-gaussian-rasterization-confidence && pip install -e .`" |
| OOM trên GPU | "Giảm batch size hoặc dùng `-r 8` thay `-r 4`" |
| `RuntimeError: CUDA out of memory` mid-training | "Restart GPU process: `nvidia-smi`, kill stale processes" |
| Cluster GPU sharing | "Check `nvidia-smi`, đợi GPU free hoặc dùng different GPU id" |

---

## 11. Honest reporting (BẮT BUỘC)

Execution session **PHẢI** báo cáo trung thực cho planning session. Không ai phạt fail — chỉ phạt che giấu.

### 11.1 Report ACCURATE state, KHÔNG report "looks good"

| Tình huống | Tốt | Tệ (KHÔNG làm) |
|-----------|-----|----------------|
| Smoke test fail | "Smoke test failed: `TypeError: expected Tensor got tuple`. Trace ở line X. Cần fix Y." | "Smoke test OK" (lừa) |
| Implementation lệch spec | "Spec yêu cầu cosine, tôi implement L1 vì cosine require shape match. Cần planning approve." | Im lặng implement L1 |
| Pre-flight skip 1 file | "Đọc 4/5 files, file 5 không tồn tại tại path spec. Báo planning để confirm path." | Skip pre-flight, đoán |
| Default OFF không verify | "Default OFF: lý thuyết identical, NHƯNG chưa test confirm. Cần planning approve risk." | "Verified default OFF" (chưa kiểm) |

### 11.2 Report EXACT numbers, KHÔNG round-up

```
✅ Đúng: "Smoke test ratio: 4.87 (expected > 5.0 → FAIL marginal)"
❌ Sai:  "Smoke test ratio ~5 (OK)" (rounded up to look pass)

✅ Đúng: "Pre-flight verify: min_valid_frac default = 0.3 trong code (spec yêu cầu 0.30, match)"
❌ Sai:  "Pre-flight OK" (no exact value)
```

### 11.3 Report DEVIATIONS từ spec

Nếu phải lệch spec, **luôn báo cáo + lý do**:

```
[NOTE] Spec yêu cầu thread args qua scene/__init__.py, NHƯNG existing code đã có pattern
khác (args passed via dataset.args). Tôi follow existing pattern (cleaner). 
Planning có muốn refactor về spec pattern?
```

### 11.4 Report UNCERTAINTY

```
[UNCERTAIN] Step 4 feature_mpc_loss có 2 places trong train.py có thể hook. 
Tôi hook ở place A (sau L_depth) vì gần hơn với pattern Step 2. 
Place B (sau Step 2 hook) cũng valid. Confirm choice?
```

### 11.5 Báo cáo kết quả smoke test FULL output

KHÔNG truncate hoặc paraphrase. Full stdout:

```
✅ Smoke test 1 — Basic shape + grad flow:
$ python -c "..."
TV depth loss: 0.123456
grad shape: torch.Size([1, 200, 200])
OK basic

✅ Smoke test 2 — Edge-preserving verify:
$ python -c "..."
Loss smooth region: 0.087612
Loss edge region:   0.012453
Ratio (smooth/edge): 7.03x — expect >5x for edge-preserving
OK edge-preserving verified
```

→ Planning session dùng exact numbers để diagnose.

### 11.6 Negative findings = equally valuable

| Tình huống | Cần report |
|-----------|------------|
| File spec không tồn tại | "Path X không tồn tại. Cần path đúng." |
| Function spec không matching | "Function signature spec: `f(a, b, c)`. Existing: `f(a, b)`. Cần planning approve add `c`." |
| Smoke test exposes bug | "Test fail vì lý do Y. Cần fix." |
| Implementation 2x effort than estimated | "Spec estimate 0.5d, thực tế cần 1d vì lý do Z. Báo trước để planning điều chỉnh." |

→ Negative findings **PHẢI** be reported. Hide → planning fails to diagnose → wasted runs.

---

## 12. Phase 11 status snapshot (cập nhật 2026-05-11)

**Phase 11 EXHAUSTED — 6/6 attempts REJECTED:**
```
Step 1 Covisibility       🟡 MARGINAL — keep code default OFF
Step 2 Perceptual DINO    ❌ REJECTED → cleanup pending
Stack (S1+S2)             ❌ REJECTED → cleanup script pending
Step 4 Cross-view MPC     ❌ REJECTED → cleanup pending
Step 5 TV depth EP        ❌ REJECTED → cleanup pending
Step 3 R_feature          ⏸ DEFERRED → cleanup pending (commented out kwargs)
```

**Phase 12 current — cleanup + new attempts:**
```
T12.0 [~] Cleanup Phase 11 REJECTED + Phase 5b modules
T12.1 [ ] Iter budget 15k test (0 code change)
T12.2 [ ] Visibility-based prune (NEW module)
T12.3 [ ] Mip-Splatting anti-aliasing (CUDA rasterizer mod)
```

**Code files post-cleanup (planned):**

KEEP:
- `utils/loss/covisibility_depth.py` (Step 1, MARGINAL keep OFF)
- Phase 8 FULL components in `utils/crs/` (winning recipe)
- Phase 7 LWEIGHT code in `train.py` (reference)
- Phase 10A code (separate decision later)

DELETE (Rule 13):
- `utils/loss/perceptual_dino.py` (Step 2)
- `utils/loss/feature_mpc_crossview.py` (Step 4)
- `utils/regularizer/tv_depth.py` (Step 5)
- `utils/feature/dino_wrapper.py` (only used by deleted Step 2/3/4)
- `utils/feature/__init__.py`
- `utils/crs/r_feature.py` (Step 3 deferred indef)
- `utils/regularizer/pseudo_*.py` (Phase 5b rejected)
- `scripts/p11s2_*`, `p11s4_*`, `p11s5_*`, `p11s12_*` (rejected)

**Code regression note**: Phase 8 paper 21.335 NOT reproducible from current code (gap −0.155 vs paper, commit `0511edd` admitted "mất config"). Paper PLY còn saved tại `output/p8/FULL_*/point_cloud/iteration_10000/`. Accept multi-seed mean 21.18 as fair baseline.

---

## 13. Cross-reference

| File | Role |
|------|------|
| `docs/00a_planning_session.md` | Planning role rules |
| `docs/00b_execution_session.md` (THIS) | Execution role rules |
| `CLAUDE.md` | Project context + công thức cốt lõi |
| `docs/03_task_queue.md` | Phase status hiện tại |
| `docs/04_decisions_log.md` | History decisions |
