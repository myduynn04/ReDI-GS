# Code Session Bootstrap — ĐỌC TRƯỚC KHI CODE

> **Đối tượng**: Mọi Claude Code session được giao implement task trên CRSGaussian.
> **Mục đích**: Tránh hỏi lại / sai sót do không biết context.
> **Thứ tự đọc**: file này → CLAUDE.md → docs/03_task_queue.md (task hiện tại).

---

## 1. Vai trò + môi trường

### 1.1 Phân biệt 2 môi trường

| Môi trường | Vai trò | Path |
|------------|---------|------|
| **Local Windows** (session này) | **Code-only** — Read/Edit/Write file | `d:\Dowload\Paper\3D representation\code\CRSGaussian\` |
| **Server Linux** (user tự chạy) | Training + inference | `~/workspace/representation-3d/duyen/CoR-GS/` |

### 1.2 ⚠️ Path mapping QUAN TRỌNG

- **Local**: project tên `CRSGaussian/`
- **Server**: SAME project nhưng tên folder là `CoR-GS/` (giữ tên BASE để git diff sạch với upstream)
- Khi viết script `.sh` chạy trên server: dùng path tương đối (`./scripts/...`, `data/...`) — KHÔNG hardcode tên folder
- Khi user nhắc "trên server", "ở server" → ngầm hiểu code chạy ở `CoR-GS/`, KHÔNG đổi gì trong code

### 1.3 ⚠️ Code session KHÔNG chạy training

- Local Windows **không** có CUDA build phù hợp + không có data
- KHÔNG `python train.py`, KHÔNG `bash scripts/p*_master.sh` từ session này
- Test chỉ ở mức **smoke test** (import, mock numpy arrays, sanity check) — **KHÔNG GPU work**
- Training thực = user chạy trên server, paste log về cho session phân tích

### 1.4 Hai conda env trên server (không nhầm)

| Env | Dùng khi nào |
|-----|--------------|
| `<crsgaussian-env>` (tên user tự đặt, có thể `cor-gs`) | `python train.py`, render, metrics |
| `dust3r` | CHỈ `scripts/precompute_dust3r.py` (Phase 10A pre-compute) |

---

## 2. Coding rules (distilled từ CLAUDE.md)

### Rule 1 — Marker `[CRSGaussian]` cho code mới

Mọi function/section thêm hoặc sửa **PHẢI** có comment marker để grep ngược tìm được sau:

```python
# ============================================================
# [CRSGaussian Phase 10A] Task: dense_init_hook
# File: scene/dataset_readers.py
# Mục đích: ...
# ============================================================
```

Inline cũng được:
```python
# ── [CRSGaussian Phase 8] R_visible computation ──
```

→ `grep -r "[CRSGaussian Phase 10A]"` phải lấy được mọi đoạn code Phase 10A.

### Rule 2 — Master switch pattern (default OFF)

Mọi feature mới:

1. **Flag** trong `arguments/__init__.py` → `self.use_<feature> = False`
2. **Gating** trong code → `if dataset.use_<feature> and <conditions>:`
3. **Default OFF** đảm bảo behavior trước-feature unchanged
4. **Verify** bằng cách: chạy 1 config without flag → metric phải match log baseline

### Rule 3 — Tách file (KHÔNG nhồi vào train.py)

- Logic mới >30 dòng → tạo file riêng tại `utils/<dir>/<feature>.py`
- `train.py` chỉ chứa: `import + gate block + 1-3 dòng gọi` (interface clean)
- Subdir: `utils/depth/`, `utils/crs/`, `utils/loss/`, `utils/init/`, `utils/regularizer/`

### Rule 4 — Read trước khi edit, không đoán API

- LUÔN `Read` file gốc trước khi sửa — KHÔNG bao giờ đoán signature/return type
- Khi tích hợp library ngoài (DUSt3R, DAV2, etc.): `Read` source code library trong `external/` để verify API
- Nếu không tìm thấy file: dùng `Grep` / `Glob` chứ KHÔNG fabricate

### Rule 5 — Không sửa BASE folders chỉ-đọc

Folders **CHỈ ĐỌC** (đừng touch):
- `FSGS/`, `DNGaussian/`, `LoopSparseGS/`, `DepthRegularizedGS/`, `SCGaussian/`
- `external/dust3r/` (foundation model code, không sửa)
- `submodules/diff-gaussian-rasterization-confidence/`, `submodules/simple-knn/` (CUDA, sửa qua user explicit request)

Sửa code → **chỉ trong**: `CoR-GS/` root files, `utils/`, `scene/`, `arguments/`, `gaussian_renderer/`, `scripts/`, `tests/`.

### Rule 6 — Không tự ý chạy training để verify

- KHÔNG đề xuất `python train.py ...` để verify code
- Verify đúng cách:
  1. Smoke test (mock data, không cần GPU): `python -c "from utils.x import f; f(mock_args); print('OK')"`
  2. Import test: `python -c "import scene; import train; print('OK')"`
  3. Verify default OFF: đọc `arguments/__init__.py` thấy default `False` → giải thích bằng lời, không cần chạy

### Rule 7 — Cleanup khi reject hướng

Khi user QUYẾT ĐỊNH bỏ một feature/ablation (đã có verdict NO):
1. **Đề xuất** cleanup list cho user approve trước khi rm
2. Cleanup gồm: `scripts/<feature>*.{sh,py}`, `utils/.../<feature>.py`, flags trong `arguments/`, gate block trong `train.py`, `output/<feature>/`, `logs/<feature>/`
3. **GIỮ**: docs/decisions log, summary log cuối (cho paper discussion)
4. Verify bằng `grep -r "<feature>" .` → phải empty sau cleanup

### Rule 8 — Comment style

- Header block đầu function/class mới (Việc 1 trong CLAUDE.md)
- Inline comment tại logic không-hiển-nhiên (Việc 2)
- KHÔNG comment code self-explanatory (`# increment counter` → bỏ)
- Comment explain **WHY**, không phải WHAT

---

## 3. Phase status hiện tại (cập nhật khi đổi)

> **Cập nhật mỗi khi xong Phase**, không để stale.

**Date last updated**: 2026-05-08

```
Phase 0-7  — DONE (xem CLAUDE.md)
Phase 8    — DONE (BREAKTHROUGH 21.335 dB, +0.125 vs No-CRS)
Phase 9    — DONE (FULL recipe LOCKED, simplifications HURT)
Phase 10   — DONE (FAILED, axis DEAD)
             AUGMENT Δ=−0.898, REPLACE Δ=−3.529, diagnostic ceiling −0.07
             → Foundation-model dense init BỎ HẲN
             Cleanup: env+checkpoint+cache+source removed
Phase 11   — CURRENT (Loss-axis Exploration, sequential strategy)
             Step 1 [~] CRS × Covisibility (depth-based, KHÔNG dùng DUSt3R) — IN PROGRESS
             Step 2 [ ] Same-view perceptual (DINOv2)
             Step 3 [ ] R_feature replace R_visible
             Step 4 [ ] Cross-view MPC (conditional)
             Best-single prob ≥+0.20 ≈ 40%. Pivot ready: regularization / depth fine-tune / accept.
Phase 12   — DEFERRED (Full Experiments — chờ Phase 11 verdict)
```

**Reference targets (LLFF 3-view PSNR)**:
- Phase 8 FULL = 21.335 ⭐ project best
- No-CRS Tier1 = 21.21
- DOC-GS = 21.38
- BinocularGS = 21.44
- ICO-GS SOTA = 22.20 (gap −0.865 → goal break)

**Active workflow rules (memory-enforced)**:
- Planning session draft prompts only, KHÔNG Write code production trực tiếp
- Mọi ablation parallelize 2 GPUs (`&` + `wait`), không single GPU sequential
- Diagnostic 1 scene first → confirm 2-3 scenes → scale 8 if winner

---

## 4. Verify-before-done checklist

Trước khi báo "done" cho user, **PHẢI** confirm:

- [ ] Mọi file đã tạo có header marker `[CRSGaussian Phase X]`
- [ ] Master switch flag default = `False`
- [ ] Logic nằm trong file riêng, không nhồi vào train.py
- [ ] Smoke test (numpy mock) PASS
- [ ] Verify default OFF: explain bằng lời tại sao behavior unchanged khi flag tắt
- [ ] List rõ file đã tạo/sửa cho user verify
- [ ] Cleanup imports không dùng (no `import X` orphan)
- [ ] KHÔNG chạy training command nào

---

## 5. Common pitfalls (tránh lặp lại)

### P1: Hardcode path tuyệt đối

❌ `dust3r_path = "/home/aidev/workspace/.../external/dust3r"`
✅ `dust3r_path = os.path.join(os.path.dirname(__file__), "..", "external", "dust3r")`
✅ Hoặc CLI arg: `--dust3r_path` (recommend cho scripts)

### P2: Không thread args xuống dataset_readers

Khi feature cần đọc flag từ `args`:
- Sửa signature `readColmapSceneInfo(..., args=None)` (default None để backward compat)
- Sửa `scene/__init__.py` truyền `args=args` kwarg vào `sceneLoadTypeCallbacks[...]`
- Trong dataset_readers, gate bằng `if args is not None and getattr(args, "<flag>", False):`

### P3: Quên LLFF train/test split logic

LLFF eval split: `train_idx = idx % llffhold != 0` (llffhold=8 default).
n_views subsample: `np.linspace(0, len(train) - 1, n_views).round()`.
→ Nếu pre-compute cần "đúng 3 train images", phải tái-implement logic này (KHÔNG dùng all images).

### P4: COLMAP coordinate convention

- COLMAP `images.bin` lưu **world→cam** extrinsic
- Many libraries (DUSt3R, NeRF, OpenGL) dùng **cam→world**
- Convert: `C2W = inv([[R_w2c, T_w2c], [0, 1]])` ↔ `R_c2w = R_w2c.T`, `T_c2w = -R_w2c.T @ T_w2c`
- LUÔN test với 1 known camera pose trước khi process batch

### P5: Confusing "image_size" vs "resolution"

- DUSt3R `image_size=512` → long edge resize cho inference
- 3DGS `-r 8` → downscale factor (1/8), output ~size/8
- KHÔNG nhầm: pre-compute và training dùng resolution KHÁC nhau là OK (DUSt3R PC ở world space, không bind theo resolution)

### P6: Auto memory check

Memory file `MEMORY.md` ở `C:\Users\OSC\.claude\projects\d--Dowload-Paper-3D-representation-code\memory\` — check trước khi đề xuất hướng đi mới.

### P7: Báo cáo dài dòng

User đã expressed: muốn report **terse**, không trailing summary. Report theo format:
- Files changed (1 line each)
- Smoke test result (1 line)
- 1 sentence verdict
- KHÔNG paragraph "I have successfully..."

---

## 6. Khi không chắc → ASK

Tốt hơn hỏi user 1 câu rõ ràng còn hơn implement sai phải redo. Cases nên ask:

1. API library không quen → đọc xong vẫn phải đoán → ASK
2. User nhắc tên feature/file không tìm thấy → ASK trước khi tạo mới
3. Architecture decision (folder structure, naming convention) → ASK
4. Cleanup hơn 5 files → ASK liệt kê + confirm trước rm
5. Modify file > 200 dòng (e.g., `train.py`, `scene/__init__.py`) → ASK confirm scope trước

---

## 7. Format prompt user thường dùng

User thường gửi prompt theo pattern:

```
# Task tên
Context: ...
Goal: ...
Files cần TẠO MỚI: ...
Files cần SỬA: ...
Quy tắc: ...
Báo cáo cần submit: ...
DON'T: ...
```

→ Đọc đủ 7 sections, KHÔNG skip phần "DON'T". Báo cáo theo format "Báo cáo cần submit".
