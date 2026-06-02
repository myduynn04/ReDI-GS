# Paper Defense — Tested-but-Removed Features

**Mục đích:** Compile defense-ready Q&A cho mọi feature đã test rồi remove. Dùng để trả lời reviewer nhanh "tại sao bạn không thử/so sánh X?".

**Pattern**: mỗi feature có entry với:
- **Q** (câu hỏi reviewer có thể hỏi)
- **A** (câu trả lời quotable, có Δ + CI + N + citation)
- **Evidence files** (log/script/memory để reproduce)
- **Note** (mechanism hypothesis nếu có)

> ⚠️ KHÔNG bao giờ trả lời "we didn't try X" — luôn cite evidence "we tested X with method Y, result Z, references ..."

---

## 1. CRS₀ Initialization (`informed_crs_init`) — REMOVED 2026-05-29

**Module purpose** (đã từng): khởi tạo CRS₀ có ý nghĩa hình học từ 3 signals (q_reproj từ COLMAP SfM error + q_depth từ DAV2-COLMAP agreement + q_view multi-view stereo support), thay vì neutral 0.5 cho tất cả Gaussian.

### Q: Tại sao remove `informed_crs_init` mà không so sánh?

**A:** **ĐÃ TEST cross-backbone N=24 paired**:

| Backbone | Source | Δ vs base | 95% CI | Verdict |
|---|---|---|---|---|
| MVS classical | Phase 20 (2026-05-23) | wash (within ±0.10) | — | safe to trim |
| RoMa v1 dense | Phase 24 (2026-05-29) | **−0.013** | [−0.081, +0.054] | ~ WASH |

- Phase 5 best WG (w_reproj=0.4, w_depth=0.6, w_view=0) đã được dùng — KHÔNG phải default neutral
- Cross-backbone (2 init methods × N=24 paired) = decision data-driven, không hardcode

**Evidence:**
- `memory/project_phase24_trim_verify_v1.md` (full data table)
- `docs/04_decisions_log.md` [2026-05-29] Phase 24 entry
- `logs/p24_trim_add_v1/c1_informed/` (24 server logs)
- Analyzer: `scripts/p24_trim_add_v1_analyze.py`

**Mechanism hypothesis**: RoMa v1 dense init đã rich enough → informed CRS₀ không thêm info-gap để vá. Trên MVS sparse cũng wash vì DropAnSH + opacity decay đã handle dirty Gaussians runtime.

### Q: Sao chỉ Phase 5 best WG, không grid search thêm?
**A:** Phase 5 đã grid search 6 weight combinations (T5.1-T5.6, AVG=20.36 best). Phase 24 dùng winner đó để give it fair chance. Nếu winner WASH → variant khác WASH hơn.

---

## 2. CRS Pruning (`use_crs_pruning`) — REMOVED 2026-05-29

**Module purpose** (đã từng): tách CRS pruning thành kênh riêng (`prune = (CRS<tau_crs AND knn_dist>tau_isolated) OR opacity<0.005`), không phụ thuộc legacy opacity threshold.

### Q: Tại sao remove `use_crs_pruning`?

**A:** **ĐÃ TEST cross-backbone N=24 paired**:

| Backbone | Source | Δ vs base | 95% CI | Verdict |
|---|---|---|---|---|
| MVS classical | Phase 20 | wash | — | safe |
| RoMa v1 dense | Phase 24 | **+0.014** | [−0.065, +0.086] | ~ WASH |

- Hyperparams Phase 4 defaults validated (T_warmup=1000, tau_crs=0.35, tau_isolated=0.1) — không arbitrary
- DropAnSH (random + crs-modulated SH dropout) + opacity decay (0.999/iter) đã cover dirty Gaussians

**Evidence**: Phase 24 logs + analyzer (như trên).

**Mechanism hypothesis**: Triple overlap với DropAnSH + opacity decay → marginal contribution = 0.

---

## 3. R Visibility-aware (`use_r_visible`) — REMOVED 2026-05-29

**Module purpose** (đã từng): Phase 8a fix cho R contamination 36.5% (Tier A4) bằng cách filter view k khi gauss_z > rendered_z × tolerance (occlusion check), chỉ aggregate views nơi Gaussian thật sự visible.

### Q: Tại sao remove `use_r_visible`?

**A:** **ĐÃ TEST cross-backbone N=24 paired**:

| Backbone | Source | Δ vs base | 95% CI | Verdict |
|---|---|---|---|---|
| MVS classical | Phase 20 | wash | — | safe |
| RoMa v1 dense | Phase 24 | **+0.013** | [−0.048, +0.081] | ~ WASH |

- Phase 8a defaults validated (occlusion_tolerance=1.05, min_visible_views=2)

**Mechanism hypothesis**: RoMa v1 dense matcher đã giải quyết occlusion gián tiếp (matching nhiều điểm + filter rejection). Trên MVS sparse, R signal weight w2=0.5 đủ nhỏ để contamination 36.5% không impact PSNR significant.

---

## 4. Stack-3 Anti-Synergy — KEY DEFENSE FINDING (2026-05-29)

**Test**: Phase 24 case c4_stack3 = base + cả 3 flags ON đồng thời.

**Result**: Δ = **−0.796 dB** [−1.078, −0.532] ❌ **HURT** (95% CI exclude 0).

**Per-scene**: 8/8 negative — horns −1.597, trex −1.845, leaves −1.319, orchids −0.745, fortress −0.376, fern −0.249, flower −0.188, room −0.045.

### Q: Synergy thường +. Tại sao 3 modules cùng OFF lại tốt hơn cùng ON?

**A:** ANTI-SYNERGY discovered:
- 3 modules alone: WASH (noise level, Δ ≈ 0 ±0.10)
- 3 modules STACKED: catastrophic −0.8 dB

**Mechanism**:
- `informed_crs_init`: set CRS₀ thấp cho nhiều points
- `use_crs_pruning`: aggressively prune low-CRS
- `use_r_visible`: strict filter R signal
- → Triple penalty → over-prune valid Gaussians, đặc biệt thin-structure (horns/trex/leaves)

**Đối lập** Phase 13 LFCF×AbsGS synergy DƯƠNG +0.071 (74% over linear sum). Đây là synergy ÂM lần đầu trong project.

### Q: Có nghĩa là remove bảo vệ user khỏi vô tình bật cả 3?
**A:** Đúng. Remove 3 flags KHÔNG CHỈ neutral mà còn **protective**.

### Q: Insight này paper-worthy?
**A:** Có — đây là **contribution #5 của paper**: "Legacy CRS regularizer stack incompatible with dense init backbone". Cảnh báo cho future work không stack legacy CRS modules trên modern dense matchers.

---

## Template cho future entries (khi reject thêm features)

```markdown
## N. Feature Name (`flag_name`) — REMOVED YYYY-MM-DD

**Module purpose** (đã từng): [mô tả ngắn 1-2 câu]

### Q: Tại sao remove `flag_name`?

**A:** **ĐÃ TEST [method] N=[size] paired**:

| Backbone/Scenario | Source | Δ vs base | 95% CI | Verdict |
|---|---|---|---|---|
| [backbone 1] | Phase X | ±Δ | [lo, hi] | WASH/HURT/HELP |
| [backbone 2] | Phase Y | ±Δ | [lo, hi] | WASH/HURT/HELP |

- [Hyperparams justification — tại sao chọn các params này, không arbitrary]
- [Cross-something evidence]

**Evidence:**
- memory/project_phaseX.md
- docs/04_decisions_log.md [YYYY-MM-DD]
- logs/pX/ (server log path)
- Analyzer: scripts/pX_analyze.py

**Mechanism hypothesis**: [tại sao module này không work]
```

---

**Maintenance**: Khi reject feature mới trong tương lai, copy template + fill in. Đây là file canonical cho reviewer defense.
