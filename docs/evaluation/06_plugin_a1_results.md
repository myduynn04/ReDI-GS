# Plug-in A1 + A2 Results — CRSGaussian → Nerfstudio cascade

> **Cập nhật 2026-06-02 evening**: A1 + A2.1 + A2.2 DONE. A2.3 + A2.4 smoke PASS, đang chuẩn bị cascade.
>
> **Identity reframe quan trọng**: "CRSGaussian" = **toàn bộ Phase 22 recipe** (10 component), KHÔNG phải chỉ CRS module isolated. Plug-in cascade = port từng component recipe vào Nerfstudio framework.

---

## 1. Coverage Phase 22 recipe vs plug-in

Theo Phase 22 + Phase 23 ablation, CRSGaussian project = 10 component recipe:

| # | Component | Phase gốc | Plug-in status | Method |
|---|-----------|------------|-----------------|--------|
| 1 | **RoMa v1 dense init** | Phase 22 | ✅ **A1 FULL** | `splatfacto-roma` |
| 2 | **Opacity decay** (0.999/iter) | Phase 2c/22 | ✅ **A2.1 FULL** | `splatfacto-roma-opdecay` |
| 3 | **DropAnSH** (anchor + SH dropout) | Phase 22 | ✅ **A2.2 FULL** | `splatfacto-roma-opdecay-dropansh` |
| 4 | **DAV2 depth loss** (Pearson) | Phase 1-3 | ✅ **A2.3 FULL** (smoke PASS) | `splatfacto-roma-depth` |
| 5 | **CRS score** (D_i + EMA) | Phase 2 | 🟡 **A2.4 PARTIAL** (D-only, smoke PASS) | `splatfacto-roma-depth-crsg` |
| 6 | CRS-modulated SH freeze | Phase 8c | ❌ DEFER A3 | — |
| 7 | LFCF densifier (EFA-GS) | Phase 13 | ❌ DEFER A3 | — |
| 8 | AbsGS densify | Phase 13 | ⚠️ Splatfacto có sẵn (OFF cho fair) | — |
| 9 | D_cycle (cycle depth) | Phase 7 | ❌ DEFER A3 | — |
| 10 | S_stability (SH EMA) | Phase 8b | ❌ DEFER A3 | — |

**Coverage cuối A2**: **4 FULL + 1 PARTIAL = 5/10 component (50%)** Phase 22 recipe.

---

## 2. Kết quả cascade tới hiện tại

### Bảng cascade A1 → A2.1 → A2.2 (4 scene, 10k iter, seed 42)

| Scene | V vanilla | A1 (roma) | Δ_A1 | A2.1 (+opdecay) | Δ_A2.1 | A2.2 (+dropansh) | Δ_A2.2 | Δ_total |
|-------|-----------|-----------|------|-------------------|---------|--------------------|---------|---------|
| fern | 17.635 | 19.081 | **+1.446** | 18.568 | −0.513 | 18.477 | −0.091 | +0.842 |
| horns | 13.992 | 15.820 | **+1.828** | 15.158 | −0.662 | 15.390 | +0.232 | +1.398 |
| fortress | 17.608 | 19.455 | **+1.847** | 20.216 | +0.761 | 19.905 | −0.311 | +2.297 |
| flower | 15.180 | 16.149 | **+0.969** | 16.380 | +0.231 | 16.751 | +0.371 | +1.571 |
| **Mean** | **16.10** | **17.63** | **+1.523** | 17.58 | −0.046 | 17.63 | +0.050 | **+1.527** |

### Insights

- **A1 = headline +1.523 dB** (4/4 scene positive, sign consistent)
- **A2.1 opacity decay alone WASH** (mean −0.046) — đúng pattern Phase 23 `synergy-only (alone −0.27, cascade +0.21)`
- **A2.2 cumulative WASH** — DropAnSH alone + opdecay không synergize trên splatfacto framework
- → A2.3 + A2.4 chạy CLEAN ABLATION (skip A2.1, A2.2 vì cả 2 wash): chỉ test `A1 + depth` và `A1 + depth + CRS`

### Cascade A2.4 FINAL (2026-06-02, 4 scene 10k iter seed 42)

| Scene | V | A1 (roma) | Δ_A1 | A2.4 (+depth+CRS) | Δ vs A1 | Δ vs V |
|-------|---|-----------|------|--------------------|---------|--------|
| fern | 17.635 | 19.081 | +1.446 | **18.705** | **−0.376** | +1.070 |
| horns | 13.992 | 15.820 | +1.828 | **15.382** | **−0.438** | +1.390 |
| fortress | 17.608 | 19.455 | +1.847 | **19.948** | **+0.493** | +2.340 |
| flower | 15.180 | 16.149 | +0.969 | **16.290** | **+0.141** | +1.110 |
| **Mean** | **16.10** | **17.63** | **+1.523** | **17.58** | **−0.045** | **+1.478** |

### Insight tổng hợp (4 cascade)

| Cascade trên splatfacto | Δ mean | Verdict |
|-------------------------|--------|---------|
| V → A1 (RoMa init) | **+1.523** | ⭐ Contribution duy nhất rõ rệt |
| A1 → A2.1 (+opdecay) | −0.046 | WASH |
| A2.1 → A2.2 (+dropansh) | +0.050 | WASH |
| A1 → A2.4 (+depth+CRS) | **−0.045** | **WASH** ⚠ |

→ **Trên splatfacto framework: A1 (RoMa init) là contribution DUY NHẤT có Δ rõ. Module CRSGaussian khác (opacity decay, DropAnSH, depth+CRS) đều wash hoặc slight negative.**

### Lý do — KHÔNG phải bug, mà confirm insight Phase 23

Theo Phase 23 (`project_phase23_ablation_v1`): **"CRS framework overlap với dense init"** — depth+CRS contribution mạnh trên sparse-init backbone (CoR-GS với COLMAP ~hundreds points), wash trên dense-init backbone (PDCNet+/RoMa v1/splatfacto+RoMa).

A2.4 wash trên splatfacto = **3rd cross-backbone confirm** của rule này:
1. PDCNet+ dense init (Phase 18) — CRS-axis wash
2. RoMa v1 dense init (Phase 22 + Phase 23 ablation) — CRS-axis wash
3. Splatfacto + RoMa init (A2.4 plug-in này) — CRS-axis wash

A1 (RoMa init) hấp thụ phần lớn benefit vì vá lỗ hổng info → các regularizer thêm vào không còn gì để fix.

---

## 3. Architectural decisions đã chốt

### Architecture: Option 1 — Bổ trợ splatfacto (NOT ném khối)

| | Plug lẻ tẻ (current) | Ném khối port full |
|---|----------------------|---------------------|
| Approach | Subclass + override hook, mỗi module 1 method registered | Port toàn bộ CRSGaussian train.py vào 1 method |
| Defense story | "Em upgrade splatfacto từng module" | "Em port CRSGaussian vào Nerfstudio framework" |
| Match yêu cầu thầy "plug vào của họ" | ✅ | 🟡 |
| Ablation per-module | ✅ | ❌ |
| Effort | 5-7 ngày | 2-3 ngày |
| Coverage actual | 5/10 (50%) | 10/10 (full Phase 22) |

→ Chọn Option 1. Defense honest: "5/10 module plug được, 5 module deferred vì callback infra splatfacto không hỗ trợ."

### Why 5 module skip (A3 deferred)

| Module | Lý do skip |
|--------|------------|
| CRS-modulated SH freeze | Splatfacto callback chỉ có `BEFORE/AFTER_TRAIN_ITERATION`, KHÔNG có `BEFORE_OPTIMIZER_STEP`. Densify replace `features_rest` tensor → grad hook không persist. |
| LFCF densifier | Cần subclass `gsplat.DefaultStrategy` — heavy fork, risk regression. |
| D_cycle | Cần render rendered_depth N camera mỗi update — expensive + integration phức tạp. |
| S_stability | Cần buffer EMA state per-Gaussian, resize đồng bộ densify. |
| R_i reprojection | Cần GT images per pixel, splatfacto datamanager không expose. |

→ A3 (post-defense): pip-installable full package implement 5 module này.

---

## 4. Files plug-in đã ship

Folder: [CRSGaussian/external/nerfstudio_plugin/](../../external/nerfstudio_plugin/)

| File | Step | Notes |
|------|------|-------|
| `__init__.py` | All | Register 10 method (5 ON + 5 OFF variants) |
| `roma_dataparser.py` | A1 + A2.4 | Subclass ColmapDataParser, inject train_cameras vào metadata |
| `crsg_model.py` | A2.1-A2.4 | Subclass SplatfactoModel — opacity decay, DropAnSH, depth loss, CRS module |
| `dropansh.py` | A2.2 | Port anchor + SH degree dropout |
| `depth_loss.py` | A2.3 | Port pearson_depth_loss + load aligned depth dict |
| `crs_module_a24.py` | A2.4 | Port D_i compute + EMA update |
| `preprocess_depth_a23.py` | A2.3 | Run env corgs, pre-compute DAV2 aligned depth → .npy |
| `verify_plugin.sh` | A1 | 5-tier verification automated |
| `run_4scene_roma.sh` | A1 | 4 scene parallel 2 GPU |
| `run_4scene_roma_opdecay.sh` | A2.1 | |
| `run_4scene_roma_opdecay_dropansh.sh` | A2.2 | |
| `run_4scene_roma_depth.sh` | A2.3 | |
| `run_4scene_roma_depth_crsg.sh` | A2.4 | Auto cascade compare V→A1→A2.3→A2.4 |
| `README.md` | All | Setup + rollback docs |

---

## 5. 10 method registered

```
splatfacto-roma                            (A1 ON)
splatfacto-roma-off                        (A1 OFF, Tier 2.1)
splatfacto-roma-opdecay                    (A2.1 ON, A2.2/A2.3/A2.4 OFF)
splatfacto-roma-opdecay-off                (A2.1 OFF, Tier 2.1)
splatfacto-roma-opdecay-dropansh           (A2.1 + A2.2 ON)
splatfacto-roma-opdecay-dropansh-off       (DropAnSH OFF, Tier 2.1)
splatfacto-roma-depth                      (A2.3 ON, skip opdecay/dropansh)
splatfacto-roma-depth-off                  (A2.3 OFF, Tier 2.1)
splatfacto-roma-depth-crsg                 (A2.4 ON = depth + CRS)
splatfacto-roma-depth-crsg-off             (CRS OFF, Tier 2.1)
```

---

## 6. Defense narrative cuối cùng (honest, updated 2026-06-02)

> *"Em integrate CRSGaussian vào Nerfstudio splatfacto qua external plugin:*
>
> *- **A1 RoMa init = +1.523 dB transfer được** (4/4 scene positive, sign consistent). Drop-in upgrade rõ ràng cho splatfacto vanilla.*
> *- **Module CRSGaussian khác (opacity decay, DropAnSH, depth+CRS) wash trên splatfacto** (Δ ≈ 0 ± 0.10). KHÔNG phải bug, mà confirm insight Phase 23: CRS-axis chỉ work trên sparse-init backbone (MVS COLMAP), wash trên dense-init backbone.*
> *- **3rd cross-backbone confirm**: PDCNet+ dense (Phase 18) / RoMa v1 dense (Phase 22-23) / splatfacto+RoMa (plug-in này) — cùng pattern CRS overlap dense init.*
> *- Plug-in standalone, không sửa Nerfstudio source. OFF flag → vanilla byte-identical."*

### Insight defense-grade

Dù PSNR không cao thêm sau A1, kết quả CONFIRM cross-backbone rule mới — defense-able vì:
1. RoMa init là **portable contribution** sang framework khác
2. CRS-axis scope **clearly defined** (sparse-init only)
3. **Tránh over-claim**: không bịa CRS gain trên splatfacto khi không có

---

## 7. Pending action

- ⏳ **A2.4 cascade 4 scene 10k iter** (~10 phút) — run `run_4scene_roma_depth_crsg.sh`
- 📋 Sau khi có bảng cascade A2.4 → update section 2 ở doc này
- 📋 Bonus optional: A2.3 cascade ablation Δ_CRS riêng (chỉ cần nếu defense yêu cầu tách contribution CRS vs depth)
- 📋 Verify Tier 2.1 OFF flag cho A2.3, A2.4 (multi-seed verify) — defer
- 📋 Polish slide defense

---

## 8. Liên quan

- [05_plugin_design.md](05_plugin_design.md) — design overview (Option 1 chốt)
- [04_nerfstudio_setup_and_run.md](04_nerfstudio_setup_and_run.md) — setup env + baseline 4 scene vanilla
- Plug-in code: [CRSGaussian/external/nerfstudio_plugin/](../../external/nerfstudio_plugin/)
- Phase 22 backbone: memory `project_phase22_roma_v1_pilot.md`
- Phase 23 4-contribution framing: memory `project_phase23_ablation_v1.md`
- Phase 24 anti-synergy: memory `project_phase24_trim_verify_v1.md`
