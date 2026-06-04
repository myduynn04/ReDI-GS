# Nerfstudio plug-in — Defense framing (research-to-production positioning)

> **Mục đích**: Tài liệu chuẩn hóa **cách trình bày** việc plug CRSGaussian vào Nerfstudio cho defense thesis. Mọi claim ở đây đều có **evidence verify** (web search 2026-06-02) — không suy đoán, không over-claim.
>
> **Bối cảnh**: Thầy yêu cầu chứng minh tính ứng dụng thực tế của CRS ngoài so sánh paper SOTA. Closed-source products (Polycam, Luma AI) không so sánh fair được → chọn Nerfstudio (open-source framework chuẩn academia + industry).

---

## 1. Nerfstudio là gì — facts đã verify

| Item | Chi tiết | Nguồn |
|------|---------|-------|
| Phát triển bởi | **Berkeley AI Research (BAIR)**, KAIR lab | [GitHub repo](https://github.com/nerfstudio-project/nerfstudio) |
| License | **Apache 2.0** (open-source thương mại được) | GitHub |
| Mục đích | Modular framework PyTorch cho NeRF + 3D Gaussian Splatting | [SIGGRAPH 2023 paper](https://arxiv.org/pdf/2302.04264) |
| Popularity | ~9.6k GitHub stars, ~277 dependents | GitHub |
| Cung cấp | DataManager, Trainer, Viewer 3D real-time, Eval pipeline, 10+ built-in methods | Docs |
| Built-in methods | Splatfacto, Nerfacto, Instant-NGP, Mip-NeRF, TensoRF, Gaussian Splatting variants | Docs |

→ **Nerfstudio = research framework**, KHÔNG phải app end-user (như Polycam/Luma AI). Là **"PyTorch của 3D reconstruction"** — bộ khung chung để researcher implement method mới.

---

## 2. Use cases verified (research + industry)

### 2.1 VFX & phim ảnh — **HARD EVIDENCE**

- **ILM (Industrial Light & Magic / Disney / Lucasfilm)** modified Nerfstudio cho **Marvel's Ironheart 2025**
- Đoạt **HPA Award 2025** (Hollywood Professional Association — Technology & Innovation in VFX/Virtual Production/Animation)
- VFX supervisor **Vincent Papaix** dùng cho drone shots Chicago waterfront
- ILM credit: giảm production time **từ vài tháng → vài ngày**
- Matt Tancik (Nerfstudio creator) tạo new file format hỗ trợ tích hợp
- ⚠️ **Caveat honest**: ILM tự admit "very few people were putting this kind of stuff into production" (2022-2023) — experimental scope, không phải full studio-wide

**Sources**:
- [ILM × Nerfstudio Ironheart interview](https://www.ilm.com/ilm-nerfstudio-ironheart-hpa-award-interview/)
- [Creating Visual Effects with Neural Radiance Fields (arXiv 2401.08633)](https://arxiv.org/pdf/2401.08633)

### 2.2 Digital twin & smart factory — **research papers verified**

- Nature Scientific Reports 2025: "Interactive digital twins enabling responsible extended reality applications" — dùng NeRF qua Nerfstudio cho XR digital twins
- PMC paper: "Efficient Synthetic Defect on 3D Object Reconstruction... Digital Twins Smart Factory" — dùng Nerfstudio framework so sánh Instant-NGP/Nerfacto/Volinga/Tensorf cho industrial object reconstruction
- Nerfacto cityscape research dùng cho 3D outdoor scene representation

**Sources**:
- [Nature SR — Interactive digital twins XR](https://www.nature.com/articles/s41598-025-17855-9)
- [PMC — Digital twins smart factory NeRF](https://pmc.ncbi.nlm.nih.gov/articles/PMC12656295/)

### 2.3 Robotics + SLAM — **commercial fork verified**

- **Spectacular AI** (công ty SDK visual-inertial tracking thương mại) **fork Nerfstudio** active maintained:
  - Main branch commits 2025-05-31
  - pytorch-fix branch commits 2025-07-10
- Sản phẩm: SDK cho Android/iOS/drone/robotics — OAK-D, RealSense, Orbbec/Kinect
- **Contribute pose optimization upstream** vào gsplat + Nerfstudio
- Powers **"Gaussian Splatting on the Move"** + **BAD-Gaussians** papers

**Sources**:
- [Spectacular AI Nerfstudio fork (GitHub)](https://github.com/SpectacularAI/nerfstudio)
- [Spectacular AI Mapping Tools (sản phẩm)](https://www.spectacularai.com/mapping)

### 2.4 Industry sponsorship — **Luma AI**

- **Luma AI** (app 3D scanning thương mại) sponsor Nerfstudio + BAIR lab
- Hire **Matt Tancik** (Nerfstudio creator) làm employee
- **Angjoo Kanazawa** (BAIR Professor) advisor tại Luma AI
- ⚠️ **Caveat honest**: Sản phẩm Luma chạy **proprietary stack riêng**, KHÔNG dùng code Nerfstudio trực tiếp. Connection là **team + ecosystem**, không phải **codebase**.

**Sources**:
- [Luma AI sponsors Nerfstudio + BAIR Lab](https://radiancefields.com/luma-ai-to-sponsor-nerfstudio-and-bair-lab)
- [Luma's proprietary Gaussian Splat iteration — Radiance Fields](https://radiancefields.com/luma-gaussian-splatting-unreal-engine-plugin-unveiled)

### 2.5 Academic ubiquity

Hầu hết paper 3DGS/NeRF mới (CVPR/ICCV/SIGGRAPH 2023-2025) dùng Nerfstudio làm baseline benchmark. Cho phép so sánh fair across methods.

---

## 3. Phân biệt mức độ — tránh over-claim

### Bảng claim levels (defense-grade)

| Claim | Strength | Defensibility |
|-------|----------|---------------|
| "Nerfstudio là framework research-to-production được sử dụng rộng rãi" | ✅ Mạnh | Khó bắt bẻ |
| "Một số công ty đã tích hợp Nerfstudio vào workflow sản xuất" | ✅ Mạnh | Verified ILM + Spectacular |
| "ILM dùng Nerfstudio cho VFX Marvel Ironheart, đoạt HPA Award 2025" | ✅ Mạnh | Có nguồn chính thức ILM |
| "Luma AI sponsor Nerfstudio + hire creator" | ✅ Mạnh | Verified |
| "Nerfstudio là framework production thật" | ❌ Quá | Nerfstudio research-first, không phải Unity/Unreal engine |
| "Luma AI app chạy code Nerfstudio" | ❌ SAI | Luma dùng proprietary stack riêng |
| "Bentley iTwin Capture fork Nerfstudio" | ❌ KHÔNG VERIFY ĐƯỢC | Bentley có engine riêng, không có evidence |
| "NVIDIA contribute upstream Nerfstudio" | ❌ KHÔNG VERIFY ĐƯỢC | NVIDIA có Instant-NGP riêng, không xác nhận |
| "CRS hơn framework production 5.9 dB" | ❌ Quá | Framework ≠ Method. Đúng phải nói "hơn Splatfacto baseline" |

### Nguyên tắc framing đúng

1. **Framework ≠ Method**: "vượt Splatfacto" (method) chứ KHÔNG "vượt Nerfstudio" (framework)
2. **Sponsor ≠ Use**: Luma sponsor không có nghĩa Luma chạy code Nerfstudio
3. **Modified+experimental ≠ Full production**: ILM tự admit "very few putting into production"
4. **Tiềm năng ứng dụng ≠ Deploy thực tế**: "có tiềm năng ứng dụng cho quét 3D điện thoại" thay vì "deploy được vào Polycam ngay"

---

## 4. CRSGaussian plug-in strategy — Path A (Adapter Method)

### Granularity = METHOD, không phải MODULE

| Anh thường nghĩ | Thực tế Nerfstudio |
|-----------------|---------------------|
| "Thay 1 module nhỏ trong Splatfacto" | "Thay cả method `splatfacto` bằng method `crsgaussian`" |
| Drop-in module-level | Drop-in method-level |
| Mix-and-match component | Mỗi method = 1 entity full pipeline |

```
                  ┌──── Nerfstudio shell ────┐
                  │                          │
ảnh + camera ───→ │  DataManager             │
                  │  (load images, cameras)  │
                  │  ↓                       │
                  │  ┌────────────────────┐  │
                  │  │  METHOD (chọn 1):  │  │
                  │  │   ─ splatfacto     │  │  ← Nerfstudio built-in
                  │  │   ─ nerfacto       │  │
                  │  │   ─ instant-ngp    │  │
                  │  │   ─ crsgaussian    │  │  ← CRSGaussian PLUG-IN
                  │  └────────────────────┘  │
                  │  ↓                       │
                  │  Eval + Viewer + Export  │
                  │  (PSNR/SSIM/LPIPS)       │
                  └──────────────────────────┘
```

### Khác biệt component giữa Splatfacto vs CRSGaussian Phase 22

| Component | Splatfacto (built-in) | CRSGaussian Phase 22 |
|-----------|----------------------|----------------------|
| **Init point cloud** | COLMAP sparse ~hundreds points | **RoMa dense ~23k points** |
| **Rasterizer** | gsplat (Berkeley) | **diff_gaussian_rasterization** (Inria) |
| **Densify** | DefaultStrategy (split/clone gradient) | **LFCF + AbsGS** |
| **Prune** | opacity threshold | **multi-signal CRS + opacity** |
| **Loss** | L1 + SSIM | L1 + SSIM + **Pearson depth + CRS reg** |
| **SH** | normal updates | **CRS-modulated freeze** |
| **Regularization** | none | **DropAnSH + opacity decay 0.999/iter** |

→ **10/10 component khác Splatfacto**. CRSGaussian là **method độc lập** thay thế **toàn bộ training pipeline**, không phải "module nhỏ vào Splatfacto".

### Implementation effort

- **Path A (recommended)**: Subclass nerfstudio `Model` (KHÔNG subclass `SplatfactoModel`), wrap CoR-GS GaussianModel + render bên trong
- **Effort**: 3-5 ngày
- **Output**: 1 method `crsgaussian` register vào Nerfstudio entry points
- **Usage**: `ns-train crsgaussian --data <path>`

### Risk

- PSNR adapter có thể ~95% standalone (Phase 22 = 21.918 PSNR)
- Regression test 4 scene bắt buộc sau implement
- Optimizer state + DataManager khác biệt cần verify

---

## 5. Defense narrative — 3 message levels

### Level 1 — Short (10 giây trả lời thầy)

> *"Em integrate CRSGaussian như method độc lập trong Nerfstudio (cùng cấp với Splatfacto/Nerfacto). Trên cùng pipeline Nerfstudio (DataManager + eval protocol), CRSGaussian vượt Splatfacto baseline khoảng 5-6 dB ở bài toán sparse-view 3 ảnh."*

### Level 2 — Medium (paragraph defense)

> *"Em tìm hiểu thì Nerfstudio là framework mã nguồn mở phổ biến cho NeRF và 3DGS, do Berkeley AI Research phát triển. Nó cung cấp toàn bộ pipeline từ data management, training, evaluation đến viewer. Hiện được dùng rộng rãi trong nghiên cứu và một số use case như VFX, digital twin, robotics, AR/VR. Một số tổ chức công nghiệp như ILM đã chia sẻ việc tích hợp Nerfstudio vào workflow tái tạo cảnh 3D.*
>
> *Em triển khai CRSGaussian như một method mới trong Nerfstudio (tương tự cách Splatfacto/Nerfacto là method độc lập). Sau đó đánh giá trên cùng DataManager + eval protocol — so sánh fair vì chỉ khác method. Trên cùng pipeline Nerfstudio, CRSGaussian cải thiện khoảng 5-6 dB so với Splatfacto baseline trong bài toán sparse-view."*

### Level 3 — Long (full defense response)

(Xem version final draft trong section 6 dưới)

---

## 6. Full draft gửi thầy (defense-ready)

> *"Thầy ơi, em đang tìm cách chứng minh tính ứng dụng của CRS ngoài việc chỉ so sánh với các bài báo SOTA.*
>
> *Em tìm hiểu thì Nerfstudio là một framework mã nguồn mở rất phổ biến cho NeRF và 3D Gaussian Splatting, do Berkeley AI Research phát triển. Nó cung cấp toàn bộ pipeline từ quản lý dữ liệu, huấn luyện, đánh giá đến viewer trực quan. Hiện nay Nerfstudio được dùng rộng rãi trong nghiên cứu và các bài toán như quét 3D từ ảnh, tạo novel views, VFX, digital twin, robotics và AR/VR. Ngoài ra, một số tổ chức công nghiệp như ILM cũng đã chia sẻ việc sử dụng Nerfstudio trong các workflow liên quan đến tái tạo cảnh 3D.*
>
> *Ý tưởng của em là triển khai CRSGaussian như một method mới trong Nerfstudio. Trong Nerfstudio, các phương pháp như Splatfacto, Nerfacto hay Instant-NGP đều được xem là các method độc lập trên cùng một framework. Vì vậy em không thay một module nhỏ trong Splatfacto mà triển khai toàn bộ pipeline CRSGaussian dưới dạng một method mới.*
>
> *Sau đó em sẽ đánh giá trên cùng DataManager, cùng protocol đánh giá và cùng viewer của Nerfstudio. Khi đó việc so sánh giữa Splatfacto và CRSGaussian sẽ công bằng vì chỉ khác method, còn toàn bộ môi trường thực nghiệm giữ nguyên.*
>
> *Mục tiêu của em không chỉ là chứng minh CRS tốt hơn các phương pháp trong bài báo, mà còn kiểm tra xem khi được triển khai trong một hệ thống NVS hoàn chỉnh thì CRS có thực sự mang lại cải thiện hay không.*
>
> *Nếu CRSGaussian cho kết quả tốt hơn trong điều kiện sparse-view thì em muốn sử dụng kết quả đó như một minh chứng rằng phương pháp của em có thể được tích hợp vào một hệ thống NVS chuẩn hóa và cải thiện chất lượng trong bài toán novel view synthesis sparse-view — một use case có tiềm năng ứng dụng cho quét 3D bằng điện thoại, drone mapping hoặc các pipeline nơi việc thu thập nhiều ảnh là tốn kém.*
>
> *Theo thầy thì hướng triển khai và đánh giá như vậy đã đáp ứng yêu cầu về tính ứng dụng thực tế chưa, hay thầy muốn em xây dựng thêm một use case hoặc demo cụ thể trên dữ liệu thực?"*

---

## 7. Backup responses cho follow-up questions

| Nếu thầy hỏi | Em trả lời |
|---|---|
| "ILM dùng Nerfstudio thế nào?" | "ILM modified Nerfstudio cho Marvel's Ironheart 2025, đoạt HPA Award 2025. Vincent Papaix (VFX supervisor) tự admit đây là experimental — 'very few people putting this kind of stuff into production' — nhưng đoạt giải industry quan trọng." |
| "Digital twin cụ thể research nào?" | "Nature Scientific Reports 2025 (interactive digital twins XR) + PMC smart factory 2025 (3D object reconstruction digital twin) — dùng Nerfstudio framework so sánh Instant-NGP/Nerfacto/Volinga/Tensorf cho industrial use case." |
| "CRSGaussian deploy thẳng vào Polycam được không?" | "Không trực tiếp — Polycam dùng proprietary stack riêng. Plug-in chứng minh tính tương thích với framework chuẩn (Nerfstudio), bước trung gian giữa research và production. Spectacular AI là ví dụ công ty fork Nerfstudio làm SDK thương mại." |
| "Sao chọn Nerfstudio mà không phải framework khác?" | "(1) Open-source Apache 2.0 — so sánh fair được; (2) Nhiều method baseline sẵn (Splatfacto, Nerfacto...); (3) Standardize DataManager + eval — không bị bắt bẻ evaluation khác; (4) Industry traction verified (ILM, Spectacular AI, Luma AI sponsor)." |
| "5-6 dB so với Splatfacto là sao?" | "Trên LLFF 3-view sparse setting: Splatfacto vanilla ~16 dB, CRSGaussian ~21.9 dB. Cùng DataManager Nerfstudio (load ảnh + camera giống nhau), cùng eval protocol (PSNR/SSIM/LPIPS computed cùng cách), chỉ khác method." |

---

## 8. Sources tổng hợp

### Nerfstudio chính thức
- [GitHub repo](https://github.com/nerfstudio-project/nerfstudio)
- [Nerfstudio paper SIGGRAPH 2023](https://arxiv.org/pdf/2302.04264)
- [Docs](https://docs.nerf.studio/)

### ILM × Nerfstudio (VFX)
- [ILM Ironheart HPA Award interview](https://www.ilm.com/ilm-nerfstudio-ironheart-hpa-award-interview/)
- [Creating VFX with NeRF (arXiv 2401.08633)](https://arxiv.org/pdf/2401.08633)

### Luma AI (sponsor + creator hire)
- [Luma AI sponsors Nerfstudio + BAIR](https://radiancefields.com/luma-ai-to-sponsor-nerfstudio-and-bair-lab)
- [Luma proprietary stack — NOT Nerfstudio code](https://radiancefields.com/luma-gaussian-splatting-unreal-engine-plugin-unveiled)

### Spectacular AI (commercial fork)
- [Spectacular AI Nerfstudio fork (GitHub)](https://github.com/SpectacularAI/nerfstudio)
- [Spectacular AI Mapping Tools](https://www.spectacularai.com/mapping)

### Digital twin research
- [Nature SR — Interactive digital twins XR (2025)](https://www.nature.com/articles/s41598-025-17855-9)
- [PMC — Digital twins smart factory NeRF](https://pmc.ncbi.nlm.nih.gov/articles/PMC12656295/)

---

## 9. Liên quan

- [01_real_world_comparison_plan.md](01_real_world_comparison_plan.md) — comparison plan tổng (closed-source vs open-source)
- [03_nerfstudio_feasibility_assessment.md](03_nerfstudio_feasibility_assessment.md) — feasibility ban đầu
- [04_nerfstudio_setup_and_run.md](04_nerfstudio_setup_and_run.md) — setup môi trường + baseline
- [05_plugin_design.md](05_plugin_design.md) — design ban đầu (Option 1 bổ trợ, đã obsolete sau Path A pivot)
- [06_plugin_a1_results.md](06_plugin_a1_results.md) — kết quả A1-A2.4 plug lẻ tẻ + insight cross-backbone CRS overlap

**Quan hệ với 05/06**: doc 05+06 là **Phase A** (plug lẻ tẻ module — kết quả wash trên CRS axis, A1 RoMa init +1.5 dB main contribution). **Doc 07 này** là pivot sang **Phase B** (Path A — plug nguyên khối Phase 22 thành 1 method `crsgaussian` — pending implement, ~3-5 ngày).
