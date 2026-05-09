#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os


# ── [CRSGaussian T5.1] str2bool helper cho argparse ──
# ParamGroup dùng action="store_true" cho bool → không truyền False từ CLI được.
# Với args default=True (vd. crs_init_use_reproj), cần type=str2bool
# để hỗ trợ: --crs_init_use_reproj False
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', '0'):
        return False
    raise ValueError(f"Boolean value expected, got '{v}'")


class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    if value is True:
                        # [CRSGaussian T5.1] default=True → dùng str2bool để cho phép --key False
                        group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=str2bool)
                    else:
                        group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    if value is True:
                        # [CRSGaussian T5.1] default=True → dùng str2bool để cho phép --key False
                        group.add_argument("--" + key, default=value, type=str2bool)
                    else:
                        group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.n_views = 0
        self.rand_pcd = False
        self.n_sparse = -1
        # ── [CRSGaussian T1.3] Depth prior flags ──
        # Default False: training chạy như CoR-GS gốc, không cần DAV2
        # True: precompute depth prior + align với COLMAP trước training loop
        self.use_depth_prior = False
        self.dav2_path = "../Depth-Anything-V2"
        self.dav2_encoder = "vitl"
        # ── [CRSGaussian T5.1] Informed CRS₀ Initialization ──
        # Dùng geometry info có sẵn sau alignment để set CRS₀ có ý nghĩa
        # hình học, thay vì neutral 0.5. Xem docs/09_informed_crs_init_plan.md
        # Master switch: False → behavior cũ (CRS₀=0.5 tất cả)
        self.informed_crs_init = False
        # Component switches — mỗi cái bật/tắt độc lập cho ablation.
        # Default=True: khi bật informed_crs_init, dùng cả 3 signals.
        # Dùng str2bool: --crs_init_use_reproj False để tắt từ CLI.
        self.crs_init_use_reproj = True   # q_reproj: SfM reprojection quality
        self.crs_init_use_depth  = True   # q_depth:  DAV2-COLMAP depth agreement
        self.crs_init_use_view   = True   # q_view:   multi-view stereo support
        # Weights — tự normalize về sum=1 khi component bị tắt.
        # Grid search: w ∈ {0.0, 0.2, 0.33, 0.5, 0.8, 1.0}
        self.crs_init_w_reproj = 0.333    # weight cho q_reproj
        self.crs_init_w_depth  = 0.333    # weight cho q_depth
        self.crs_init_w_view   = 0.334    # weight cho q_view
        # Hyperparameters
        self.crs_init_tau_r = 2.5         # reproj error normalization. Ablate: {2.0, 2.5, 3.0}
        self.crs_init_gamma = 5.0         # logit scale factor. Ablate: {3.0, 5.0, 8.0}
        self.crs_init_eta   = 0.7         # densify inherit factor. Ablate: {0.5, 0.7, 0.9}
        # Densify inherit — riêng biệt với informed_crs_init.
        # True: child CRS₀ = clip(η * CRS_parent, 0, 0.5)
        # False (default): child CRS₀ = 0.5 (neutral, behavior cũ)
        self.crs_densify_inherit = False
        # ── [CRSGaussian Pseudo-depth] Pseudo-view depth loss ──
        # Ép Gaussian model đúng geometry ở novel views (giảm overfit train-test gap).
        # Forward warp aligned DAV2 depth từ training cam gần nhất → pseudo-cam
        # → so với rendered depth tại pseudo-cam (Pearson loss).
        # Master switch: False → behavior cũ (không có pseudo depth loss).
        # Cần --use_depth_prior=True (để có aligned_depth_dict làm reference).
        self.use_pseudo_depth_loss   = False    # master switch
        self.lambda_pseudo_depth     = 0.05     # loss weight (như fixed depth loss)
        self.pseudo_depth_start_iter = 2000     # bắt đầu sau warmup, sau khi geometry sơ bộ ổn
        self.pseudo_depth_ramp_iters = 500      # linear ramp tránh shock
        self.pseudo_depth_interval   = 5        # mỗi N iter (tránh tính mỗi iter cho rẻ)
        # ── [CRSGaussian Pseudo-photo] Pseudo-view photometric consistency (Approach 2) ──
        # Forward warp GT IMAGE từ training cam → pseudo-cam (dùng aligned depth)
        # → so với rendered image tại pseudo-cam (L1 masked loss).
        # Khác pseudo depth: signal photometric mạnh hơn, detect floater qua parallax.
        # Cần --use_depth_prior=True (cần aligned depth để warp).
        self.use_pseudo_photo_loss    = False   # master switch
        self.lambda_pseudo_photo      = 0.01    # loss weight — bắt đầu thấp
        self.pseudo_photo_start_iter  = 1000    # bắt đầu sau warmup
        # ── [CRSGaussian Phase 2c] Opacity decay (inspired by Binocular3DGS) ──
        # Multiply opacity mỗi iter để tạo continuous pressure → zombie
        # Gaussian giảm opacity dần → bị CRS/legacy pruning loại tự nhiên.
        # Default OFF → baseline không đổi.
        self.use_opacity_decay           = False   # master switch
        self.opacity_decay_factor        = 0.995   # multiply per iter
        self.opacity_decay_extend_densify = False  # extend densify_until_iter = iterations
        # ── [CRSGaussian Tier A] Formula-level diagnostics ──
        # Bật dump D_i, R_i, CRS distributions + 2 test bổ sung (synthetic
        # floater discrimination + occlusion contamination) để diagnose
        # bản thân công thức CRS có information value không.
        # Default OFF — chỉ bật trong session diagnostic, không ảnh hưởng training.
        # KHÔNG sửa CRS update / pruning behavior — pure read-only instrumentation.
        self.tier_a_diag       = False
        self.tier_a_diag_iters = "1100,3000,5000,10000"   # match update_crs schedule (>T_warmup AND %100==0)

        # ── [CRSGaussian Phase 11 Step 2] Same-view perceptual loss DINOv2 ──
        # DINO patch features distance giữa render và GT cùng cam. Self-
        # supervised → robust hơn LPIPS-VGG (ImageNet supervised). Compute
        # mỗi N iter (freq) để giảm cost; gradient flow ngược về image_render.
        # Default OFF — backward compat.
        self.use_perceptual_dino             = False
        self.lambda_perceptual_dino          = 0.05
        self.perceptual_dino_start_iter      = 1500     # sau geometry sơ bộ
        self.perceptual_dino_freq            = 5        # mỗi 5 iter compute
        self.perceptual_dino_mode            = "cosine"   # "cosine" | "l1"
        self.perceptual_dino_crs_weight      = False    # weight per-patch bởi CRS_pix

        # ── [CRSGaussian Phase 11 Step 3] R_feature replace R_visible ──
        # Drop-in replacement R_visible: pairwise DINO patch sim cross-view tại
        # Gaussian projections. Gating xảy ra trong update_crs(). Default OFF
        # → R_visible (Phase 8) vẫn dùng.
        self.use_r_feature                   = False

        # ── [CRSGaussian Phase 11 Step 4] Cross-view Feature MPC ──
        # Forward warp F_B từ neighbor cam_B sang cam_A bằng aligned depth +
        # camera geometry, so với F_A_render. Cross-view consistency tại
        # feature level. Cần aligned_depth_dict + DINO. Default OFF.
        self.use_feature_mpc                 = False
        self.lambda_feature_mpc              = 0.05
        self.feature_mpc_start_iter          = 2000     # later than Step 2 (geometry stable)
        self.feature_mpc_freq                = 10       # cost reduction
        self.feature_mpc_min_valid_frac      = 0.3      # skip nếu < 30% patch valid

        # ── [CRSGaussian Phase 11 Step 1] Depth-based covisibility reweight ──
        # Per-pixel weight cho L_phot từ (a) covisibility — số views thấy điểm 3D
        # tại pixel (forward-warp aligned DAV2 depth) + (b) optional combine
        # với CRS_pix render. Reuse aligned_depth_dict — KHÔNG cần external
        # foundation model (DUSt3R/MASt3R).
        # Prerequisites: --use_depth_prior=True (cần aligned depth).
        # Default OFF → behavior y hệt Phase 8 FULL (weight_map fall back path
        # cũ, hoặc Phase 7 reweighter nếu use_loss_reweight=True).
        self.use_coreliability_reweight          = False
        self.coreliability_gamma                 = 0.3      # weight floor ∈ [γ, 1]
        self.coreliability_combine_crs           = False    # combine với CRS_pix render
        self.coreliability_combine_mode          = "min"    # "min" | "mean"
        self.coreliability_depth_consistency_thr = 0.05     # 5% relative depth threshold

        # ── [CRSGaussian Phase 10A] Dense init via DUSt3R ──
        # Replace/augment COLMAP sparse PC (~3000 pts) bằng DUSt3R dense PC (~30k+ pts)
        # at scene init. Foundation model PREDICTIVE (not generative) → predict 3D
        # point map từ image pairs, align với COLMAP poses.
        # CoMapGS (CVPR 2025) precedent: MASt3R-based init +0.65 dB.
        # Inference chạy 1-lần qua scripts/precompute_dust3r.py → cache .npz/scene.
        # Training-time wrapper CHỈ load cache + filter + (optional) merge với COLMAP.
        # Default OFF — backward compat.
        self.use_dense_init             = False
        self.dust3r_cache_dir           = "cache/dust3r_init"   # nơi precompute_dust3r.py ghi <scene>.npz
        self.dense_init_mode            = "augment"   # "augment" (COLMAP + DUSt3R) | "replace" (DUSt3R only)
        self.dense_init_conf_threshold  = 1.5         # DUSt3R log-confidence threshold (loại points yếu)
        self.dense_init_max_points      = 50000       # subsample limit (avoid OOM)
        self.dense_init_dedupe_radius   = 0.01        # KDTree merge radius — fraction of scene_extent

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.use_confidence = False
        self.use_color = True

        # ── [CRSGaussian Track B] Dropout regularization ──
        # Master switch + per-knob flags để ablate độc lập.
        # Default OFF: use_dropout=False → toàn bộ gating block trong renderer
        # bị skip → kết quả MATCH baseline A1 (reproduce exactly).
        # Mode: "uniform" (B1, Co-Adapt) | "sh_norm" (B3) | "hybrid" (B4).
        # Các field current_iter / train_mode gán dynamic trong train.py.
        self.use_dropout        = False
        self.dropout_mode       = "uniform"
        self.dropout_base       = 0.1
        self.dropout_w_crs      = 0.0
        self.dropout_w_sh       = 0.0
        self.dropout_max        = 0.6
        self.dropout_start_iter = 0

        # ── [CRSGaussian DropAnSH] Anchor + SH degree dropout ──
        # Reproduce DropAnSH-GS technique. Default OFF → baseline không đổi.
        # Combined với B1 dropout qua AND mask trong renderer.
        # Schedule truyền qua 3 int riêng (sched0/1/2) thay list vì ParamGroup
        # không auto-handle list type với argparse.
        self.use_dropansh         = False
        self.dropansh_pa          = 0.02
        self.dropansh_k           = 10
        self.dropansh_psh         = 0.2
        self.dropansh_sched0      = 2000
        self.dropansh_sched1      = 4000
        self.dropansh_sched2      = 6000
        self.dropansh_total_iter  = 10000
        self.dropansh_crs_anchor  = False   # Phase 3: weighted anchor sampling
        self.dropansh_crs_sh      = False   # Phase 3: CRS-modulated SH dropout

        # ── [CRSGaussian Phase 2d Stage A] Density-aware anchor sampling ──
        # Anchor weighted theo local density thay vì random.
        #   "uniform"    → behavior cũ, random anchor (default, baseline)
        #   "voxel"      → weighted bởi voxel bin count (cheap, topology-only)
        #   "covariance" → weighted bởi sum Bhattacharyya overlap với k-NN (shape-aware)
        #   "crs"        → Phase 3α: weighted bởi (1-CRS) quality signal
        #   "crs_voxel"  → Phase 3β: voxel × (1-CRS) composite
        # crs_anchor (Phase 3 old flag) priority hơn density_method khi cả 2 bật.
        self.dropansh_density_method = "uniform"

        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.prune_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.prune_threshold = 0.005
        self.start_sample_pseudo = 2000
        self.end_sample_pseudo = 10000
        self.sample_pseudo_interval = 1
        self.dist_thres = 10.
        self.random_background = False
        self.absdensify = False
        # ── [CRSGaussian T4.2] CRS pruning params ──
        self.T_warmup = 1000        # iter bắt đầu CRS active. Ablate: {500, 1000, 2000}
        self.tau_crs = 0.35         # ngưỡng CRS để prune. Ablate: {0.25, 0.35, 0.45}
        self.tau_densify = 0.45     # ngưỡng CRS để chặn densify (Phase 4 future)
        self.tau_isolated = 0.1     # ngưỡng isolation (× scene_extent). Ablate: {0.05, 0.10, 0.20}
        self.use_pos_constraint = False  # [debug] bật/tắt position constraint (T4.1)
        self.use_crs_pruning = False     # [debug] bật/tắt CRS pruning (T4.2)
        # ── [CRSGaussian] CRS update hyperparameters ──
        self.crs_ema_decay = 0.9         # EMA decay cho CRS update. Ablate: {0.5, 0.7, 0.9}
        self.crs_update_interval = 100   # Mỗi bao nhiêu iter update CRS. Ablate: {25, 50, 100}
        # ── [CRSGaussian Hướng D MVP] Render-Native Reliability Coupling ──
        # 3-layer RNRC fix:
        #   L1 (RC): proxy RC từ render (opacity × max_radii × coverage) — MVP dùng proxy.
        #   L2 (continuous EMA β=0.99): update CRS mỗi iter thay vì snapshot 100-iter.
        #   L3 (differentiable α-coupling): α_eff = α × CRS.detach() trong forward render.
        # Mode "L1g":  L1 + snapshot mỗi 100 iter + CRS pruning gate
        # Mode "L12g": L1 + L2 (continuous EMA) + CRS pruning gate
        # Mode "full": L1 + L2 + L3 (differentiable, no separate prune gate)
        # Default OFF → behavior gốc không đổi.
        self.use_rnrc                  = False
        self.rnrc_mode                 = "full"        # "L1g" | "L12g" | "full"
        self.rnrc_beta                 = 0.99          # EMA factor cho continuous
        self.rnrc_warmup               = 1000          # global warmup, iters đầu CRS=1
        self.rnrc_per_gauss_warmup     = 200           # per-Gaussian warmup sau spawn
        self.rnrc_floor                = 0.5           # CRS floor trong warmup window
        self.rnrc_norm_mode            = "median"      # "median" | "p75"
        self.rnrc_tau                  = 0.35          # prune threshold cho L1g/L12g modes
        # ── [CRSGaussian DIAG E1] Freeze SH hyperparameter ──
        # Iter sau đó SH (f_dc + f_rest) bị freeze (lr=0). Chỉ xyz/opacity/
        # scaling/rotation tiếp tục update.
        # Default 0 → KHÔNG freeze (behavior cũ). > 0 → freeze tại iter này.
        # Dùng để test H4: "SH overfit memorize training views là nguyên nhân
        # chính của train-test gap 16dB".
        # Ablate: {0, 3000, 5000, 7000}
        self.freeze_sh_after = 0

        # ── [CRSGaussian DIAG A2] Freeze chỉ DC (f_dc), f_rest tự do ──
        # Isolate DC contribution vs rest trong SH overfit.
        # freeze_dc_only=True → freeze_dc() được gọi tại freeze_dc_start_iter.
        self.freeze_dc_only = False
        self.freeze_dc_start_iter = 1000

        # ── [CRSGaussian Tier 2-min] D_cycle + Loss reweighter ──
        # Last CRS attempt sau 7 mechanism variants ceiling +0.07 dB.
        # Dual upgrade: Signal D → D_cycle (no DAV2, multi-view internal),
        #               Mechanism gate prune → per-pixel loss reweighter (loss path).
        # Default OFF — backward compat (Rule 11). Verify với flag OFF
        # phải byte-identical baseline.
        # Layer 1 — Signal upgrade (D_cycle replace D_DAV2)
        self.use_d_cycle         = False    # master switch: thay D_DAV2 bằng D_cycle
        self.d_cycle_warmup      = 1000     # iters đầu vẫn dùng D_DAV2 (scene chưa converge)
        self.d_cycle_sigma       = 5.0      # cycle error normalization (pixels). Ablate {3,5,8}
        self.d_cycle_update_freq = 100      # mỗi N iter compute D_cycle (cache giữa các update)
        # Layer 2 — Mechanism upgrade (per-pixel loss reweighter)
        self.use_loss_reweight   = False    # master switch: weight L_recon bằng CRS_pix
        self.lossw_gamma         = 0.5      # w(p) ∈ [γ, 1] khi CRS_pix ∈ [0, 1]. Ablate {0.3,0.5,0.7}
        self.lossw_render_freq   = 100      # mỗi N iter re-render CRS map (cache giữa các update)

        # ── [CRSGaussian Phase 8a] R_visible — visibility-aware reprojection consistency ──
        # Fix R contamination 36.5% (Tier A4): chỉ aggregate views nơi Gaussian
        # thực sự visible (gauss_z ≤ rendered_z × tolerance). Reuse render depth
        # maps pattern từ d_cycle. Default OFF.
        self.use_r_visible                = False
        self.r_visible_occlusion_tolerance = 1.05   # gauss_z ≤ rendered_z × tolerance
        self.r_visible_min_views          = 2       # min visible views, else neutral 0.5

        # ── [CRSGaussian Phase 8b] S_stability — SH coefficient stability ──
        # Track features_rest variance qua EMA → detect SH drift (memorize vs stable).
        # NEW signal dimension cho CRS formula (3-component: D + R + S).
        # Default OFF.
        self.use_sh_reliability   = False    # master switch
        self.sh_stability_ema_beta = 0.95    # EMA decay (higher = slower update)
        self.sh_stability_warmup  = 1000     # iters đầu skip (SH chưa stable)
        self.crs_w_s              = 0.33     # weight w_s; auto-norm w_d=w_r khi S OFF

        # ── [CRSGaussian Phase 8c] CRS-modulated SH freeze ──
        # Per-Gaussian thay thế global freeze_sh_after. CRS thấp → freeze SH grad.
        # Default OFF (giữ behavior cũ với freeze_sh_after).
        self.use_crs_modulated_sh_freeze = False
        self.crs_freeze_tau              = 0.5    # CRS < tau → zero _features_rest grad
        self.crs_freeze_start            = 1000   # iter bắt đầu apply selective freeze

        # ── [CRSGaussian Phase 9] D-only formula + cross-backbone test ──
        # Phase 8 attribution: R alone HURTS (-0.111), S adds nothing (-0.023),
        # SH freeze là hero (+0.189). Phase 9 test simplification: drop R (D-only)
        # và cross-backbone (CRS-mod freeze trên A1+B1β thay global freeze).
        # Default OFF (giữ Phase 8 behavior).
        self.disable_r_signal             = False    # True → w_r=0, skip R compute (D-only formula)
        self.disable_global_sh_freeze     = False    # True → bypass freeze_sh_after, dùng CRS-mod freeze thay

        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
