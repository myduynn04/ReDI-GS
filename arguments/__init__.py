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

"""
    
Convention: attribute nào trong __init__ có tên bắt đầu bằng _ thì tự động được đăng ký thêm shorthand flag = - + chữ cái đầu của tên.
"""

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
        # ── [CRSGaussian Phase 2c] Opacity decay (inspired by Binocular3DGS) ──
        # Multiply opacity mỗi iter để tạo continuous pressure → zombie
        # Gaussian giảm opacity dần → bị CRS/legacy pruning loại tự nhiên.
        # Default OFF → baseline không đổi.
        self.use_opacity_decay           = False   # master switch
        self.opacity_decay_factor        = 0.995   # multiply per iter
        self.opacity_decay_extend_densify = False  # extend densify_until_iter = iterations
        # ── [CRSGaussian Phase 37 H1] Q_init từ RoMa — CỜ NẠP DỮ LIỆU ──
        # Đặt ở ModelParams vì Scene.__init__ chỉ nhận ModelParams, và đây là
        # chuyện NẠP dữ liệu. Cờ SỬ DỤNG (w_Q, center, log) nằm ở
        # OptimizationParams vì update_crs nhận `opt`.
        #
        # Cần sidecar <data>/<scene>/<n>_views/dense/fused.romav1.qinit.npz,
        # VÀ fused.ply đang dùng phải là bản sinh cùng lượt (romav1_p37) —
        # lệch chiều dài thì Scene raise chứ không chạy tiếp. Xem docs/37.
        #
        # S1 8/8 scene: certainty BÃO HOÀ (p05 = 1.000 mọi scene) → w_cert = 0.
        self.use_roma_qinit = False   # master switch
        self.qinit_w_cert   = 0.0     # trọng số certainty — S1 chứng minh vô dụng
        self.qinit_w_reproj = 1.0     # trọng số reprojection quality
        # ── [CRSGaussian Tier A] Formula-level diagnostics ──
        # Bật dump D_i, R_i, CRS distributions + 2 test bổ sung (synthetic
        # floater discrimination + occlusion contamination) để diagnose
        # bản thân công thức CRS có information value không.
        # Default OFF — chỉ bật trong session diagnostic, không ảnh hưởng training.
        # KHÔNG sửa CRS update / pruning behavior — pure read-only instrumentation.
        self.tier_a_diag       = False
        self.tier_a_diag_iters = "1100,3000,5000,10000"   # match update_crs schedule (>T_warmup AND %100==0)

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

        # ── [CRSGaussian Phase 17 — C1] DSINE monocular-normal prior ──
        # External-prior thứ 2 (song song DAV2-depth). Loss C1a: L1 giữa
        # normal-từ-rendered-depth (differentiable, camera-[0,1]) và
        # DSINE-normal precompute (scripts/p17_c1_preprocess_dsine.py →
        # <source_path>/c1_normal_dir/<stem>.npy). HONEST: C1a là
        # adaptation (dn-splatter detach depth + supervise CUDA C1b);
        # Default OFF → không ảnh hưởng recipe production.
        self.use_c1_normal       = False     # master switch
        self.c1_normal_lambda    = 0.10      # primary pre-registered λ
        self.c1_normal_start_iter = 0        # iter bắt đầu áp loss
        self.c1_normal_dir       = "c1_dsine_normals"  # subdir under source_path

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
        self.T_warmup = 1000        # iter bắt đầu CRS active
        self.tau_crs = 0.35         # ngưỡng CRS để prune
        self.tau_densify = 0.45     # ngưỡng CRS để chặn densify (Phase 4 future)
        self.tau_isolated = 0.1     # ngưỡng isolation (× scene_extent)
        self.use_pos_constraint = False  # [debug] bật/tắt position constraint (T4.1)
        # ── [CRSGaussian] CRS update hyperparameters ──
        self.crs_ema_decay = 0.9         # EMA decay cho CRS update
        self.crs_update_interval = 100   # Mỗi bao nhiêu iter update CRS
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
        self.d_cycle_sigma       = 5.0      # cycle error normalization (pixels)
        self.d_cycle_update_freq = 100      # mỗi N iter compute D_cycle (cache giữa các update)
        # Layer 2 — Mechanism upgrade (per-pixel loss reweighter)
        self.use_loss_reweight   = False    # master switch: weight L_recon bằng CRS_pix
        self.lossw_gamma         = 0.5      # w(p) ∈ [γ, 1] khi CRS_pix ∈ [0, 1]
        self.lossw_render_freq   = 100      # mỗi N iter re-render CRS map (cache giữa các update)


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
        self.crs_freeze_tau              = 0.65   # [Phase 28] tuned default (was 0.5); CRS < tau → zero _features_rest grad
        self.crs_freeze_start            = 1000   # iter bắt đầu apply selective freeze

        # ── [CRSGaussian Phase 9] D-only formula + cross-backbone test ──
        # Phase 8 attribution: R alone HURTS (-0.111), S adds nothing (-0.023),
        # SH freeze là hero (+0.189). Phase 9 test simplification: drop R (D-only)
        # và cross-backbone (CRS-mod freeze trên A1+B1β thay global freeze).
        # Default OFF (giữ Phase 8 behavior).
        self.disable_r_signal             = False    # True → w_r=0, skip R compute (D-only formula)
        self.disable_global_sh_freeze     = False    # True → bypass freeze_sh_after, dùng CRS-mod freeze thay

        # ── [CRSGaussian Phase 37 H1] Q_init — CỜ SỬ DỤNG ──
        # Cờ NẠP (use_roma_qinit / qinit_w_cert / qinit_w_reproj) ở ModelParams.
        # Đây là cờ điều khiển việc Q_init tác động vào logit CRS thế nào.
        #
        # qinit_w_in_crs = 0.0 → buffer VẪN đi qua prune/densify và VẪN được
        # dump ra q_init.npz, nhưng KHÔNG đụng logit → LOG-ONLY, chế độ của S2
        # (docs/37 §13.6). S3 mới bật lên 0.25.
        self.qinit_w_in_crs     = 0.0      # w_Q. 0 = LOG-ONLY
        self.qinit_center       = True     # trừ trung bình trước khi cộng — CHỐNG CONFOUND
        self.qinit_log_interval = 500      # log phân phối + tỉ lệ freeze mỗi N iter (0 = tắt)

        # ── [CRSGaussian Phase 13] LFCF densification (EFA-GS port) ──
        # Replace standard clone/split với tolerance-based decision +
        # diffscale isotropify + probabilistic split. Default OFF — behavior
        # identical Phase 8 FULL khi use_lfcf=False.
        self.use_lfcf                  = False    # master switch
        self.lfcf_init_scaling_max     = 1.5      # TaT default — enlarge ceiling
        self.lfcf_init_scaling_min     = 1.0      # depth-adaptive lower bound
        self.lfcf_last_scaling_max     = 1.0      # decay target tại densify_until
        self.lfcf_pow                  = 1.0      # decay rate (linear)
        self.lfcf_splitting_ub         = 1.0      # split prob upper bound
        self.lfcf_interval_times       = 2        # LFCF mỗi 2 × densify_interval
        self.lfcf_tolerance            = 1e-5     # FP-stability tolerance for grad compare
        self.lfcf_diffscale            = True     # volume-preserving isotropify

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
