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
