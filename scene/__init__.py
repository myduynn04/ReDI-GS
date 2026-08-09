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

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.pose_utils import generate_random_poses_llff, generate_random_poses_360
from scene.cameras import PseudoCamera

class Scene:
    """Load COLMAP/Blender scene, dung camera list va do point cloud init
    vao GaussianModel. Metadata (input.ply, cameras.json) ghi ra model_path."""
    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        self.model_path = args.model_path # Lưu model_path (đường dẫn output folder) vào self để dùng trong save() sau này
        self.source_path = args.source_path  # Lưu source_path (đường dẫn data folder, vd: data/nerf_llff_data/fern) vào self
        print(f"args.source_path  is {args.source_path }")
        self.loaded_iter = None         # loaded_iter = None nghĩa là đang train mới từ đầu, không resume từ checkpoint
        # Nhận reference đến GaussianModel được tạo từ bên ngoài (train.py).
        # KHÔNG tạo GaussianModel mới ở đây — Scene và train.py dùng chung 1 object.
        self.gaussians = gaussians
        # [NOT USE] Xử lý trường hợp resume từ checkpoint.
        # Production luôn train mới (load_iteration=None) nên block này không chạy.
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # Key của dict là resolution_scale (thường chỉ có 1.0 = full resolution).
        self.train_cameras = {}   # 3 cameras dùng để train (LLFF 3-view)
        self.test_cameras = {}    # các cameras còn lại dùng để eval PSNR
        self.pseudo_cameras = {}  # [NOT USE] cameras ảo nội suy giữa train views — CoRGS legacy
        # bounds trong LLFF là [near_depth, far_depth] — khoảng cách gần nhất và xa nhất có vật thể trong scene, tính từ góc nhìn của camera.
        # Cụ thể, nó được load từ file poses_bounds.npy (file chuẩn của LLFF dataset)
        self.bounds = None        # [NOT USE] giới hạn không gian của scene, lấy từ camera đầu tiên

        # ── Nhận dạng loại dataset dựa vào cấu trúc thư mục ──
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            if args.source_path.find('llff') != -1:
                print("############ load llff ############")
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.n_views, rand_pcd=args.rand_pcd)
            elif args.source_path.find('mipnerf360') != -1:
                print("############ load mipnerf360 ############")
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.n_views, rand_pcd=args.rand_pcd)
            elif args.source_path.find('DTU') != -1:
                print("############ load DTU ############")
                scene_info = sceneLoadTypeCallbacks["DTU"](args.source_path, args.images, args.eval, args.n_views, rand_pcd=args.rand_pcd)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.n_views, rand_pcd=args.rand_pcd)
        else:
            assert False, "Could not recognize scene type!"

        # ── Lưu metadata ra output folder (chỉ khi train mới, không phải resume) ──
        if not self.loaded_iter:
            # Copy file fused.ply (point cloud init) vào output/input.ply
            # Mục đích: lưu lại init để sau này biết run đó dùng init nào (MVS hay RoMa v1)
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
                
            # Gom tất cả cameras (train + test) vào một list để xuất ra JSON, cụ thể ở cameras.json
            json_cams = []
            camlist = []
            
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)   # thêm test cameras vào trước. Do thêm trc nên trong json test camera là 0,1,2
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)  # thêm train cameras vào sau
            
            # Chuyển từng camera object sang dict JSON-serializable
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))

            # Lưu toàn bộ camera info ra output/cameras.json
            # Dùng để visualize hoặc load lại sau mà không cần re-parse COLMAP
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # [NOT USE] Shuffle thứ tự cameras.
        # Production truyền shuffle=False → block này không chạy.
        # Mục đích khi dùng: tránh model bị bias theo thứ tự camera của COLMAP.
        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)


        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(self.cameras_extent, 'cameras_extent')

        # ── Load cameras theo từng resolution scale ──
        # resolution_scale = 1.0 (multi-res của 3DGS gốc, bài KHÔNG dùng → loop 1 lần)
        # Việc giảm ảnh 8 lần là do cờ -r 8 (= args.resolution), KHÔNG phải resolution_scale.
        # Số chia thực tế = resolution_scale × args.resolution = 1.0 × 8 = 8.
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras", resolution_scale)
            # Chuyển CameraInfo (raw data từ COLMAP) thành Camera object (có tensor GPU)
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args) # resolution_scale của mình là -r 8 tức là giảm kích thước 8 lần

            print("Loading Test Cameras", resolution_scale)
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            
            # [NOT USE] Tạo pseudo cameras — CoRGS legacy.
            # Vẫn được tạo ra nhưng không có loss nào dùng đến trong production.
            pseudo_cams = []
            if args.source_path.find('llff') != -1:
                pseudo_poses = generate_random_poses_llff(self.train_cameras[resolution_scale])
            elif args.source_path.find('mipnerf360') != -1:
                pseudo_poses = generate_random_poses_360(self.train_cameras[resolution_scale])
            elif args.source_path.find('synthetic') != -1:
                pseudo_poses = generate_random_poses_360(self.train_cameras[resolution_scale])
            elif args.source_path.find('DTU') != -1:
                pseudo_poses = generate_random_poses_llff(self.train_cameras[resolution_scale])

            # [NOT USE]
            # Lấy camera đầu tiên làm template để copy FoV và image size
            view = self.train_cameras[resolution_scale][0]
            self.bounds = view.bounds # lưu bounds của scene từ camera đầu tiên
            for pose in pseudo_poses:
                pseudo_cams.append(PseudoCamera(
                    R=pose[:3, :3].T, 
                    T=pose[:3, 3], 
                    FoVx=view.FoVx, 
                    FoVy=view.FoVy,
                    width=view.image_width, height=view.image_height
                ))
            self.pseudo_cameras[resolution_scale] = pseudo_cams


        # ── Khởi tạo Gaussian model từ point cloud ──
        if self.loaded_iter:
            # [NOT USE] Resume từ checkpoint: load .ply đã train từ output folder.
            # Production không dùng nhánh này vì luôn train mới.
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                  "point_cloud",
                                                  "iteration_" + str(self.loaded_iter),
                                                  "point_cloud.ply"))
        else:
            # Train mới: khởi tạo Gaussians từ point cloud init (fused.ply = MVS hoặc RoMa v1).
            # Mỗi điểm trong point cloud → 1 Gaussian với vị trí, màu, opacity, scale, rotation ban đầu.
            # ── [CRSGaussian P37 H1] Nạp Q_init từ sidecar RoMa (nếu bật) ──
            # Sidecar do scripts/p37_romav1_preprocess_qinit.py sinh ra, nằm cạnh
            # fused.ply. Chỉ số của nó khớp TỪNG PHẦN TỬ với vertex của
            # fused.ply.romav1_p37 — nên fused.ply đang dùng PHẢI là bản đó.
            #
            # Cố ý KHÔNG nuốt lỗi: thiếu file hoặc lệch chiều dài → raise.
            # Lệch index mà chạy tiếp thì mọi kết quả sau đó sai âm thầm, đó là
            # loại lỗi khó phát hiện nhất. Thà dừng ngay.
            q_init_t = None
            if getattr(args, "use_roma_qinit", False):
                from utils.crs.qinit_roma import find_qinit_sidecar, load_q_init, to_gpu_buffer
                sc = find_qinit_sidecar(args.source_path, args.n_views)
                if sc is None:
                    raise FileNotFoundError(
                        f"[P37] use_roma_qinit=True nhưng không thấy sidecar tại "
                        f"{args.source_path}/{args.n_views}_views/dense/fused.romav1.qinit.npz\n"
                        f"  → chạy scripts/p37_romav1_preprocess_qinit.py trước."
                    )
                q_np = load_q_init(
                    sc,
                    w_cert=getattr(args, "qinit_w_cert", 0.0),
                    w_reproj=getattr(args, "qinit_w_reproj", 1.0),
                    expect_n=len(scene_info.point_cloud.points),
                )
                q_init_t = to_gpu_buffer(q_np)
                print(f"[P37] Q_init sidecar: {sc}")

            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent,
                                            q_init=q_init_t)
            self.init_point_cloud = scene_info.point_cloud # Lưu lại point cloud ban đầu để có thể phân tích sau này

    def save(self, iteration):
        # Lưu Gaussian model ra file .ply tại iter được chỉ định.
        # Được gọi khi iteration nằm trong saving_iterations (mặc định iter 10000).
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        # ── [CRSGaussian P37 H1] Dump Q_init + spawn_iter cạnh ply ──
        # KHÔNG nhét vào capture()/restore() — hai hàm đó unpack đúng 13 trường,
        # thêm trường sẽ phá mọi checkpoint cũ. File .npz riêng an toàn hơn.
        # S2 cần cả hai: q_init để đo tương quan, spawn_iter để lọc Gaussian
        # gốc (spawn_iter == 0) khỏi con cháu (được điền giá trị trung tính).
        q = getattr(self.gaussians, "_q_init", None)
        if q is not None:
            sp = getattr(self.gaussians, "spawn_iter", None)
            np.savez_compressed(
                os.path.join(point_cloud_path, "q_init.npz"),
                q_init=q.detach().cpu().numpy().reshape(-1),
                spawn_iter=(sp.detach().cpu().numpy().reshape(-1)
                            if sp is not None else np.zeros(q.shape[0], dtype=np.int32)),
            )

    def getTrainCameras(self, scale=1.0):
        # Trả về list training cameras ở resolution scale cho trước
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        # Trả về list test cameras ở resolution scale cho trước
        return self.test_cameras[scale]

    def getPseudoCameras(self, scale=1.0):
        # [NOT USE] Trả về list pseudo cameras — CoRGS legacy.
        # Nếu không có pseudo cameras nào → trả về [None] thay vì list rỗng.
        if len(self.pseudo_cameras) == 0:
            return [None]
        else:
            return self.pseudo_cameras[scale]
