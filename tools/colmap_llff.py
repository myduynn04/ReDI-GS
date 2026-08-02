# ============================================================
# [Khối COLMAP — Phase 1, OFFLINE] Sinh dữ liệu COLMAP cho LLFF sparse-view
# File: tools/colmap_llff.py
#
# Chạy 1 LẦN trước khi train, KHÔNG nằm trong training loop.
# Nhiệm vụ: từ ảnh gốc của mỗi scene, chọn 3 training view rồi chạy pipeline
# COLMAP (feature → match → triangulate → dense stereo) để sinh ra:
#   - camera poses (sparse/0/images.txt, cameras.txt)
#   - point cloud triangulated (3_views/triangulated/points3D)
#   - dense fused.ply (3_views/dense/fused.ply)  ← init cho Gaussian (MVS baseline)
#
# fused.ply này chính là cái mà scene/dataset_readers.py đọc lúc train.
# (Phương pháp của bạn sau đó THAY fused.ply MVS này bằng RoMa v1 — xem p22.)
# ============================================================
import os
import numpy as np
import sys
import sqlite3

IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

# ── Các câu lệnh SQL tạo bảng của COLMAP database (database.db) ──
# COLMAP lưu keypoints/descriptors/matches vào SQLite. Các hằng dưới đây là
# schema chuẩn của COLMAP — KHÔNG cần sửa, chỉ để tạo DB rỗng đúng format.

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


# Chuyển numpy array ↔ BLOB (binary) để lưu/đọc trong SQLite. Helper của COLMAP.
def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

# Lớp bọc SQLite connection của COLMAP — cung cấp các hàm tạo bảng + update camera.
class COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)
        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (model, width, height, array_to_blob(params), camera_id))
        return cursor.lastrowid

# Làm tròn theo kiểu Python 3 (round-half-to-even) để chọn index view nhất quán.
def round_python3(number):
    rounded = round(number)
    if abs(number - rounded) == 0.5:
        return 2.0 * round(number / 2.0)
    return rounded

# Chạy 1 lệnh shell (thường là lệnh colmap ...) + in ra exit code để debug.
def run_cmd(cmd):
    print(f"  [CMD] {cmd}")
    sys.stdout.flush()
    ret = os.system(cmd)
    print(f"  [RET] exit code = {ret}")
    sys.stdout.flush()
    return ret

# ⭐ Hàm chính — chạy toàn bộ pipeline COLMAP cho 1 scene, sinh ra fused.ply.
# scene = tên scene (fern...), base_path = thư mục data, n_views = số training view (3).
def pipeline(scene, base_path, n_views):
    print(f"\n{'='*60}")
    print(f"[SCENE] Processing: {scene}")
    print(f"{'='*60}")
    sys.stdout.flush()

    llffhold = 8                              # cứ 8 ảnh lấy 1 làm test (convention NeRF/LLFF)
    view_path = str(n_views) + '_views'       # tên folder output, vd "3_views"
    scene_path = base_path + scene            # đường dẫn tới scene, vd .../nerf_llff_data/fern

    print(f"[CHECK] Scene path: {scene_path}")
    print(f"[CHECK] Exists: {os.path.exists(scene_path)}")
    if os.path.exists(scene_path):
        print(f"[CHECK] Contents: {os.listdir(scene_path)}")
    sys.stdout.flush()

    os.chdir(scene_path)
    print(f"[STEP 1] Removing old {view_path} folder...")
    sys.stdout.flush()
    os.system('rm -rf ' + view_path)
    os.mkdir(view_path)
    os.chdir(view_path)
    os.makedirs('created', exist_ok=True)
    os.makedirs('triangulated', exist_ok=True)
    os.makedirs('images', exist_ok=True)

    print(f"[STEP 2] Converting COLMAP model to TXT...")
    sys.stdout.flush()
    run_cmd('colmap model_converter --input_path ../sparse/0/ --output_path ../sparse/0/ --output_type TXT')

    print(f"[STEP 3] Reading images.txt...")
    sys.stdout.flush()
    images = {}
    with open('../sparse/0/images.txt', "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                fid.readline().split()
                images[image_name] = elems[1:]

    print(f"[INFO] Total images found: {len(images)}")
    sys.stdout.flush()

    # ── Chọn training views (cùng logic 2 bước với dataset_readers.py) ──
    img_list = sorted(images.keys(), key=lambda x: x)
    # Bước 1: bỏ các ảnh test (idx chia hết 8) → còn train pool
    train_img_list = [c for idx, c in enumerate(img_list) if idx % llffhold != 0]
    if n_views > 0:
        # Bước 2: từ train pool lấy đúng n_views ảnh cách đều nhau
        idx_sub = [round_python3(i) for i in np.linspace(0, len(train_img_list)-1, n_views)]
        train_img_list = [c for idx, c in enumerate(train_img_list) if idx in idx_sub]

    print(f"[INFO] Selected {n_views} training views: {train_img_list}")
    sys.stdout.flush()

    print(f"[STEP 4] Copying selected images...")
    sys.stdout.flush()
    for img_name in train_img_list:
        run_cmd(f'cp ../images/{img_name} images/{img_name}')

    os.system('cp ../sparse/0/cameras.txt created/.')
    with open('created/points3D.txt', "w") as fid:
        pass
        
    # Đoạn này là để tạo sparse point cloud đáng tin cậy
    # STEP 5: Trích đặc trưng SIFT trên mỗi ảnh (điểm keypoint + mô tả). Đây là bước
    # tìm "điểm dễ nhận diện" để sau này khớp giữa các ảnh.
    print(f"[STEP 5] Extracting SIFT features...")
    sys.stdout.flush()
    res = os.popen('colmap feature_extractor --database_path database.db --image_path images --SiftExtraction.max_image_size 4032 --SiftExtraction.max_num_features 32768 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1').read()
    print(f"[INFO] Feature extractor output: {res[:200] if res else 'empty'}")
    sys.stdout.flush()

    # STEP 6: Khớp đặc trưng giữa mọi cặp ảnh (exhaustive = thử tất cả cặp). Tìm
    # keypoint nào ở ảnh này ứng với keypoint nào ở ảnh kia.
    print(f"[STEP 6] Matching features...")
    sys.stdout.flush()
    run_cmd('colmap exhaustive_matcher --database_path database.db --FeatureMatching.use_gpu 0 --FeatureMatching.max_num_matches 4096')

    print(f"[STEP 7] Reading database...")
    sys.stdout.flush()
    db = COLMAPDatabase.connect('database.db')
    db_images = db.execute("SELECT * FROM images")
    img_rank = [db_image[1] for db_image in db_images]
    print(f"[INFO] DB images: {img_rank}")
    sys.stdout.flush()

    with open('created/images.txt', "w") as fid:
        for idx, img_name in enumerate(img_rank):
            print(f"  Writing: {img_name}")
            data = [str(1 + idx)] + [' ' + item for item in images[os.path.basename(img_name)]] + ['\n\n']
            fid.writelines(data)

    # STEP 8: Tam giác hóa (triangulation) — từ các match 2D + camera poses (đã biết),
    # tính ra vị trí 3D của mỗi điểm. Đây là SPARSE point cloud (thưa).
    print(f"[STEP 8] Triangulating points...")
    sys.stdout.flush()
    run_cmd('colmap point_triangulator --database_path database.db --image_path images --input_path created --output_path triangulated --Mapper.ba_local_max_num_iterations 40 --Mapper.ba_local_max_refinements 3 --Mapper.ba_global_max_num_iterations 100')

    # STEP 9: Xuất model triangulated ra dạng TXT (để đọc bằng Python sau này).
    print(f"[STEP 9] Converting triangulated model...")
    sys.stdout.flush()
    run_cmd('colmap model_converter --input_path triangulated --output_path triangulated --output_type TXT')

    
    # Đoạn này là để tạo MVS [NOT USE]
    # STEP 10: Khử méo ống kính (undistort) — chuẩn hóa ảnh về mô hình pinhole lý tưởng,
    # chuẩn bị cho stereo dense.
    print(f"[STEP 10] Undistorting images...")
    sys.stdout.flush()
    run_cmd('colmap image_undistorter --image_path images --input_path triangulated --output_path dense')

    # STEP 11: Patch-match stereo — ước lượng depth dày đặc cho từng pixel bằng cách so
    # khớp patch giữa các ảnh. Đây là bước MVS (Multi-View Stereo), tạo point cloud DÀY.
    print(f"[STEP 11] Patch match stereo...")
    sys.stdout.flush()
    run_cmd('colmap patch_match_stereo --workspace_path dense')

    # STEP 12: Hợp nhất (fusion) các depth map từ nhiều view thành 1 point cloud dense
    # → fused.ply. ĐÂY là file init Gaussian (MVS baseline) mà training đọc.
    print(f"[STEP 12] Stereo fusion...")
    sys.stdout.flush()
    run_cmd('colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply')

    print(f"[DONE] Scene {scene} completed!\n")
    sys.stdout.flush()


if __name__ == '__main__':
    # [CRSGaussian] Guard để `from colmap_llff import pipeline` không trigger
    # loop này — tránh race condition + xóa 3_views có sẵn.
    #
    # Cấu hình qua biến môi trường, không cần sửa file:
    #   DATA_ROOT  thư mục chứa các scene   (mặc định data/nerf_llff_data)
    #   N_VIEWS    số view train            (3 | 6 | 9)
    #   SCENES     danh sách scene, cách nhau bằng khoảng trắng
    base_path = os.environ.get('DATA_ROOT', 'data/nerf_llff_data')
    n_views = int(os.environ.get('N_VIEWS', '3'))
    scenes = os.environ.get(
        'SCENES', 'fern flower fortress horns leaves orchids room trex').split()
    print(f"[colmap_llff] DATA_ROOT={base_path}  N_VIEWS={n_views}  SCENES={scenes}")
    for scene in scenes:
        pipeline(scene, base_path=base_path, n_views=n_views)
