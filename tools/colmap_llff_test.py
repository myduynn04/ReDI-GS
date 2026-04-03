import os
import numpy as np
import sys
import sqlite3

IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

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

def round_python3(number):
    rounded = round(number)
    if abs(number - rounded) == 0.5:
        return 2.0 * round(number / 2.0)
    return rounded

def run_cmd(cmd):
    print(f"  [CMD] {cmd}")
    sys.stdout.flush()
    ret = os.system(cmd)
    print(f"  [RET] exit code = {ret}")
    sys.stdout.flush()
    return ret

def pipeline(scene, base_path, n_views):
    print(f"\n{'='*60}")
    print(f"[SCENE] Processing: {scene}")
    print(f"{'='*60}")
    sys.stdout.flush()

    llffhold = 8
    view_path = str(n_views) + '_views'
    scene_path = base_path + scene

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

    img_list = sorted(images.keys(), key=lambda x: x)
    train_img_list = [c for idx, c in enumerate(img_list) if idx % llffhold != 0]
    if n_views > 0:
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

    print(f"[STEP 5] Extracting SIFT features...")
    sys.stdout.flush()
    res = os.popen('colmap feature_extractor --database_path database.db --image_path images --SiftExtraction.max_image_size 4032 --SiftExtraction.max_num_features 32768 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1').read()
    print(f"[INFO] Feature extractor output: {res[:200] if res else 'empty'}")
    sys.stdout.flush()

    print(f"[STEP 6] Matching features...")
    sys.stdout.flush()
    run_cmd('colmap exhaustive_matcher --database_path database.db --FeatureMatching.use_gpu 0 --FeatureMatching.max_num_matches 16384')

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

    print(f"[STEP 8] Triangulating points...")
    sys.stdout.flush()
    run_cmd('colmap point_triangulator --database_path database.db --image_path images --input_path created --output_path triangulated --Mapper.ba_local_max_num_iterations 40 --Mapper.ba_local_max_refinements 3 --Mapper.ba_global_max_num_iterations 100')

    print(f"[STEP 9] Converting triangulated model...")
    sys.stdout.flush()
    run_cmd('colmap model_converter --input_path triangulated --output_path triangulated --output_type TXT')

    print(f"[STEP 10] Undistorting images...")
    sys.stdout.flush()
    run_cmd('colmap image_undistorter --image_path images --input_path triangulated --output_path dense')

    print(f"[STEP 11] Patch match stereo...")
    sys.stdout.flush()
    run_cmd('colmap patch_match_stereo --workspace_path dense')

    print(f"[STEP 12] Stereo fusion...")
    sys.stdout.flush()
    run_cmd('colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply')

    print(f"[DONE] Scene {scene} completed!\n")
    sys.stdout.flush()


for scene in ['fern']:
    pipeline(scene, base_path='/home/aidev/workspace/representation-3d/duyen/CoR-GS/data/nerf_llff_data/', n_views=3)
