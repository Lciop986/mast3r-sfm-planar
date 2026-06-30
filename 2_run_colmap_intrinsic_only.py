from tqdm import tqdm
import shutil
import os
import subprocess
import numpy as np
import cv2
import json
from glob import glob
from utils.database import blob_to_array, COLMAPDatabase
from argparse import ArgumentParser


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()  # not R but R^-1
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def camTodatabase(txtfile, database_path):
    camModelDict = {'SIMPLE_PINHOLE': 0,
                    'PINHOLE': 1,
                    'SIMPLE_RADIAL': 2,
                    'RADIAL': 3,
                    'OPENCV': 4,
                    'FULL_OPENCV': 5,
                    'SIMPLE_RADIAL_FISHEYE': 6,
                    'RADIAL_FISHEYE': 7,
                    'OPENCV_FISHEYE': 8,
                    'FOV': 9,
                    'THIN_PRISM_FISHEYE': 10}

    # Open the database.
    db = COLMAPDatabase.connect(database_path)

    idList = list()
    modelList = list()
    widthList = list()
    heightList = list()
    paramsList = list()
    # Update real cameras from .txt
    with open(txtfile, "r") as cam:
        lines = cam.readlines()
        for i in range(0, len(lines), 1):
            if lines[i][0] != '#':
                strLists = lines[i].split()
                cameraId = int(strLists[0])
                cameraModel = camModelDict[strLists[1]]  # SelectCameraModel
                width = int(strLists[2])
                height = int(strLists[3])
                paramstr = np.array(strLists[4:12])
                params = paramstr.astype(np.float64)
                idList.append(cameraId)
                modelList.append(cameraModel)
                widthList.append(width)
                heightList.append(height)
                paramsList.append(params)
                camera_id = db.update_camera(cameraModel, width, height, params, cameraId)

    # Commit the data to the file.
    db.commit()
    # Read and check cameras.
    rows = db.execute("SELECT * FROM cameras")
    for i in range(0, len(idList), 1):
        camera_id, model, width, height, params, prior = next(rows)
        params = blob_to_array(params, np.float64)
        assert camera_id == idList[i]
        assert model == modelList[i] and width == widthList[i] and height == heightList[i]
        assert np.allclose(params, paramsList[i])

    # Close database.db.
    db.close()


###############################################################################
parser = ArgumentParser()
# parser.add_argument("--cases", nargs="+", type=int, default=[])
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--image_folder", type=str, default="images_undistorted")
parser.add_argument("--camera_file", type=str, default=None)
parser.add_argument("--image_scale_factor", type=int, default=4)
parser.add_argument("--image_sample_factor", type=int, default=1)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--dense_recon", action="store_true", default=False)
args = parser.parse_args()


scene_name = os.path.basename(args.data_path)
gpu_index = args.gpu

if args.camera_file is None:
    args.camera_file = os.path.join(args.data_path, "camera_calib.txt")

with open(args.camera_file, "r") as f:
    camera = json.load(f)

image_w = camera["width"]
image_h = camera["height"]
focal_x = camera["focal_x"]
focal_y = camera["focal_y"]
center_x = camera["center_x"]
center_y = camera["center_y"]
camera_model = "PINHOLE"

print(f"processing {args.data_path} ...")
scene_path = args.data_path
image_path = os.path.join(scene_path, args.image_folder)

sparse_path = os.path.join(scene_path, "sparse")
model_path = os.path.join(scene_path, "model")
dense_path = os.path.join(scene_path, "dense")
shutil.rmtree(model_path, ignore_errors=True)
shutil.rmtree(sparse_path, ignore_errors=True)
shutil.rmtree(dense_path, ignore_errors=True)
os.makedirs(model_path, exist_ok=True)
os.makedirs(sparse_path, exist_ok=True)
os.makedirs(dense_path, exist_ok=True)

# shutil.copytree(os.path.join(sparse_path, "0"), sparse_path, dirs_exist_ok=True)

images_list = sorted(os.listdir(image_path))

if args.image_scale_factor > 1 :
    new_image_folder = args.image_folder + f"_{args.image_scale_factor}"
    new_image_path = os.path.join(scene_path, new_image_folder)
    image_w = int(image_w / args.image_scale_factor)
    image_h = int(image_h / args.image_scale_factor)
    focal_x = focal_x / args.image_scale_factor
    focal_y = focal_y / args.image_scale_factor
    center_x = center_x / args.image_scale_factor
    center_y = center_y / args.image_scale_factor
    if not os.path.exists(new_image_path):
        os.makedirs(new_image_path, exist_ok=True)
        print(f"Down scale images ... factor={args.image_scale_factor}")
        for idx, image_name in enumerate(tqdm(images_list)):
            image_file = os.path.join(image_path, image_name)
            if os.path.isdir(image_file):
                sub_image_dir = image_name
                sub_images_list = glob(os.path.join(image_file, "*.JPG"))
                for sub_idx, image_file in enumerate(tqdm(sub_images_list)):
                    if sub_idx % args.image_sample_factor != 0:
                        continue
                    image = cv2.imread(image_file)
                    image = cv2.resize(image, (image_w ,image_h), interpolation=cv2.INTER_AREA)
                    image_name = os.path.basename(image_file)
                    os.makedirs(os.path.join(new_image_path, sub_image_dir), exist_ok=True)
                    new_image_file = os.path.join(new_image_path, sub_image_dir, image_name)
                    cv2.imwrite(new_image_file, image)
            else:
                if idx % args.image_sample_factor != 0:
                    continue
                image = cv2.imread(image_file)
                image = cv2.resize(image, (image_w, image_h), interpolation=cv2.INTER_AREA) ## INTER_AREA 能缓和摩尔纹的
                new_image_file = os.path.join(new_image_path, image_name)
                cv2.imwrite(new_image_file, image)

    image_path = new_image_path
    print("new image path: ", image_path)

cameras = [[1, camera_model, image_w, image_h, focal_x, focal_y, center_x, center_y]]
## fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
# cameras = [[1, camera_model, image_w, image_h, focal_x, focal_y, center_x, center_y, k1, k2, p1, p2, k3, 0, 0, 0]]
camera_file = os.path.join(model_path, "cameras.txt")
with open(camera_file, "w") as f:
    for cam in cameras:
        line = " ".join([str(elem) for elem in cam])
        f.write(line + "\n")

images_file = os.path.join(model_path, "images.txt")
with open(images_file, "w") as f:
    f.write("")

point3d_file = os.path.join(model_path, "points3D.txt")
with open(point3d_file, "w") as f:
    f.write("")

database_file = os.path.join(model_path, "database.db")

logfile_name = os.path.join(scene_path, "colmap_output.txt")
logfile = open(logfile_name, "w")

feature_extractor_args = [
    "colmap", "feature_extractor",
    "--database_path", database_file,
    "--image_path", image_path,
    "--ImageReader.camera_model", camera_model,
    "--ImageReader.single_camera", "1",
    # "--ImageReader.mask_path", mask_path
    "--SiftExtraction.gpu_index", gpu_index
]
feature_output = subprocess.check_output(feature_extractor_args, universal_newlines=True, shell=False)
logfile.write(feature_output)
# print(feature_output)
print("Features extracted")

### update camera intrinsics in db ###
camTodatabase(txtfile=camera_file, database_path=database_file)

exhaustive_matcher_args = [
    'colmap', "exhaustive_matcher", #exhaustive_matcher, sequential_matcher
    '--database_path', database_file,
    "--SiftMatching.gpu_index", gpu_index
]
match_output = subprocess.check_output(exhaustive_matcher_args, universal_newlines=True, shell=False)
logfile.write(match_output)
# print(match_output)
print("feature matched")

mapper_args = [
    "colmap", "mapper",
    "--database_path", database_file,
    "--image_path", image_path,
    "--output_path", sparse_path
]
mapper_output = subprocess.check_output(mapper_args, universal_newlines=True, shell=False)
logfile.write(mapper_output)
# print(mapper_output)
print("mapper finished")
shutil.copytree(os.path.join(sparse_path, "0"), sparse_path, dirs_exist_ok=True)

if args.dense_recon == True:
    image_undistorter_args = [
        "colmap", "image_undistorter",
        "--image_path", image_path,
        "--input_path", sparse_path,
        "--output_path", dense_path,
    ]
    image_undistorter_output = subprocess.check_output(image_undistorter_args, universal_newlines=True, shell=False)
    logfile.write(image_undistorter_output)
    # print(image_undistorter_output)
    print("image undistorter finished")

    patch_match_stereo_args = [
        "colmap", "patch_match_stereo",
        "--workspace_path", dense_path,
        "--PatchMatchStereo.gpu_index", gpu_index
    ]
    patch_match_stereo_output = subprocess.check_output(patch_match_stereo_args, universal_newlines=True, shell=False)
    logfile.write(patch_match_stereo_output)
    # print(patch_match_stereo_output)
    print("patch match stereo finished")

    stereo_fusion_args = [
        "colmap", "stereo_fusion",
        "--workspace_path", dense_path,
        "--output_path", os.path.join(dense_path, "fused.ply")
    ]
    stereo_fusion_output = subprocess.check_output(stereo_fusion_args, universal_newlines=True, shell=False)
    logfile.write(stereo_fusion_output)
    # print(stereo_fusion_output)
    print("stereo fusion finished")

    poisson_mesher_args = [
        "colmap", "poisson_mesher",
        "--input_path", os.path.join(dense_path, "fused.ply"),
        "--output_path", os.path.join(dense_path, "meshed-poisson.ply")
    ]
    poisson_mesher_output = subprocess.check_output(poisson_mesher_args, universal_newlines=True, shell=False)
    logfile.write(poisson_mesher_output)
    print("poisson mesher finished")
    logfile.close()

    shutil.rmtree(os.path.join(dense_path, "stereo"))