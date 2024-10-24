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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        # sys.stdout.write('\r')
        # # the exact output you're looking for:
        # sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        # sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)            
        else:

            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        # print(image.size[1],FovY, image.size[0],FovX)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, num_requested=-1):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    if num_requested==-1:
        scene_info = SceneInfo(point_cloud=pcd,
                            train_cameras=train_cam_infos,
                            test_cameras=test_cam_infos,
                            nerf_normalization=nerf_normalization,
                            ply_path=ply_path)
        return scene_info
    else:
        import random
        random.seed(10)
        scene_info = SceneInfo(point_cloud=pcd,
                            train_cameras=random.sample(train_cam_infos, num_requested),
                            test_cameras=train_cam_infos,
                            nerf_normalization=nerf_normalization,
                            ply_path=ply_path)
        return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx
            
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def get_xyz_from_json(filename):
    # Open and load the JSON file
    with open(filename, 'r') as file:
        data = json.load(file)
    
    # Extract the x, y, z coordinates from the Centrepoint
    centrepoint = data.get("Centrepoint", {})
    x = centrepoint.get("x", None)
    y = centrepoint.get("y", None)
    z = centrepoint.get("z", None)
    
    return x, y, z
def JSON_to_camera(rot, pos):
    # Initialize the world-to-camera transformation matrix W2C
    W2C = np.zeros((4, 4))
    W2C[:3, :3] = rot
    W2C[:3, 3] = pos
    W2C[3, 3] = 1.0

    # Compute the camera-to-world transformation matrix Rt by inverting W2C
    Rt = np.linalg.inv(W2C)

    # Extract the rotation matrix R and translation vector T from Rt
    R = Rt[:3, :3].T
    T = Rt[:3, 3]

    return R, T

def unreal_to_colmap_me(x,y,z):
    return [y, -z, x] # FOR COLMAP

def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def calculate_camera_parameters(cam_centre, principal_point):
    # Compute the camera's forward direction (vector pointing from cam_centre to principal_point)
    forward = -(principal_point - cam_centre)
    forward = forward / np.linalg.norm(forward)  # Normalize the forward vector

    buttom = np.array([0, 1, 0])
    # Compute the right direction as the cross product of forward and the up_vector
    right = np.cross(buttom, forward)
    right = right / np.linalg.norm(right)  # Normalize the right vector

    # Compute the new up direction as the cross product of forward and right vectors
    buttom = np.cross(forward, right)

    # Construct the rotation matrix: [right, up, forward]
    rotation_matrix = np.vstack([right, buttom, forward]).T  # Transpose to fit rotation matrix convention
    tvec = -rotation_matrix.T @ cam_centre

    return rotation_matrix, tvec

import re
def readUnrealCamerasFromTransforms(meta_data_file, directory, white_background, extension=".jpeg"):
    cam_infos = []
    x,y,z = get_xyz_from_json(meta_data_file)
    principal_axis_point = np.array(unreal_to_colmap_me(x,y,z))
    extrinsic_files = os.listdir(directory)
    extrinsic_files = [os.path.join(directory, f) for f in extrinsic_files if '.json' in f]
    # infos = []
    pattern = r'img_(\d+)\.json'
    for idx, param_file in enumerate(extrinsic_files, start=0):
        with open(param_file) as json_file:
            json_number = re.search(pattern, param_file).group(1)
            cam_name = directory + '/img_{}{}'.format(json_number, extension)
            image_name = Path(cam_name).stem
            image = Image.open(cam_name)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            cam = json.load(json_file)
            tvec = np.array(unreal_to_colmap_me(cam["Transform"]["x"], cam["Transform"]["y"], cam["Transform"]["z"]))
            R, _ = calculate_camera_parameters(principal_axis_point, tvec)
            R, tvec = JSON_to_camera(R, tvec)

            # tvec = np.array([tvec[0], tvec[1], tvec[2]])
            # infos.append(R.ravel().tolist() + tvec.tolist())
            focal_length_x = cam["Intrinsic"]["fx"]
            focal_length_y = cam["Intrinsic"]["fy"]
            focal_length_x = 672.199236681
            focal_length_y = 672.199236681
            FovY = focal2fov(focal_length_y, image.size[1])
            FovX = focal2fov(focal_length_x, image.size[0])
            # FovY = 1.351322864871
            # FovX = 1.91772702
            # print(image.size[1],FovY, image.size[0],FovX)
            # print(arr.shape)
            cam_infos.append(CameraInfo(uid=int(json_number), R=R, T=tvec, FovY=FovY, FovX=FovX, image=Image.open(cam_name), image_path=cam_name, image_name=image_name, width=image.size[0], height=image.size[1]))
    # infos = np.array(infos)
    # n = np.random.randint(300)
    # np.savetxt("output_unreal_{}.csv".format(n), infos, delimiter=",", fmt='%f')   
    # print('-----------------------------------------------------------------', "output_unreal_{}.csv".format(n), '\n')     
    return cam_infos
def readUnrealCamerasFromTransformsOld(meta_data_file, directory, white_background, extension=".jpeg"):
    cam_infos = []
    x,y,z = get_xyz_from_json(meta_data_file)
    principal_axis_point = unreal_to_colmap_me(x,y,z)
    extrinsic_files = os.listdir(directory)
    extrinsic_files = [os.path.join(directory, f) for f in extrinsic_files if '.json' in f]

    for idx, param_file in enumerate(extrinsic_files, start=0):
        with open(param_file) as json_file:
            cam_name = directory + '/img_{}{}'.format(idx, extension)
            image_name = Path(cam_name).stem
            image = Image.open(cam_name)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            cam = json.load(json_file)
            tvec = np.array([cam["Transform"]["x"], cam["Transform"]["y"], cam["Transform"]["z"]])

            # Extract and convert the rotation matrix to a quaternion (why not use the DCM directly?)
            R = np.array([
                [cam["Transform"]["r11"], cam["Transform"]["r12"], cam["Transform"]["r13"]],
                [cam["Transform"]["r21"], cam["Transform"]["r22"], cam["Transform"]["r23"]],
                [cam["Transform"]["r31"], cam["Transform"]["r32"], cam["Transform"]["r33"]]
            ])

            focal_length_x = cam["Intrinsic"]["fx"]
            focal_length_y = cam["Intrinsic"]["fy"]
            FovY = focal2fov(focal_length_y, image.size[1])
            FovX = focal2fov(focal_length_x, image.size[0])
            # print(image.size[1],FovY, image.size[0],FovX)
            cam_infos.append(CameraInfo(uid=idx, R=R, T=tvec, FovY=FovY, FovX=FovX, image=image,
                            image_path=cam_name, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos
def readUnrealSyntheticInfo(path, white_background, eval, extension=".jpeg"):
    print("Reading Training Transforms")
    train_cam_infos = readUnrealCamerasFromTransforms(os.path.join(path, 'metadata.json'), os.path.join(path, 'train_uniform'), white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readUnrealCamerasFromTransforms(os.path.join(path, 'metadata.json'), os.path.join(path, 'test'), white_background, extension)
    # Parse the JSON data
    print(os.path.join(path, 'metadata.json'))
    unreal_cams = json.load(open(os.path.join(path, 'metadata.json')))

    # Extract radius
    radius = unreal_cams['Radius']

    # Extract centrepoint
    centrepoint = unreal_cams['Centrepoint']
    x = centrepoint['x']
    y = centrepoint['y']
    z = centrepoint['z']
    x,y,z = unreal_to_colmap_me(x,y,z)
    nerf_normalization = getNerfppNorm(train_cam_infos)
    print(nerf_normalization)
    nerf_normalization['radius'] = unreal_cams['Radius'] * 2 * 1.2
    nerf_normalization['translate'] = np.array([-x,-y,-z])
    print(nerf_normalization)
    

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Unreal scenes
        xyz = np.random.random((num_pts, 3)) * (2 * radius) - radius
        xyz[:, 0] += x
        xyz[:, 1] += y
        xyz[:, 2] += z
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Unreal" : readUnrealSyntheticInfo,
}