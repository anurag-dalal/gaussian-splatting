# import camorph.camorph as camorph
 
 
# cams = camorph.read_cameras('COLMAP',r'/mnt/c/MyFiles/Datasets/unreal_data_colmap/colmap_v1_radial/sparse/0')
# # camorph.visualize(cams)
# print(dir(cams[0]))
# print(cams[0].t)
# camorph.visualize(cams)
# # # camorph.write_cameras('fbx', r'\\path\to\file.fbx', cams)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pyquaternion import Quaternion
from Camera import Camera
def visualize(cams, show=True):
    """
    This function takes a list of camorph cameras and visualizes them with matplotlib

    :param cams: A list of camorph Cameras
    :type cams: list[Camera]
    :return: None
    """
    x,y,z = zip(*[x.t for x in cams])
    dir = np.array([0,0,1])
    up = np.array([0,-1,0])
    maxdist = max([np.linalg.norm(x.t) for x in cams])
    dir = dir * maxdist * 0.2
    up = up * maxdist * 0.2
    #dir = np.array([1, 0, 0])
    #up = np.array([0, 0, 1])
    dirx,diry,dirz = zip(*[x.r.rotate(dir) for x in cams])
    upx, upy, upz = zip(*[x.r.rotate(up) for x in cams])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(x, y, z, dirx, diry, dirz, color='r', label='Camera Front')
    ax.quiver(x, y, z, upx, upy, upz, color='g', label='Camera Up')

    ax.set_xlim([-maxdist, maxdist])
    ax.set_ylim([-maxdist, maxdist])
    ax.set_zlim([-maxdist, maxdist])

    ax.scatter(x,y,z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend()
    ax.axis('equal')
    if show:
        plt.show()
    else:
        return fig
    
from typing import Union
import numpy as np
import math
from pyquaternion import Quaternion
import warnings, json


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

def compute_rotation_matrix(cam_centre, principal_point):
    # Compute the camera's forward direction (vector pointing from cam_centre to principal_point)
    forward = principal_point - cam_centre
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
def unreal_to_colmap_me(x,y,z):
    return [y, -z, x]

import os
param_file_path = '/mnt/c/MyFiles/Datasets/unreal_data_colmap/data/train_uniform'
param_files = [os.path.join(param_file_path, f) for f in os.listdir(param_file_path) if f.endswith('.json')]
cams = []
for param_file in param_files:
    x,y,z=get_xyz_from_json('/mnt/c/MyFiles/Datasets/unreal_data_colmap/data/metadata.json')
    principal_point = np.array(unreal_to_colmap_me(x,y,z))


    with open(param_file) as json_file:
        cam = json.load(json_file)
        cam_centre = np.array(unreal_to_colmap_me(cam["Transform"]["x"], cam["Transform"]["y"], cam["Transform"]["z"]))

    rotation_matrix, tvec = compute_rotation_matrix(cam_centre, principal_point)
    cam = Camera()
    cam.r = Quaternion(matrix=rotation_matrix)
    cam.t = cam_centre
    cams.append(cam)
visualize(cams)

# # Calculate inverse of the rotation matrix
# inv_rotation_matrix = np.linalg.inv(rotation_matrix)

# # Calculate the transpose of the rotation matrix
# transpose_rotation_matrix = np.transpose(rotation_matrix)

# # Calculate the difference between inverse and transpose
# difference = inv_rotation_matrix - transpose_rotation_matrix

# # Calculate the norm of the difference
# norm_difference = np.linalg.norm(difference)

# print("Norm of (inv(R) - R^T):", norm_difference)
# print(cam_centre)
# print(Quaternion(matrix=rotation_matrix))