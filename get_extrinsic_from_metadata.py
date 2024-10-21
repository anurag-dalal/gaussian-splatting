import numpy as np
from scipy.spatial.transform import Rotation as R

def unreal_to_colmap_me(x,y,z):
    return y, -z, x
def unreal_to_colmap_chatgpt(x,y,z):
    return x, -z, y

def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def calculate_camera_parameters(optical_center, principal_axis_point):
    """
    Calculate camera rotation matrix and translation vector in COLMAP format.

    :param optical_center: The optical center of the camera (x, y, z)
    :param principal_axis_point: A point on the principal axis of the camera (a, b, c)
    :return: Rotation matrix and translation vector in COLMAP format
    """
    # Optical center (x, y, z)
    optical_center = np.array(optical_center)

    # Principal axis point (a, b, c)
    principal_axis_point = np.array(principal_axis_point)

    # Calculate the z-axis (view direction) of the camera
    z_cam = normalize(principal_axis_point - optical_center)

    # Assume the world Y-axis points downward (0, -1, 0)
    y_world = np.array([0, -1, 0])

    # Calculate the x-axis (right vector) of the camera
    x_cam = normalize(np.cross(y_world, z_cam))

    # Calculate the y-axis (up vector) of the camera
    y_cam = np.cross(z_cam, x_cam)

    # Camera rotation matrix (R)
    rotation_matrix = np.vstack([x_cam, y_cam, z_cam])

    # Camera translation vector (T)
    translation_vector = -rotation_matrix @ optical_center

    return rotation_matrix, translation_vector
import json
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
def get_optical_centre_from_json(filename):
    # Open and load the JSON file
    with open(filename, 'r') as file:
        data = json.load(file)
    
    # Extract the x, y, z coordinates from the Centrepoint
    centrepoint = data.get("Transform", {})
    x = centrepoint.get("x", None)
    y = centrepoint.get("y", None)
    z = centrepoint.get("z", None)
    return x, y, z
# Example usage
optical_center = get_optical_centre_from_json('/mnt/c/MyFiles/Datasets/unreal_data_colmap/data/train_uniform/img_243.json')  # Optical center (x, y, z)
principal_axis_point = get_xyz_from_json('/mnt/c/MyFiles/Datasets/unreal_data_colmap/data/metadata.json')  # Point on the principal axis (a, b, c)

x,y,z = unreal_to_colmap_me(optical_center[0], optical_center[1], optical_center[2])
optical_center = [x,y,z]
print(optical_center)
x,y,z = unreal_to_colmap_me(principal_axis_point[0], principal_axis_point[1], principal_axis_point[2])
principal_axis_point = [x,y,z]
rotation_matrix, translation_vector = calculate_camera_parameters(optical_center, principal_axis_point)

print("Rotation Matrix (R):")
print(rotation_matrix)
r = R.from_matrix(rotation_matrix)
print("Rotation Quat (R):")
print(r.as_quat())
print("\nTranslation Vector (T):")
print(translation_vector)