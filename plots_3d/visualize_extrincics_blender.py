#
# First parts are based on:
# https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/scene/colmap_loader.py
# (see copyright notice in that file).
#
# read_intrinsics_json, read_extrinsics_json, and tests: written by K. M. Knausgård 2024-01 (License: MIT)
#

import numpy as np
import collections
import json
import os
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2, imutils

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids", "R", "path"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)



def sorted_image_files(directory):
    # Regex pattern to match image files and extract camera numbers
    pattern = re.compile(r'.*img_(\d+).*\.jpeg$')

    # List all files in the directory
    files = os.listdir(directory)

    # Filter and extract camera numbers
    images = []
    for file in files:
        match = pattern.match(file)
        if match:
            camera_number = int(match.group(1))
            images.append((camera_number, file))

    # Sort by camera number
    images.sort()

    # Extract sorted file names
    sorted_files = [file for _, file in images]

    return sorted_files


def read_intrinsics_json(path, camera_model_type="OPENCV"):
    """
    Written by K. M. Knausgård 2024-01
    Modified to handle both PINHOLE and OPENCV camera models.
    """
    cameras = {}
    with open(path, "r") as file:
        data = json.load(file)
        if camera_model_type not in CAMERA_MODEL_NAMES:
            raise ValueError(f"Unsupported camera model type: {camera_model_type}")

        model_id = CAMERA_MODEL_NAMES[camera_model_type].model_id

        for idx, cam in enumerate(data["Cameras"], start=1):
            width = cam["FovVideo"]["Right"]
            height = cam["FovVideo"]["Bottom"]

            if camera_model_type == "OPENCV":
                # Extract OPENCV model parameters
                params = np.array([
                    cam["Intrinsic"]["FocalLengthU"],
                    cam["Intrinsic"]["FocalLengthV"],
                    cam["Intrinsic"]["CenterPointU"],
                    cam["Intrinsic"]["CenterPointV"],
                    cam["Intrinsic"]["RadialDistortion1"],
                    cam["Intrinsic"]["RadialDistortion2"],
                    cam["Intrinsic"]["TangentalDistortion1"],
                    cam["Intrinsic"]["TangentalDistortion2"],
                    # Add more parameters if available and necessary
                ])
            elif camera_model_type == "PINHOLE":
                # Extract PINHOLE model parameters
                params = np.array([
                    cam["Intrinsic"]["FocalLengthU"],
                    cam["Intrinsic"]["FocalLengthV"],
                    cam["Intrinsic"]["CenterPointU"],
                    cam["Intrinsic"]["CenterPointV"],
                    # PINHOLE model usually has only these four parameters
                ])
            else:
                # Handle other camera models if necessary
                params = np.array([])  # Placeholder for other models

            cameras[idx] = Camera(id=idx, model=model_id, width=width, height=height, params=params)
    return cameras

def calculate_transformation_matrix(x1, y1, z1, x2, y2, z2):
    # Translation Component
    translation_matrix = np.array([[1, 0, 0, x2],
                                   [0, 1, 0, y2],
                                   [0, 0, 1, z2],
                                   [0, 0, 0, 1]])

    # Rotation Component
    u = np.array([x1 - x2, y1 - y2, z1 - z2])
    norm_u = np.linalg.norm(u)
    theta = np.arccos(np.dot(u, [0, 0, 1]) / norm_u)  # angle between u and z-axis

    if norm_u != 0:
        k = u / norm_u
        k_cross = np.array([[0, -k[2], k[1]],
                            [k[2], 0, -k[0]],
                            [-k[1], k[0], 0]])
        k_cross_square = np.dot(k_cross, k_cross)
        rotation_matrix = np.eye(3) + k_cross + (k_cross_square * (1 - np.cos(theta))) / np.dot(theta, theta)
    else:
        rotation_matrix = np.eye(3)

    # Combining Translation and Rotation Components
    # transformation_matrix = np.dot(translation_matrix, np.concatenate((rotation_matrix, np.zeros((3, 1))), axis=1))
    # transformation_matrix[3, 3] = 1

    return rotation_matrix

def get_rotation_matrix(x1, y1, z1, x2, y2, z2):
    P = np.array([x2, y2, z2])
    C = np.array([x1, y1, z1])
    # Ensure the vectors are normalized
    P = P / np.linalg.norm(P)
    C = C / np.linalg.norm(C)

    # Get the angle between the vectors
    dot_product = np.dot(P, C)
    angle = np.arccos(dot_product)

    # Get the axis of rotation using the cross product
    axis = np.cross(P, C)

    # Construct the rotation matrix using the axis and angle
    sin_theta = np.sin(angle)
    cos_theta = np.cos(angle)
    one_minus_cos_theta = 1 - cos_theta
    axis_x = axis[0]
    axis_y = axis[1]
    axis_z = axis[2]

    rotation_matrix = np.array([
        [cos_theta + axis_x ** 2 * one_minus_cos_theta, axis_x * axis_y * one_minus_cos_theta - axis_z * sin_theta, axis_x * axis_z * one_minus_cos_theta + axis_y * sin_theta],
        [axis_x * axis_y * one_minus_cos_theta + axis_z * sin_theta, cos_theta + axis_y ** 2 * one_minus_cos_theta, axis_y * axis_z * one_minus_cos_theta - axis_x * sin_theta],
        [axis_x * axis_z * one_minus_cos_theta - axis_y * sin_theta, axis_y * axis_z * one_minus_cos_theta + axis_x * sin_theta, cos_theta + axis_z ** 2 * one_minus_cos_theta]
    ])

    return rotation_matrix
def get_rotation_matrix2(x1, y1, z1, x2, y2, z2):
    L = np.array([x2, y2, z2])
    V = np.array([x1, y1, z1])
    V = L-V

    # Calculate the forward and up vectors for the object
    forward = V - L
    up = np.array([0, 0, 1]) # Assuming initial up vector is positive z-axis

    # Ensure forward and up vectors are orthogonal
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    up = np.cross(right, forward)

    # Construct the rotation matrix using the forward, up, and right vectors
    rotation_matrix = np.array([
        [right[0], up[0], forward[0]],
        [right[1], up[1], forward[1]],
        [right[2], up[2], forward[2]]
    ])

    return rotation_matrix


def get_camera_extrinsic(x1, y1, z1, x2, y2, z2):
    P = np.array([x2, y2, z2])
    C = np.array([x1, y1, z1])
    # Calculate the forward and up vectors for the camera
    forward = C - P
    up = np.array([0, 1, 0]) # Assuming initial up vector is positive y-axis

    # Ensure forward and up vectors are orthogonal
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    up = np.cross(right, forward)

    # Construct the rotation matrix using the forward, up, and right vectors
    rotation_matrix = np.array([right,up,-forward])
    return rotation_matrix

def read_extrinsics_json(paths):
    """
    Written by K. M. Knausgård 2024-01
    """

    images = {}
    for idx, path in enumerate(paths, start=1):
        with open(path, 'r') as file:
            cam = json.load(file)
            # for idx, cam in enumerate(data["Cameras"], start=1):
            # Extract the translation vector
            tvec = np.array(cam["global_location"])

            # Extract and convert the rotation matrix to a quaternion (why not use the DCM directly?)
            R = np.array(cam["rotation_matrix"])
            qvec = rotmat2qvec(R)

            # Since the JSON does not contain direct image references, we use placeholders
            image_id = idx  # Placeholder image ID, TODO(KMK): Use the actual image ID
            camera_id = idx  # Assuming each camera corresponds to one 'image' for now TODO(KMK).
            image_name = "Camera_{}".format(idx)  # Placeholder image name
            xys = np.empty((0, 2))  # Placeholder for 2D points as we don't have this data
            point3D_ids = np.empty(0)  # Placeholder for 3D point IDs

            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids,
                R=R, path=path[:-5-9]+'.png'
            )
    return images


def test_qualisys_json_loader():
    # Define the directory and file name
    directory_name = '/home/anurag/codes/gaussian_splatting_kristian_fork/gaussian-splatting/scene/qualisys_json_loader_test_data'
    file_name = 'Test_Recording_Qualisys_MoCap_Miqus_Hybrid_Motionlab_UniversityOfAgder_Norway_0001.json'

    # Join the directory and file name to create a path
    test_data_file_path = os.path.join(directory_name, file_name)

    # Test the read_intrinsics_json function
    intrinsics = read_intrinsics_json(test_data_file_path)
    print("Intrinsics Data:")
    for camera_id, camera in intrinsics.items():
        print(
            f"Camera ID: {camera_id}, Model: {camera.model}, Width: {camera.width}, Height: {camera.height}, Params: {camera.params}")

    # Test the read_extrinsics_json function
    extrinsics = read_extrinsics_json(test_data_file_path)
    print("\nExtrinsics Data:")
    for image_id, image in extrinsics.items():
        print(f"Image ID: {image_id}, QVec: {image.qvec}, TVec: {image.tvec}")


def visualize_cameras(images, arrow_length_mm=1000):
    plt.figure()

    # Define a unit vector in the direction the camera is facing (along negative Z-axis)
    unit_vector = np.array([0, 0, -1])

    # Draw each camera
    for image_id, image in images.items():
        # Camera position (negative because extrinsics are camera-to-world)
        camera_pos = image.tvec

        # Draw camera position
        plt.scatter(camera_pos[0], camera_pos[1], c='r', marker='o')

        # Calculate the camera orientation using DCM
        camera_orientation = image.qvec2rotmat().T @ unit_vector

        # Calculate the end point of the orientation line (500 mm along the orientation vector)
        line_end = camera_pos[:2] + (arrow_length_mm * camera_orientation[:2])

        # Draw camera orientation line
        plt.plot([camera_pos[0], line_end[0]], [camera_pos[1], line_end[1]], 'b-')

    # Draw reference frame (coordinate axes)
    axis_length = 2 * arrow_length_mm  # Length of the reference frame axes
    plt.quiver(0, 0, axis_length, 0, color='red', angles='xy', scale_units='xy', scale=1)
    plt.quiver(0, 0, 0, axis_length, color='green', angles='xy', scale_units='xy', scale=1)

    # Set labels and show plot
    plt.xlabel('X axis (mm)')
    plt.ylabel('Y axis (mm)')
    plt.title('Camera Extrinsics Visualization (2D)')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    # Extract the current limits and ranges
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    # Find the maximum range and calculate the extra 10% for padding
    max_range = max(x_range, y_range, z_range)
    padding = max_range * 0.1

    # Set the axis limits with the same scale and padding
    ax.set_xlim([x_limits[0] - padding, x_limits[1] + padding])
    ax.set_ylim([y_limits[0] - padding, y_limits[1] + padding])
    ax.set_zlim([0, z_range + padding])

from matplotlib import cm
def visualize_cameras_3d(images, arrow_length_mm=1000, v=0):
    fig = plt.figure()
    fig.set_size_inches(12,12)
    ax = fig.add_subplot(111, projection='3d')
    if v==0:
        ax.view_init(elev=90, azim=-90, roll=0)
    elif v==1:
        ax.view_init(elev=30, azim=-68, roll=0)
    # Create a 'floor' grid at z=0
    #create_floor(ax, images, grid_size=arrow_length_mm)

    # Define axis lengths
    axis_length = arrow_length_mm

    # Draw reference frame axes
    # X-axis in red, Y-axis in green, Z-axis in blue
    ax.quiver(0, -0, 0, axis_length, 0, 0, color='red')
    ax.quiver(0, -0, 0, 0, axis_length, 0, color='green')
    ax.quiver(0, -0, 0, 0, 0, axis_length, color='blue')
    ax.scatter(0, -0, 0, c='g', marker='o')
    center = [0, 0, 0]
    r = 500

    # generate the sphere mesh
    u, v = np.mgrid[0:2*np.pi:200j, 0:np.pi/2:100j]
    x = center[0] + r*np.cos(u)*np.sin(v)
    y = center[1] + r*np.sin(u)*np.sin(v)
    z = center[2] + r*np.cos(v)
    # draw the opaque sphere
    # ax.plot_surface(x, y, z, alpha=0.5)

    # Draw each camera
    img_small_arr = np.zeros((30,100,3), dtype=np.uint8)
    
    for image_id, image in images.items():
        # Camera position
        camera_pos = image.tvec
        img_small_arr = cv2.imread(image.path)
        img_small_arr = imutils.resize(img_small_arr, width=128)
        h, w, _ = img_small_arr.shape

        # Draw camera position
        ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], c='r', marker='o')

        # Calculate the camera orientation using DCM
        # camera_orientation = image.R.T @ np.array([0, 0, -1])
        camera_orientation = image.qvec2rotmat().T @ np.array([0, 0, -1])
        normal_vector = camera_orientation / np.linalg.norm(camera_orientation)
        # Calculate the end point of the orientation line (500 mm along the orientation vector)
        line_end = camera_pos + (arrow_length_mm * camera_orientation)

        # Draw camera orientation line
        ax.plot([camera_pos[0], line_end[0]],
                [camera_pos[1], line_end[1]],
                [camera_pos[2], line_end[2]], 'b-')    
    

        # Define dimensions of the small image
        img_height, img_width = img_small_arr.shape[:2]
        w, h = img_width, img_height

        # Calculate corner points of the surface
        x_min, x_max = -img_width / 2, img_width / 2
        y_min, y_max = -img_height / 2, img_height / 2

        v1 = np.cross(normal_vector, np.array([1, 0, 0]))
        v2 = np.cross(normal_vector, v1)

        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)

        half_width = w / 2
        half_height = h / 2

        top_left = camera_pos + half_height * v1 - half_width * v2
        top_right = camera_pos + half_height * v1 + half_width * v2
        bottom_left = camera_pos - half_height * v1 - half_width * v2
        bottom_right = camera_pos - half_height * v1 + half_width * v2

        # plot 4 corner points
        # ax.plot(top_left[0], top_left[1], top_left[2], c='g', marker='o')
        # ax.plot(top_right[0], top_right[1], top_right[2], c='g', marker='o')
        # ax.plot(bottom_left[0], bottom_left[1], bottom_left[2], c='g', marker='o')
        # ax.plot(bottom_right[0], bottom_right[1], bottom_right[2], c='g', marker='o')


        x_top_left, y_top_left = top_left[0], top_left[1]
        x_top_right, y_top_right = top_right[0], top_right[1]
        x_bottom_left, y_bottom_left = bottom_left[0], bottom_left[1]
        x_bottom_right, y_bottom_right = bottom_right[0], bottom_right[1]

        x_min = min(x_top_left, x_top_right, x_bottom_left, x_bottom_right)
        x_max = max(x_top_left, x_top_right, x_bottom_left, x_bottom_right)
        y_min = min(y_top_left, y_top_right, y_bottom_left, y_bottom_right)
        y_max = max(y_top_left, y_top_right, y_bottom_left, y_bottom_right)


        # Create a mesh grid for the surface
        x = np.linspace(x_min, x_max, img_width)
        y = np.linspace(y_min, y_max, img_height)
        X, Y = np.meshgrid(x, y)

        # A = top_left[1] * (bottom_right[2] - top_right[2]) - top_right[1] * (bottom_left[2] - top_left[2])
        # B = top_left[0] * (bottom_right[2] - top_right[2]) - top_right[0] * (bottom_left[2] - top_left[2])
        # C = -top_left[0] * top_left[1] * (bottom_right[2] - top_right[2]) + top_right[0] * top_right[1] * (bottom_left[2] - top_left[2]) + bottom_left[0] * bottom_left[1] * (top_right[2] - top_left[2]) - bottom_right[0] * bottom_right[1] * (top_left[2] - top_right[2])
        # D = -A * top_left[0] - B * top_left[1] - C * top_left[2]

        # Calculate surface points by offsetting from camera position along normal vector
        Z = np.zeros_like(X)  # Initialize Z values
        # Offset each point based on its position relative to the center
        for i in range(h):
            for j in range(w):
                # pos = np.array([X[i,j], Y[i,j], camera_pos[2]])
                # Z[i, j] = camera_pos[2] + np.dot(pos, normal_vector)
                Z[i, j] = (-normal_vector[0]*(X[i,j]-camera_pos[0]) - normal_vector[1]*(Y[i,j]-camera_pos[1]))/(normal_vector[2]+0.001) + camera_pos[2]
                Z[i, j] = 0
                # Z[i, j] = (-normal_vector[0]*(x[j]-camera_pos[0]) - normal_vector[1]*(y[i]-camera_pos[1]))/normal_vector[2] + camera_pos[2]

        #Z = (-A * X - B * Y - D) / C
        x = np.linspace(-w / 2, w / 2, img_width)
        y = np.linspace(-h / 2, h / 2, img_height)
        X = np.zeros((h,w))
        Y = np.zeros((h,w))
        Z = np.zeros((h,w))
        for i in range(h):
            for j in range(w):
                P = camera_pos - x[j] * v2 + y[i] * v1
                X[i,j] = P[0]
                Y[i,j] = P[1]
                Z[i,j] = P[2]


        # ax.plot_surface(X, Y, Z, facecolors=img_small_arr / 255, shade=False)
        

        # # Define surface dimensions and coordinates
        # surface_width = 50
        # surface_height = 50
        # surface_center = np.array([camera_pos[0], camera_pos[1], camera_pos[2]])  # Example center, modify as needed
        # u = np.array([0, 0, -1])  # Example up vector, modify as needed

        # # Calculate surface coordinates
        # v = np.cross(camera_orientation, u)
        # u = np.cross(v, camera_orientation)
        # u /= np.linalg.norm(u)
        # v /= np.linalg.norm(v)
        # u *= surface_width / 2
        # v *= surface_height / 2
        # x = np.linspace(surface_center[0] - u[0], surface_center[0] + u[0], w)
        # y = np.linspace(surface_center[1] - v[1], surface_center[1] + v[1], h)
        # print(x)
        # # Create meshgrid
        # xx, yy = np.meshgrid(x, y)
        # zz = surface_center[2] * np.ones_like(xx)

        # ax.plot_surface(xx, yy, zz, facecolors=img_small_arr / 255, shade=False)


    # x = np.arange(-400,400,1)
    # y = np.arange(-400,400,1)
    # xx, yy = np.meshgrid(x, y)
    # zz =  np.ones_like(xx)
    # ax.plot_surface(xx, yy, zz,cmap=cm.coolwarm, linewidth=0, alpha=0.3, antialiased=True)
    
    ax.set_zlim(-500, ax.get_zlim()[1])

    # Set labels
    ax.set_xlabel('X axis (mm)')
    ax.set_ylabel('Y axis (mm)')
    ax.set_zlabel('Z axis (mm)')
    ax.set_aspect('equal')  # https://matplotlib.org/3.6.0/users/prev_whats_new/whats_new_3.6.0.html#equal-aspect-ratio-for-3d-plots
    # ax.set_title('Camera Position and orientation.')
    ax.set_axis_off()
    plt.show()
    return fig


def create_floor(ax, images, grid_size):
    # Find the range for x and y axes from camera positions
    x_values = [image.tvec[0] for image in images.values()]
    y_values = [image.tvec[1] for image in images.values()]

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    # Extend the range by a fraction of the grid size for padding (removed the * 1.1)
    x_min -= grid_size
    x_max += grid_size
    y_min -= grid_size
    y_max += grid_size

    # Create a grid of points
    X, Y = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
    Z = np.zeros_like(X)

    # Plot the transparent plane
    ax.plot_surface(X, Y, Z, color='blue', alpha=0.4)

    # Plot the wireframe grid with less alpha and higher zorder
    ax.plot_wireframe(X, Y, Z, color='gray', alpha=0.5)




if __name__ == "__main__":
    # Run tests
    # test_qualisys_json_loader()


    # Visualize cameras for test case

    # Load extrinsic data
    directory = '/mnt/c/MyFiles/Datasets/blender_test_data'
    extrinsic_files = os.listdir(directory)
    extrinsic_files = [os.path.join(directory, f) for f in extrinsic_files if '.json' in f]
    # print(extrinsic_files)
    # extrinsic_file = os.path.join(directory, 'Test_Recording_Qualisys_MoCap_Miqus_Hybrid_Motionlab_UniversityOfAgder_Norway_0001.json')
    extrinsics = read_extrinsics_json(extrinsic_files)


    # Visualize cameras with sorted extrinsic parameters
    # visualize_cameras(sorted_images)

    # # Enable interactive mode for 3D visualization
    # plt.ion()
    for v in [0]:
        p = visualize_cameras_3d(extrinsics, 0.5, v)
