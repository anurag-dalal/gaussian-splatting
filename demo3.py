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
import warnings, json
import struct, os

def cam_to_json(R, T):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.T
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    return rot, pos
def json_to_cam(rot, pos):
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

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)
def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = []
    #infos = []
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            R = qvec2rotmat(qvec)
            #infos.append(R.ravel().tolist() + tvec.tolist())
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images.append({'position':tvec, 'rotation':R})

    return images

def plot_3d_cameras(camera_data, ax, quiv=True):
    maxdist = max([np.linalg.norm(camera['position']) for camera in camera_data])*0.1
    x,y,z = zip(*[camera['position'] for camera in camera_data])

    if quiv:
        rx,ry,rz = zip(*[camera['rotation'][:, 0] for camera in camera_data])
        ax.quiver(x, y, z, rx, ry, rz, length=maxdist, color='r', label='x')
        rx,ry,rz = zip(*[camera['rotation'][:, 1] for camera in camera_data])
        ax.quiver(x, y, z, rx, ry, rz, length=maxdist, color='g', label='y')
        rx,ry,rz = zip(*[camera['rotation'][:, 2] for camera in camera_data])
        ax.quiver(x, y, z, rx, ry, rz, length=maxdist, color='b', label='z')
        ax.legend()
    ax.scatter(x,y,z, color='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('equal')
    


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

# =========================================================================================================
def compute_rotation_matrix_colmap(cam_centre, principal_point):
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



from scipy.spatial.transform import Rotation as R
param_file_path = '/mnt/c/MyFiles/Datasets/unreal_data_colmap/data/train_uniform'
param_files = [os.path.join(param_file_path, f) for f in os.listdir(param_file_path) if f.endswith('.json')]
camera_data = []
for param_file in param_files:
    x,y,z=get_xyz_from_json('/mnt/c/MyFiles/Datasets/unreal_data_colmap/data/metadata.json')
    principal_point = np.array(unreal_to_colmap_me(x,y,z))


    with open(param_file) as json_file:
        cam = json.load(json_file)
        cam_centre = np.array(unreal_to_colmap_me(cam["Transform"]["x"], cam["Transform"]["y"], cam["Transform"]["z"]))

    rotation_matrix, _ = compute_rotation_matrix_colmap(cam_centre, principal_point)
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()
    colmap_quaternion = np.roll(quaternion, shift=1)
    colmap_rotation = qvec2rotmat(colmap_quaternion)
    colmap_translation = -np.dot(rotation_matrix, cam_centre)
    #rotation_matrix, tvec = json_to_cam(rotation_matrix, cam_centre)
    #camera_data.append({'position':colmap_translation, 'rotation':colmap_rotation})
    camera_data.append({'position':cam_centre, 'rotation':rotation_matrix})

fig = plt.figure(figsize=(10, 8))

# --------------- UNREAL CAMERA DATA
ax1 = fig.add_subplot(221, projection='3d')
plot_3d_cameras(camera_data, ax1)
ax1.set_title('UNREAL PROCESSED')

# --------------- UNREAL AFTER GS JSON
with open('/home/anurag/codes/gaussian_splatting_kristian_fork/gaussian-splatting/output/5185301f-7/cameras.json', 'r') as f:
    camera_data = json.load(f)
# Convert 'position' and 'rotation' entries to NumPy arrays
for camera in camera_data:
    camera['position'] = np.array(camera['position'])
    camera['rotation'] = np.array(camera['rotation'])
ax2 = fig.add_subplot(222, projection='3d')
plot_3d_cameras(camera_data, ax2)
ax2.set_title('UNREAL AFTER GS JSON')

# --------------- COLMAP bin
camera_data = read_extrinsics_binary('/mnt/c/MyFiles/Datasets/unreal_data_colmap/colmap_v1_radial/sparse/0/images.bin')
ax3 = fig.add_subplot(223, projection='3d')
plot_3d_cameras(camera_data, ax3)
ax3.set_title('COLMAP BIN')

# --------------- COLMAP AFTER GS JSON
with open('/home/anurag/codes/gaussian_splatting_kristian_fork/gaussian-splatting/output/4f93b624-5_COLMAP/cameras.json', 'r') as f:
    camera_data = json.load(f)
# Convert 'position' and 'rotation' entries to NumPy arrays
for camera in camera_data:
    camera['position'] = np.array(camera['position'])
    camera['rotation'] = np.array(camera['rotation'])
ax4 = fig.add_subplot(224, projection='3d')
plot_3d_cameras(camera_data, ax4)
ax4.set_title('COLMAP AFTER GS JSON')


plt.show()



# R = camera_data[0]['rotation']
# T = camera_data[0]['position']

# rot, pos = cam_to_json(R, T)
# R_new, T_new = json_to_cam(rot, pos)

# print(R, R_new)
# print(T, T_new)