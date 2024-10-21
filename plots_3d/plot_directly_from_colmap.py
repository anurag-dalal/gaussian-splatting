import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import read_write_model as colmap_rw

def plot_cameras_from_colmap(cameras_file, images_file):
    # Read the camera and image data from COLMAP's binary files
    cameras = colmap_rw.read_cameras_binary(cameras_file)
    images = colmap_rw.read_images_binary(images_file)
    
    # List to store camera positions and orientations
    camera_positions = []
    camera_orientations = []
    
    # Loop through the images and extract camera pose (rotation and translation)
    for image_id, image_data in images.items():
        R = image_data.qvec2rotmat()  # Rotation matrix from quaternion
        t = image_data.tvec  # Translation vector
        
        # Camera position in world coordinates (COLMAP uses camera-to-world transformation)
        camera_position = -R.T @ t
        camera_positions.append(camera_position)
        
        # Camera orientation (for plotting orientation vector)
        camera_orientations.append(R.T[:, 2])  # Z-axis of the camera

    camera_positions = np.array(camera_positions)
    camera_orientations = np.array(camera_orientations)
    
    # Plot the camera locations
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    
    
    # Plot camera orientations as quiver arrows
    ax.quiver(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
              camera_orientations[:, 0], camera_orientations[:, 1], camera_orientations[:, 2],
              length=0.5, color='b', label='Camera Orientations')
    # Plot camera positions
    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], c='r', marker='o', label='Camera Positions')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Locations and Orientations')
    
    plt.legend()
    plt.show()

# Example usage
cameras_file = '/mnt/c/MyFiles/Datasets/unreal_data_colmap/colmap_v1_radial/sparse/0/cameras.bin'
images_file = '/mnt/c/MyFiles/Datasets/unreal_data_colmap/colmap_v1_radial/sparse/0/images.bin'

plot_cameras_from_colmap(cameras_file, images_file)
