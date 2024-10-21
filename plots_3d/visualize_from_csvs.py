import numpy as np
def read_csv_to_rotations_and_translations(file_path):
    # Lists to store rotations and translations
    rotations = []
    translations = []

    # Read the CSV file
    with open(file_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            # Split the line into numbers
            values = list(map(float, line.strip().split(',')))

            # Extract rotation matrix (first 9 values)
            rotation_matrix = np.array(values[:9]).reshape(3, 3)
            rotations.append(rotation_matrix)

            # Extract translation vector (next 3 values)
            translation_vector = np.array(values[9:12])
            translations.append(translation_vector)

    return rotations, translations

# Example usage
file_path = '/home/anurag/codes/gaussian_splatting_kristian_fork/gaussian-splatting/plots_3d/output_unreal_172.csv'  # Replace with the actual file path
# file_path = '/home/anurag/codes/gaussian_splatting_kristian_fork/gaussian-splatting/plots_3d/output_colmap.csv'  # Replace with the actual file path
# file_path = '/home/anurag/codes/gaussian_splatting_kristian_fork/gaussian-splatting/plots_3d/output_blender.csv'  # Replace with the actual file path
rotations, translations = read_csv_to_rotations_and_translations(file_path)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_translations_3d(rotations, translations, arrow_length_mm=0.1):
    # Create a new 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract the x, y, z coordinates from the translations list
    x_vals = [t[0] for t in translations]
    y_vals = [t[1] for t in translations]
    z_vals = [t[2] for t in translations]

    # Plot the points in 3D space
    ax.scatter(x_vals, y_vals, z_vals, c='r', marker='o')

    # Label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set a title
    ax.set_title('3D Plot of Translations')

    for i in range(len(translations)):
        R = np.array(rotations[i])
        T = np.array(translations[i])
        camera_orientation = R.T @ np.array([0, 0, -1])
        line_end = T + (arrow_length_mm * camera_orientation)
        ax.plot([T[0], line_end[0]],
                [T[1], line_end[1]],
                [T[2], line_end[2]], 'b-')
    ax.set_zlim(ax.get_zlim()[0], ax.get_zlim()[1])
    # Show the plot
    plt.show()
plot_translations_3d(rotations, translations)

# import json

# def extract_rotation_and_location(json_file):
#     # Open and read the JSON file
#     with open(json_file, 'r') as file:
#         data = json.load(file)
    
#     # Extract the rotation matrix and flatten it to 1D list
#     rotation_matrix = [item for sublist in data['rotation_matrix'] for item in sublist]
    
#     # Extract the global location
#     global_location = data['global_location']
    
#     # Combine rotation and location into a single 1D list
#     combined_list = rotation_matrix + global_location
    
#     return combined_list

# # Example usage
# json_file = 'camera_data.json'  # Replace with the actual file path
# # result = extract_rotation_and_location(json_file)
# # print(result)
# blender_path = '/mnt/c/MyFiles/Datasets/blender_test_data/'
# json_files = []
# infos = []
# import os
# # Walk through the directory and find all .json files
# for root, dirs, files in os.walk(blender_path):
#     for file in files:
#         if file.endswith(".json"):
#             infos.append(extract_rotation_and_location(os.path.join(root, file)))
# infos = np.array(infos)
# np.savetxt("output_blender.csv", infos, delimiter=",", fmt='%f')