import numpy as np
import collections
import json
import os
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2, imutils




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
            tvec = np.array([cam["Transform"]["x"], cam["Transform"]["y"], cam["Transform"]["z"]])

            # Extract and convert the rotation matrix to a quaternion (why not use the DCM directly?)
            R = np.array([
                [cam["Transform"]["r11"], cam["Transform"]["r12"], cam["Transform"]["r13"]],
                [cam["Transform"]["r21"], cam["Transform"]["r22"], cam["Transform"]["r23"]],
                [cam["Transform"]["r31"], cam["Transform"]["r32"], cam["Transform"]["r33"]]
            ])
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
                R=R, path=path[:-5]+'.jpeg'
            )
    return images


if __name__ == "__main__":
    # Run tests
    # test_qualisys_json_loader()


    # Visualize cameras for test case

    # Load extrinsic data
    path = '/mnt/c/MyFiles/Datasets/UnrealData3'
    directory = os.path.join(path, 'train_uniform')
    extrinsic_files = os.listdir(directory)
    extrinsic_files = [os.path.join(directory, f) for f in extrinsic_files if '.json' in f]
    # print(extrinsic_files)
    # extrinsic_file = os.path.join(directory, 'Test_Recording_Qualisys_MoCap_Miqus_Hybrid_Motionlab_UniversityOfAgder_Norway_0001.json')
    extrinsics = read_extrinsics_json(extrinsic_files)
    meta_data_file = path + '/metadata.json'
    metadata=json.load(open(meta_data_file))

    # # Sort image files
    sorted_files = sorted_image_files(directory)
    # # Create a new dictionary for images sorted by their file names
    sorted_images = {}
    for file in sorted_files:
        # Extract camera number from file name
        camera_number = int(re.search(r'img_(\d+)', file).group(1))
        if camera_number in extrinsics:
            sorted_images[camera_number] = extrinsics[camera_number]

    # Visualize cameras with sorted extrinsic parameters
    # visualize_cameras(sorted_images)

    # # Enable interactive mode for 3D visualization
    # plt.ion()
    for v in [0,1]:
        p = visualize_cameras_3d(sorted_images, metadata["Radius"]//20, v)
        resolution_value = 200
        p.savefig(path+"/vis_{}_{}_{}_{}.png".format('Scene_01', metadata['Radius'], metadata['TrueNumRenders_uniform'],v), format="png", dpi=resolution_value)
    all_img = []
    cp = .23
    for v in [0,1]:
        im = cv2.imread(path+"/vis_{}_{}_{}_{}.png".format('Scene_01', metadata['Radius'], metadata['TrueNumRenders_uniform'],v))
        h,w,_ = im.shape
        im = im[int(h*cp):h-int(h*cp), int(w*cp):h-int(w*cp),:]
        all_img.append(im)
    img = cv2.hconcat(all_img)
    cv2.imwrite(path+"/vis_{}_{}_{}_{}.png".format('Scene_01', metadata['Radius'], metadata['TrueNumRenders_uniform'],'all'), img)
    # # Keep the plot open
    # plt.show(block=True)