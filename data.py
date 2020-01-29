import os
import numpy as np
import imageio
import json

def data_load(num_initial, num_scenes):

    base_path = "/media/jon/files/TFG/train.1/train"
    frames = 100
    scene_count = 0

    image_list = []
    depth_list = []
    mask_list = []
    meta_list = []

    for scene in range(num_initial, num_scenes):

        if os.path.exists("{}/{:05d}".format(base_path, scene)):

            for i in range(0, frames):

                filename_image = "{}/{:05d}/scene/scene_{:03d}.png".format(base_path, scene, i + 1)
                image = imageio.imread(filename_image)  # Numpy array with RGB format.
                image_list.append(image)

                filename_depth = "{}/{:05d}/depth/depth_{:03d}.png".format(base_path, scene, i + 1)
                depth = imageio.imread(filename_depth, as_gray = True)  # Numpy array with grayscale format.
                depth_list.append(depth)

                filename_mask = "{}/{:05d}/masks/masks_{:03d}.png".format(base_path, scene, i + 1)
                mask = imageio.imread(filename_mask, as_gray = True)  # Numpy array with grayscale format.
                mask_list.append(mask)

            filename_meta = "{}/{:05d}/status.json".format(base_path, scene)
            with open(filename_meta) as meta_json:
                meta = json.load(meta_json)
            meta_list.append(meta)

            scene_count = scene_count + 1

    image_array = np.array(image_list)
    depth_array = np.array(depth_list)
    mask_array = np.array(mask_list)

    depth_array = np.reshape(depth_array, (depth_array.shape[0], depth_array.shape[1], depth_array.shape[2], 1))
    mask_array = np.reshape(mask_array, (mask_array.shape[0], mask_array.shape[1], mask_array.shape[2], 1))

    print("Loaded scenes: ", scene_count)
    print("Shape of image array: ", image_array.shape)
    print("Shape of depth array: ", depth_array.shape)
    print("Shape of mask array: ", mask_array.shape)

    return image_array, depth_array, mask_array, meta_list, scene_count

def make_sets(data_array, scene_count):

    frames = 100

    train_scenes = int(scene_count * 0.7) *  frames
    vali_scenes = int(scene_count * 0.2) * frames
    test_scenes = (scene_count * frames - (train_scenes + vali_scenes))

    train_data = data_array[:train_scenes, :, :, :]
    vali_data = data_array[train_scenes:(train_scenes + vali_scenes), :, :, :]
    test_data = data_array[(train_scenes + vali_scenes):, :, : , :]

    print("Scenes for training: ", train_scenes)
    print("Scenes for validation: ", vali_scenes)
    print("Scenes for testing: ", test_scenes)

    return train_data, vali_data, test_data
