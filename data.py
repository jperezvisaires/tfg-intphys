import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import json

def data_load(num_scenes, num_initial=1, x_size=0, y_size=0):

    base_path = "./train-reduced/train"
    frames = 100
    scene_count = 0

    image_list = []
    depth_list = []
    mask_list = []
    meta_list = []

    for scene in range(num_initial, num_initial+num_scenes):

        if os.path.exists("{}/{:05d}".format(base_path, scene)):

            for i in range(frames):

                filename_image = "{}/{:05d}/scene/scene_{:03d}.png".format(base_path, scene, i + 1)
                image = imread(filename_image)  # Numpy array with RGB format.
                if (x_size + y_size) != 0:
                    image = resize(image, (x_size, y_size), preserve_range=True)
                image_list.append(image)

                filename_depth = "{}/{:05d}/depth/depth_{:03d}.png".format(base_path, scene, i + 1)
                depth = imread(filename_depth, as_gray = True)  # Numpy array with grayscale format.
                if (x_size + y_size) != 0:
                    depth = resize(depth, (x_size, y_size), preserve_range=True)
                depth_list.append(depth)

                filename_mask = "{}/{:05d}/masks/masks_{:03d}.png".format(base_path, scene, i + 1)
                mask = imread(filename_mask, as_gray = True)  # Numpy array with grayscale format.
                if (x_size + y_size) != 0:
                    mask = resize(mask, (x_size, y_size), preserve_range=True, anti_aliasing=False, order=0)
                mask_list.append(mask)

            filename_meta = "{}/{:05d}/status.json".format(base_path, scene)
            with open(filename_meta) as meta_json:
                meta = json.load(meta_json)
            meta_list.append(meta)

            scene_count += 1

        else:
            print("Error")

    image_array = np.array(image_list)
    depth_array = np.array(depth_list)
    mask_array = np.array(mask_list)

    depth_array = np.reshape(depth_array, (depth_array.shape[0], depth_array.shape[1], depth_array.shape[2], 1))
    mask_array = np.reshape(mask_array, (mask_array.shape[0], mask_array.shape[1], mask_array.shape[2], 1))

    mask_transform(mask_array, meta_list)

    print("Loaded scenes: ", scene_count)
    print("Shape of image array: ", image_array.shape)
    print("Shape of depth array: ", depth_array.shape)
    print("Shape of mask array: ", mask_array.shape)

    return image_array, depth_array, mask_array, meta_list, scene_count

def mask_transform(mask_array, meta_list):

    frames = 100
    offset = 0

    for meta in meta_list:

        for i in range(frames):

            mask = mask_array[offset+i, :, :, :]
            mask_meta = meta["frames"][i]["masks"]

            if "floor" in mask_meta:
                mask[mask == mask_meta["floor"]] = 0
            if "walls" in mask_meta:
                mask[mask == mask_meta["walls"]] = 1
            if "sky" in mask_meta:
                mask[mask == mask_meta["sky"]] = 2
            if "object_1" in mask_meta:
                mask[mask == mask_meta["object_1"]] = 3
            if "object_2" in mask_meta:
                mask[mask == mask_meta["object_2"]] = 3
            if "object_3" in mask_meta:
                mask[mask == mask_meta["object_3"]] = 3
            if "occluder_1" in mask_meta:
                mask[mask == mask_meta["occluder_1"]] = 4
            if "occluder_2" in mask_meta:
                mask[mask == mask_meta["occluder_2"]] = 4

        offset += 100

def make_sets(data_array, scene_count):

    frames = 100

    train_scenes = int(scene_count * 0.8)
    train_frames = train_scenes * frames
    vali_scenes = int(scene_count * 0.1)
    vali_frames = vali_scenes * frames
    test_scenes = scene_count - train_scenes - vali_scenes
    test_frames = test_scenes * frames

    train_data = data_array[:train_frames, :, :, :]
    vali_data = data_array[train_frames:(train_frames + vali_frames), :, :, :]
    test_data = data_array[(train_frames + vali_frames):, :, : , :]

    return train_data, vali_data, test_data
