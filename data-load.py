import os
import numpy as np
import imageio
import json
import matplotlib.pyplot as plt

def data_load(num_initial, num_sims):

    base_path = "/media/jon/files/TFG/train.1/train"
    frames = 100
    sim_count = 0

    image_list = []
    depth_list = []
    mask_list = []
    meta_list = []

    for sim in range(num_initial, num_sims):

        if os.path.exists("{}/{:05d}".format(base_path, sim)):

            for i in range(0, frames):

                filename_image = "{}/{:05d}/scene/scene_{:03d}.png".format(base_path, sim, i + 1)
                image = imageio.imread(filename_image)  # Numpy array with RGB format.
                image_list.append(image)

                filename_depth = "{}/{:05d}/depth/depth_{:03d}.png".format(base_path, sim, i + 1)
                depth = imageio.imread(filename_depth, as_gray = True)  # Numpy array with grayscale format.
                depth_list.append(depth)

                filename_mask = "{}/{:05d}/masks/masks_{:03d}.png".format(base_path, sim, i + 1)
                mask = imageio.imread(filename_mask, as_gray = True)  # Numpy array with grayscale format.
                mask_list.append(mask)

            filename_meta = "{}/{:05d}/status.json".format(base_path, sim)
            meta = json.load(filename_meta)
            meta_list.append(meta)

            sim_count = sim_count + 1

    print("Loaded scenes: ", sim_count)

    return image_list, depth_list, mask_list, meta_list
