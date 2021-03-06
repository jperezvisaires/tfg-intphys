import numpy as np
import h5py

BASE_PATH = "train"

def data_dictionaries_segmentation(num_scenes, path_hdf5, num_initial=1, test=False, base_path=BASE_PATH):

    frames = 100

    list_samples = []
    list_targets_mask = []
    
    for scene in range(num_initial, num_initial+num_scenes):
            
        scene_path = "{}/{:05d}".format(base_path, scene)
            
        with h5py.File(path_hdf5, "r") as f:    
            
            if scene_path in f:
        
                for i in range(frames):
            
                    path_image = scene_path + "/scene/scene_{:03d}".format(i+1)
                    list_samples.append(path_image)
            
                    path_mask = scene_path + "/masks/masks_{:03d}".format(i+1)
                    list_targets_mask.append(path_mask)

    partition = partition_dictionary(list_samples, num_scenes, test)  

    targets = targets_dictionary(list_samples, list_targets_mask)

    return partition, targets

def data_dictionaries_depth(num_scenes, path_hdf5, num_initial=1, test=False, base_path=BASE_PATH):

    frames = 100

    list_samples = []
    list_targets_depth = []
    
    for scene in range(num_initial, num_initial+num_scenes):
            
        scene_path = "{}/{:05d}".format(base_path, scene)
            
        with h5py.File(path_hdf5, "r") as f:    
            
            if scene_path in f:
        
                for i in range(frames):
            
                    path_image = scene_path + "/scene/scene_{:03d}".format(i+1)
                    list_samples.append(path_image)
            
                    path_depth = scene_path + "/depth/depth_{:03d}".format(i+1)
                    list_targets_depth.append(path_depth)

    partition = partition_dictionary(list_samples, num_scenes, test)  

    targets = targets_dictionary(list_samples, list_targets_depth)

    return partition, targets


def data_dictionaries_prediction(num_scenes, path_hdf5, input_frames=3, prediction_frame=5, space_frames=5, num_initial=1, test=False, base_path=BASE_PATH):

    frames = 100
    last_frame = frames - prediction_frame - (input_frames - 1) * space_frames

    list_samples = []
    list_targets = []
    
    for scene in range(num_initial, num_initial+num_scenes):
            
        scene_path = "{}/{:05d}".format(base_path, scene)
            
        with h5py.File(path_hdf5, "r") as f:    
            
            if scene_path in f:
        
                for i in range(last_frame):

                    list_samples_memory = []
                    
                    for j in range(input_frames):
                    
                        path_image = scene_path + "/masks/masks_{:03d}".format((i+1) + (j*space_frames))
                        list_samples_memory.append(path_image)
                    
                    list_samples.append(list_samples_memory)

                    path_image = scene_path + "/masks/masks_{:03d}".format((i+1) + (input_frames - 1) * space_frames + prediction_frame)
                    list_targets.append(path_image)

    partition = partition_dictionary_prediction(list_samples, num_scenes, test, last_frame, prediction_frame)

    targets = targets_dictionary_prediction(list_samples, list_targets)

    return partition, targets


def partition_dictionary(list_samples, num_scenes, test):

    frames = 100

    if test:
        
        train_scenes = int(num_scenes * 0.8)
        train_frames = train_scenes * frames
    
        vali_scenes = int(num_scenes * 0.1)
        vali_frames = vali_scenes * frames

        list_samples_train = list_samples[:train_frames]
        list_samples_vali = list_samples[train_frames:(train_frames + vali_frames)]
        list_samples_test = list_samples[(train_frames + vali_frames):]
    
        partition = {"train": list_samples_train,
                     "validation": list_samples_vali,
                     "test": list_samples_test}

    else:
        
        train_scenes = int(num_scenes * 0.9)
        train_frames = train_scenes * frames
        
        list_samples_train = list_samples[:train_frames]
        list_samples_vali = list_samples[train_frames:]    
        
        partition = {"train": list_samples_train,
                     "validation": list_samples_vali}
    
    return partition

def partition_dictionary_prediction(list_samples, num_scenes, test, last_frame=85, prediction_frame=5):

    frames = 100 - (100 - last_frame)

    if test:
        
        train_scenes = int(num_scenes * 0.8)
        train_frames = train_scenes * frames
    
        vali_scenes = int(num_scenes * 0.1)
        vali_frames = vali_scenes * frames

        list_samples_train = list_samples[:train_frames]
        list_samples_vali = list_samples[train_frames:(train_frames + vali_frames)]
        list_samples_test = list_samples[(train_frames + vali_frames):]
    
        partition = {"train": list_samples_train,
                     "validation": list_samples_vali,
                     "test": list_samples_test}

    else:
        
        train_scenes = int(num_scenes * 0.9)
        train_frames = train_scenes * frames

        list_samples_train = list_samples[:train_frames]
        list_samples_vali = list_samples[train_frames:]    
        
        partition = {"train": list_samples_train,
                     "validation": list_samples_vali}
    
    return partition

def targets_dictionary(list_samples, list_targets):
    
    targets = {}
    for i in range(len(list_samples)):
        targets[list_samples[i]] = list_targets[i]
    
    return targets

def targets_dictionary_prediction(list_samples, list_targets):
    
    targets = {}
    for i in range(len(list_samples)):
        targets[list_samples[i][0]] = list_targets[i]
    
    return targets