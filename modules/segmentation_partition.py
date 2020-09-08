import numpy as np
import h5py

base_path = "train"

def partition_image2seg(num_scenes, path_hdf5, num_initial=1, test=False, base_path=base_path):

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

def targets_dictionary(list_samples, list_targets):
    
    targets = {}
    for i in range(len(list_samples)):
        targets[list_samples[i]] = list_targets[i]
    
    return targets