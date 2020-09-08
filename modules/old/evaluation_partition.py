import numpy as np
import h5py

def partition_dictionary(list_samples):
        
    partition = {"train": list_samples}
    
    return partition

def targets_dictionary(list_samples, list_targets):
    
    targets = {}
    for i in range(len(list_samples)):
        targets[list_samples[i][0]] = list_targets[i]
    
    return targets

def partition_seg2seg(path_hdf5, scene_path, input_frames=4, prediction_frame=5, space_frames=2):

    frames = 100
    last_frame = frames - prediction_frame - (input_frames - 1) * space_frames

    list_samples = []
    list_targets = []
            
    with h5py.File(path_hdf5, "r") as f:    
            
        if scene_path in f:
        
            for i in range(last_frame):

                list_samples_memory = []
                    
                for j in range(input_frames):
                    
                    path_image = scene_path + "/scene/scene_{:03d}.png".format((i+1) + (j * space_frames))
                    list_samples_memory.append(path_image)
                    
                list_samples.append(list_samples_memory)

                path_image = scene_path + "/scene/scene_{:03d}.png".format((i+1) + (input_frames - 1) * space_frames + prediction_frame)
                list_targets.append(path_image)

    targets = targets_dictionary(list_samples, list_targets)
    partition = partition_dictionary(list_samples)

    return partition, targets