# System libraries.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Avoid most Tensorflow warning errors.
import sys

# Maths and utilites.
import numpy as np
import h5py

# Keras and Tensorflow.
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

# Import my modules.
sys.path.append("./modules")
from segmentation_unet import get_unet_segmentation
from segmentation_partition import partition_image2seg
from segmentation_generators import unet_image2seg_object_generator, unet_image2seg_occlu_generator
from segmentation_losses import entropy_dice, dice, jaccard

def train_unet_image2seg_object(loss_name, 
                                block_number, 
                                load_loss_name="entropy_dice", 
                                load_block_number=1, 
                                num_scenes=1000, 
                                epochs=3, 
                                learning_rate=0, 
                                model_loading=False, 
                                model_summary=False, 
                                check_partition=False):
    
    # Check if GPU is being used.
    device_name = tf.test.gpu_device_name()

    if device_name != '/device:GPU:0':

        print('GPU device not found')
        USE_GPU = False

    else:

        print('Found GPU at: {}'.format(device_name))
        # !nvidia-smi  # GPU details.
        USE_GPU = True

    # Filename.
    filename_unet = "unet-image2seg-object-{}-{:02}".format(loss_name, block_number)
    print("Filename: " + filename_unet)
    
    
    # Create Unet model.
    unet_model = get_unet_segmentation()
    
    if model_summary:

        unet_model.summary()

    # Losses and metrics.
    dice_loss, dice_coeff = dice()
    jaccard_loss, jaccard_coeff = jaccard() 
    entropy_dice_loss = entropy_dice()

    # Compile model
    if loss_name == "L1":

        print("Selected Loss: " + loss_name)
        loss = L1_loss
        metrics_list = [dice_coeff, jaccard_coeff]

    elif loss_name == "L2":

        print("Selected Loss: " + loss_name)
        loss = L2_loss
        metrics_list = [dice_coeff, jaccard_coeff]

    elif loss_name == "entropy_dice":

        print("Selected Loss: " + loss_name)
        loss = entropy_dice_loss   
        metrics_list = [dice_coeff, jaccard_coeff, "mse"] 

    else:

        print("Select a valid loss.")

    if learning_rate:

        print("Learning Rate: " + str(learning_rate))
        optimizer = Adam(learning_rate=learning_rate)

    else:

        print("Learning Rate: " + str(1e-3))
        optimizer = Adam()

    unet_model.compile(loss=loss, optimizer=optimizer)

    # Create partition dictionaries
    path_hdf5 = "/content/temp_dataset/dataset-intphys-{:02}000.hdf5".format(block_number)
    partition, targets = partition_image2seg(num_initial=(block_number - 1) * 1000 + 1, 
                                             num_scenes=num_scenes, 
                                             path_hdf5=path_hdf5)

    # Parameters.
    params = {'dim': (288, 288),
          'path_hdf5': path_hdf5,
          'scale' : 0.5,
          'batch_size': 32,
          'num_classes': 1,
          'num_channels': 3,
          'shuffle': True}

    # Generators.
    training_generator = unet_image2seg_object_generator(partition["train"], targets, **params)
    validation_generator = unet_image2seg_object_generator(partition["validation"], targets, **params)    

    # Check partition integrity.
    if check_partition:
    
        print(partition)
        print(targets)
    
    # Select model for training.
    model = unet_model
    filename = filename_unet
    model_clean_weights = model.get_weights()

    # Configure Keras callbacks for training.
    model_checkpoint = ModelCheckpoint(filepath="./models/{}.h5".format(filename),
                                       save_best_only=False,
                                       verbose=1)

    csv_log = CSVLogger(filename="./logs/{}.csv".format(filename),
                        separator=";")

    cb_list = [model_checkpoint, csv_log]

    # Load precious model.

    if model_loading:
    
        model = load_model("./models/unet-image2seg-object-{}-{:02}.h5".format(load_loss_name, load_block_number), compile=False)
        print("Loaded model: " + "unet-image2seg-object-{}-{:02}.h5".format(load_loss_name, load_block_number))
        saved_weights = model.get_weights()
        model.compile(loss=loss, optimizer=optimizer)
        model.set_weights(saved_weights)

    # Clean weights before training
    if not model_loading:

        model.set_weights(model_clean_weights)

    # Generator training.
    train_history = model.fit(x=training_generator,
                              validation_data=validation_generator,
                              callbacks=cb_list,
                              epochs=epochs)
    
    return model

def train_unet_image2seg_occlu(loss_name, 
                               block_number, 
                               load_loss_name="entropy_dice", 
                               load_block_number=1, 
                               num_scenes=1000, 
                               epochs=3, 
                               learning_rate=0, 
                               model_loading=False, 
                               model_summary=False, 
                               check_partition=False):
    
    # Check if GPU is being used.
    device_name = tf.test.gpu_device_name()

    if device_name != '/device:GPU:0':

        print('GPU device not found')
        USE_GPU = False

    else:

        print('Found GPU at: {}'.format(device_name))
        # !nvidia-smi  # GPU details.
        USE_GPU = True

    # Filename.
    filename_unet = "unet-image2seg-occlu-{}-{:02}".format(loss_name, block_number)
    print("Filename: " + filename_unet)
    
    
    # Create Unet model.
    unet_model = get_unet_segmentation()
    
    if model_summary:

        unet_model.summary()

    # Losses and metrics.
    dice_loss, dice_coeff = dice()
    jaccard_loss, jaccard_coeff = jaccard() 
    entropy_dice_loss = entropy_dice()

    # Compile model
    if loss_name == "L1":

        print("Selected Loss: " + loss_name)
        loss = L1_loss
        metrics_list = [dice_coeff, jaccard_coeff]

    elif loss_name == "L2":

        print("Selected Loss: " + loss_name)
        loss = L2_loss
        metrics_list = [dice_coeff, jaccard_coeff]

    elif loss_name == "entropy_dice":

        print("Selected Loss: " + loss_name)
        loss = entropy_dice_loss   
        metrics_list = [dice_coeff, jaccard_coeff, "mse"] 

    else:

        print("Select a valid loss.")

    if learning_rate:

        print("Learning Rate: " + str(learning_rate))
        optimizer = Adam(learning_rate=learning_rate)

    else:

        print("Learning Rate: " + str(1e-3))
        optimizer = Adam()

    unet_model.compile(loss=loss, optimizer=optimizer)

    # Create partition dictionaries
    path_hdf5 = "/content/temp_dataset/dataset-intphys-{:02}000.hdf5".format(block_number)
    partition, targets = partition_image2seg(num_initial=(block_number - 1) * 1000 + 1, num_scenes=num_scenes, path_hdf5=path_hdf5)

    # Parameters.
    params = {'dim': (288, 288),
          'path_hdf5': path_hdf5,
          'scale' : 0.5,
          'batch_size': 32,
          'num_classes': 1,
          'num_channels': 3,
          'shuffle': True}

    # Generators.
    training_generator = unet_image2seg_occlu_generator(partition["train"], targets, **params)
    validation_generator = unet_image2seg_occlu_generator(partition["validation"], targets, **params)    

    # Check partition integrity.
    if check_partition:
    
        print(partition)
        print(targets)
    
    # Select model for training.
    model = unet_model
    filename = filename_unet
    model_clean_weights = model.get_weights()

    # Configure Keras callbacks for training.
    model_checkpoint = ModelCheckpoint(filepath="./models/{}.h5".format(filename),
                                       save_best_only=False,
                                       verbose=1)

    csv_log = CSVLogger(filename="./logs/{}.csv".format(filename),
                        separator=";")

    cb_list = [model_checkpoint, csv_log]

    # Load precious model.

    if model_loading:
    
        model = load_model("./models/unet-image2seg-occlu-{}-{:02}.h5".format(load_loss_name, load_block_number), compile=False)
        print("Loaded model: " + "unet-image2seg-occlu-{}-{:02}.h5".format(load_loss_name, load_block_number))
        saved_weights = model.get_weights()
        model.compile(loss=loss, optimizer=optimizer)
        model.set_weights(saved_weights)

    # Clean weights before training
    if not model_loading:

        model.set_weights(model_clean_weights)

    # Generator training.
    train_history = model.fit(x=training_generator,
                              validation_data=validation_generator,
                              callbacks=cb_list,
                              epochs=epochs)
    
    return model


