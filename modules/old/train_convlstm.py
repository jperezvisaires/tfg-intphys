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
from prediction_convlstm import get_convlstm
from prediction_partition import partition_seg2seg, partition_image2seg, partition_depth2seg
from prediction_generators import convlstm_seg2seg_generator
from prediction_losses import intensity, entropy_dice, dice, jaccard

def train_convlstm_seg2seg_short(loss_name, 
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
    filename_convlstm = "convlstm-pred-seg2seg-short-{}-{:02}".format(loss_name, block_number)
    print("Filename: " + filename_convlstm)

    # Create Unet model.
    input_frames = 5
    num_channels = 1
    final_filter = 1
    space_frames = 3
    prediction_frame = 3
    convlstm_model = get_convlstm(input_frames, num_channels, final_filter)

    if model_summary:

        convlstm_model.summary()

    # Define losses and metrics
    entropy_dice_loss = entropy_dice()
    dice_loss, dice_coeff = dice()
    jaccard_loss, jaccard_coeff = jaccard() 

    # Compile model
    if loss_name == "mse":

        print("Selected Loss: " + loss_name)
        loss = "mse"
        metrics_list = [dice_coeff, "mae", "binary_crossentropy"]

    elif loss_name == "mae":

        print("Selected Loss: " + loss_name)
        loss = "mae"
        metrics_list = [dice_coeff, "mse", "binary_crossentropy"]

    elif loss_name == "binary_crossentropy":

        print("Selected Loss: " + loss_name)
        loss = "binary_crossentropy"
        metrics_list = [dice_coeff, "mse", "mae"]

    elif loss_name == "entropy_dice":

        print("Selected Loss: " + loss_name)
        loss = entropy_dice_loss   
        metrics_list = [dice_coeff, "mse", "mae", "binary_crossentropy"] 

    if learning_rate:

        print("Learning Rate: " + str(learning_rate))
        optimizer = Adam(learning_rate=learning_rate)

    else:

        print("Learning Rate: " + str(1e-3))
        optimizer = Adam()

    convlstm_model.compile(loss=loss, optimizer=optimizer, metrics=metrics_list)

    # Create partition dictionaries
    path_hdf5 = "/content/temp_dataset/dataset-intphys-{:02}000.hdf5".format(block_number)
    partition, targets = partition_seg2seg(num_initial=(block_number - 1) * 1000 + 1, 
                                           num_scenes=num_scenes, 
                                           input_frames=input_frames, 
                                           space_frames=space_frames,
                                           prediction_frame=prediction_frame, 
                                           path_hdf5=path_hdf5)

    # Parameters.
    params = {'dim': (288, 288),
              'path_hdf5': path_hdf5,
              'scale' : 0.5,
              'batch_size': 16,
              'input_frames': input_frames,
              'num_channels': num_channels,
              'shuffle': True}

    # Generators.
    training_generator = convlstm_seg2seg_generator(partition["train"], targets, **params)
    validation_generator = convlstm_seg2seg_generator(partition["validation"], targets, **params)

    # Check partition integrity.
    if check_partition:
    
        print(partition)
        print(targets)

    # Select model for training.
    model = convlstm_model
    filename = filename_convlstm
    model_clean_weights = model.get_weights()

    # Configure Keras callbacks for training.
    model_checkpoint = ModelCheckpoint(filepath="./models/{}.h5".format(filename),
                                       save_best_only=True,
                                       verbose=1)

    csv_log = CSVLogger(filename="./logs/{}.csv".format(filename),
                        separator=";")

    cb_list = [model_checkpoint, csv_log]

    # Load precious model.

    if model_loading:
    
        model = load_model("./models/convlstm-pred-seg2seg-short-{}-{:02}.h5".format(load_loss_name, load_block_number), compile=False)
        print("Loaded model: " + "convlstm-pred-seg2seg-short-{}-{:02}.h5".format(load_loss_name, load_block_number))
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