import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys

INTEL = True

if INTEL:
    import tensorflow as tf
    from keras.applications import ResNet50
    from keras.layers import (Input, Reshape, Flatten, Dense, Activation,
                              UpSampling2D, BatchNormalization, Conv2D)
    from keras.models import Model
    from keras.optimizers import Adam

else:
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.layers import (Input, Reshape, Flatten, Dense, Activation,
                                         UpSampling2D, BatchNormalization, Conv2D)
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

from data import data_load, make_sets

# Input shape of the image fed to the ResNet50 network.
resnet_input_shape = (64, 64, 3)
resnet_input = Input(shape = resnet_input_shape)

# Create the Resnet50 model with weights pretrained on ImageNet.
resnet50 = ResNet50(include_top = False,
                  weights = "imagenet",
                  input_tensor = resnet_input,
                  input_shape = resnet_input_shape)

# Select the layers of the first block only
resnet = Model(inputs=resnet50.input, outputs=resnet50.get_layer("conv2_block1_out").output)
# Freeze ResNet weights so they don't change during training.
for layer in resnet.layers:
    layer.trainable = False

resnet.summary()

# Define the autoencoder behind the ResNet neural network.
main_input = resnet_input

x = main_input
x = resnet(x)
x = Reshape(target_shape = (65536, ))(x)
x = Dense(units = 2048)(x)
x = Dense(units = 32768)(x)
x = Reshape(target_shape = (8, 8, 512))(x)
x = UpSampling2D(size = 2, data_format = "channels_last", interpolation = "nearest")(x)
x = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "same", data_format = "channels_last")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = UpSampling2D(size = (2, 2), data_format = "channels_last", interpolation = "nearest")(x)
x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = "same", data_format = "channels_last")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = UpSampling2D(size = (2, 2), data_format = "channels_last", interpolation = "nearest")(x)
x = Conv2D(filters = 1, kernel_size = (3, 3), strides = (1, 1), padding = "same", data_format = "channels_last")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

main_output = x

model = Model(inputs = main_input, outputs = main_output)

model.summary()

# Set how many scenes we want to load.
num_sims = 20

# Training parameters.
epochs = 2
batch_size = 1

# Load the four types of data linked to every scene.
image_array, depth_array, mask_array, meta_list, scene_count = data_load(num_sims, x_size=64, y_size=64)

# Divide the data in training, validation and testing sets.
image_data, image_vali, image_test = make_sets(image_array, scene_count)
depth_data, depth_vali, depth_test = make_sets(depth_array, scene_count)
mask_data, mask_vali, mask_test = make_sets(mask_array, scene_count)

# Compile and train the model
loss = "mse"
optimizer = Adam()

model.compile(loss = loss, optimizer = optimizer)

train_history = model.fit(image_data, mask_data,
                          epochs = epochs,
                          batch_size = batch_size,
                          verbose = 1,
                          validation_data = (image_vali, mask_vali),
                          shuffle = True)

model.save("./models/resnet.h5")
