import tensorflow as tf
import numpy as np
import os
import sys

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Reshape, Flatten, Dense, UpSampling2D, BatchNormalization, ReLU, Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from data import data_load, make_sets

# Input shape of the image fed to the ResNet50 network.
resnet_input_shape = (288, 288, 3)
resnet_input = Input(shape = resnet_input_shape)

# Create the Resnet50 model with weights pretrained on ImageNet.
resnet = ResNet50(include_top = False,
                  weights = "imagenet",
                  input_tensor = resnet_input,
                  input_shape = resnet_input_shape)

# Freeze ResNet weights so they don't change during training.
for layer in resnet.layers:
    layer.trainable = False

resnet.summary()

# Define the autoencoder behind the ResNet neural network.
main_input = resnet_input

x = main_input
x = resnet(x)
x = Reshape(target_shape = (165888, ))(x)
# x = Dense(units = 2592)(x)
# x = Dense(units = 165888)(x)
x = Reshape(target_shape = (36, 36, 128))(x)
x = UpSampling2D(size = (2, 2), data_format = "channels_last", interpolation = "nearest")(x)
x = Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = "same", data_format = "channels_last")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = UpSampling2D(size = (2, 2), data_format = "channels_last", interpolation = "nearest")(x)
x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = "same", data_format = "channels_last")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = UpSampling2D(size = (2, 2), data_format = "channels_last", interpolation = "nearest")(x)
x = Conv2D(filters = 1, kernel_size = (3, 3), strides = (1, 1), padding = "same", data_format = "channels_last")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Activation("sigmoid")(x)

main_output = x

model = Model(inputs = main_input, outputs = main_output)

model.summary()

# Set how many scenes we want to load.
num_initial = 0
num_sims = 50

# Training parameters.
epochs = 10
batch_size = 4

# Load the four types of data linked to every scene.
image_array, depth_array, mask_array, meta_list, scene_count = data_load(num_initial, num_sims)

# Divide the data in training, validation and testing sets.
image_data, image_vali, image_test = make_sets(image_array, scene_count)
depth_data, depth_vali, depth_test = make_sets(depth_array, scene_count)
mask_data, mask_vali, mask_test = make_sets(mask_array, scene_count)

# Compile and train the model
loss = "mse"
optimizer = Adam()

model.compile(loss = loss, optimizer = optimizer)

train_history = model.fit(image_data, depth_data,
                          epochs = epochs,
                          batch_size = batch_size,
                          verbose = 1,
                          validation_data = (image_vali, depth_vali),
                          shuffle = True)
