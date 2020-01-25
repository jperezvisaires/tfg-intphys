import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Reshape, Flatten, Dense, UpSampling2D, BatchNormalization, ReLU, Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Input shape of the image fed to the ResNet50 network.
resnet_input_shape = (64, 64, 3)
resnet_input = Input(shape = resnet_input_shape)

# Create the Resnet50 model with weights pretrained on ImageNet.
resnet = ResNet50(include_top = False,
                  weights = "imagenet",
                  input_tensor = resnet_input,
                  input_shape = resnet_input_shape)

# Freeze ResNet weights so they don't change during training.
for layer in resnet.layers:
    layer.trainable = False

# resnet.summary()

loss = "mse"
optimizer = Adam

# Define the autoencoder behind the ResNet neural network.
main_input = resnet_input

x = main_input
x = resnet(x)
x = Reshape(target_shape = (8192, ))(x)
x = Dense(units = 128)(x)
x = Dense(units = 8192)(x)
x = Reshape(target_shape = (8, 8, 128))(x)
x = UpSampling2D(size = (2, 2), data_format = "channels_last", interpolation = "nearest")(x)
x = Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = "same", data_format = "channels_last")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = UpSampling2D(size = (2, 2), data_format = "channels_last", interpolation = "nearest")(x)
x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = "same", data_format = "channels_last")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = UpSampling2D(size = (2, 2), data_format = "channels_last", interpolation = "nearest")(x)
x = Conv2D(filters = 3, kernel_size = (3, 3), strides = (1, 1), padding = "same", data_format = "channels_last")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Activation("sigmoid")(x)

main_output = x

model = Model(inputs = main_input, outputs = main_output)

model.summary()
