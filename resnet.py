import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as  np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Activation, BatchNormalization,
                                     Flatten, Dense, Conv2D, AveragePooling2D,
                                     add)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, LearningRateScheduler,
                                        ReduceLROnPlateau)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

# Training Parameters.
batch_size = 32
epochs = 20
data_augmentation = True
num_classes = 10

# Subtracting pixel mean improves accuracy.
subtract_pixel_mean = True

# Model selection.
version = 1
n = 3

# Computed depth from model parameter n.
if version == 1:
    depth = 6*n + 2
elif version == 2:
    depth = 9*n + 2

# Model name, depth and version.
model_type = "ResNet{}_v{}".format(depth, version)
print("Model:", model_type)

# Load the dataset.

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# If subtract pixel mean is enabled.
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_test_mean = np.mean(x_test, axis=0)
    x_train -= x_train_mean
    x_test -= x_test_mean

# Print some info.
print("x_train shape:", x_train.shape)
print("Train samples:", x_train.shape[0])
print("Test samples:", x_test.shape[0])

def resnet_layer(layer_input,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation="relu",
                 batch_normalization=True,
                 conv_first=True
                 ):

    conv = Conv2D(filters=num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding="same",
                  kernel_initializer="he_normal",
                  kernel_regularizer=l2(1e-4))

    x = layer_input
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    layer_output = x

    return layer_output

def resnet_v1(input_shape, depth):

    # Check if depth is valid.
    if (depth - 2) % 6 != 0:
        raise ValueError("depth should be 6n+2")

    # Start model definition
    num_filters = 16
    num_res_blocks = int((depth-2) / 6)

    model_input = Input(shape=input_shape)
    x = resnet_layer(layer_input=model_input)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides=1
            if stack > 0 and res_block == 0:  # first layer, but not of the first stack
                strides = 2  # downsample
            x1 = resnet_layer(layer_input=x,
                             num_filters=num_filters,
                             strides=strides)
            x1 = resnet_layer(layer_input=x1,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:
                # linear projection for changed dimensions
                x2 = resnet_layer(layer_input=x,
                                  num_filters=num_filters,
                                  kernel_size=1,
                                  strides=strides,
                                  activation=None,
                                  batch_normalization=False)
            else:
                x2 = x
            x = add([x1,x2])
            x = Activation("relu")(x)
        num_filters *= 2
