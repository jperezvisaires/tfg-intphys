import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as  np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Activation, BatchNormalization,
                                     Flatten, Dense, Conv2D, AveragePooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, LearningRateScheduler,
                                        ReduceLROnPlateau)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

# Training Parameters
batch_size = 32
epochs = 20
data_augmentation = True
num_classes = 10

# Substracting pixel mean improves accuracy
substract_pixel_mean = True

# Model selection
version = 1
n = 3

# Computed depth from model parameter n
if version == 1:
    depth = 6*n + 2
elif version == 2:
    depth = 9*n + 2

# Model name, depth and version
model_type = "ResNet{}_v{}".format(depth, version)
print("Model:"model_type)
