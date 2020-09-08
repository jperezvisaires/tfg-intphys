import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import ConvLSTM2D, Conv2D
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout, TimeDistributed

# Conv layer.
def conv_layer(x, filters, kernel_size=5, activation="relu", batch_norm=True):

    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=1,
               padding="same",
               activation=activation)(x)
    
    if batch_norm:

        x = BatchNormalization()(x)
    
    x = Dropout(0.1)(x)
    
    return x

    

# ConvLSTM layer.
def convlstm_layer(x, filters, kernel_size=5, strides=1, activation="tanh", return_sequences=True, batch_norm=True):

    x = ConvLSTM2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding="same",
                   activation=activation,
                   dropout=0.1,
                   recurrent_dropout=0.15,
                   go_backwards=False,
                   return_sequences=return_sequences)(x)
    
    if batch_norm:

        x = BatchNormalization()(x)
    
    return x

# ConvLSTM prediction model.
def convlstm_model(input_size, 
                   scale, 
                   input_frames,
                   final_filter,
                   final_activation,
                   dropout, 
                   batch_norm):

    scaled_input = (input_frames, int(input_size[0] * scale), int(input_size[1] * scale), input_size[2])       
    
    convlstm_input = Input(shape=(scaled_input))
    
    convlstm1 = convlstm_layer(x=convlstm_input, filters=32, kernel_size=5)
    pool1 = MaxPooling3D(pool_size=(1,2,2), padding="same")(convlstm1)
    
    convlstm2 = convlstm_layer(x=pool1, filters=32, kernel_size=5)
    pool2 = MaxPooling3D(pool_size=(1,2,2), padding="same")(convlstm2)
    
    convlstm3 = convlstm_layer(x=pool2, filters=64, kernel_size=5)
    pool3 = MaxPooling3D(pool_size=(1,2,2), padding="same")(convlstm3)
    
    convlstm4 = convlstm_layer(x=pool3, filters=64, kernel_size=5)
    pool4 = MaxPooling3D(pool_size=(1,2,2), padding="same")(convlstm4)
    
    convlstm5 = convlstm_layer(x=pool4, filters=128, kernel_size=5)
    up5 = UpSampling3D(size=(1,2,2))(convlstm5)
    
    convlstm6 = convlstm_layer(x=up5, filters=64, kernel_size=5)
    up6 = UpSampling3D(size=(1,2,2))(convlstm6)
    
    convlstm7 = convlstm_layer(x=up6, filters=64, kernel_size=5)
    up7 = UpSampling3D(size=(1,2,2))(convlstm7)
    
    convlstm8 = convlstm_layer(x=up7, filters=32, kernel_size=5)
    up8 = UpSampling3D(size=(1,2,2))(convlstm8)
    
    convlstm9 = convlstm_layer(x=up8, filters=32, kernel_size=5, return_sequences=False)

    conv10 = conv_layer(x=convlstm9, filters=final_filter, kernel_size=1, activation=final_activation)

    convlstm_output = conv10

    model = Model(inputs=convlstm_input, outputs=convlstm_output)

    return model

def convlstm_model_skip(input_size, 
                        scale, 
                        input_frames,
                        final_filter,
                        final_activation, 
                        batch_norm):

    scaled_input = (input_frames, int(input_size[0] * scale), int(input_size[1] * scale), input_size[2])       
    
    convlstm_input = Input(shape=(scaled_input))
    
    convlstm1 = convlstm_layer(x=convlstm_input, filters=32, kernel_size=7)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=2, padding="same"))(convlstm1)
    
    convlstm2 = convlstm_layer(x=pool1, filters=32, kernel_size=7)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=2, padding="same"))(convlstm2)
    
    convlstm3 = convlstm_layer(x=pool2, filters=64, kernel_size=5)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=2, padding="same"))(convlstm3)
    
    convlstm4 = convlstm_layer(x=pool3, filters=64, kernel_size=5)
    pool4 = TimeDistributed(MaxPooling2D(pool_size=2, padding="same"))(convlstm4)
    
    convlstm5_1 = convlstm_layer(x=pool4, filters=128, kernel_size=3)
    convlstm5_2 = convlstm_layer(x=convlstm5_1, filters=128, kernel_size=3)
    up5 = TimeDistributed(UpSampling2D(size=2))(convlstm5_2)
    
    convlstm6 = convlstm_layer(x=up5, filters=64, kernel_size=5)
    concat6 = Concatenate(axis=-1)([convlstm4, convlstm6])
    up6 = TimeDistributed(UpSampling2D(size=2))(concat6)
    
    convlstm7 = convlstm_layer(x=up6, filters=64, kernel_size=5)
    concat7 = Concatenate(axis=-1)([convlstm3, convlstm7])
    up7 = TimeDistributed(UpSampling2D(size=2))(concat7)
    
    convlstm8 = convlstm_layer(x=up7, filters=32, kernel_size=7)
    concat8 = Concatenate(axis=-1)([convlstm2, convlstm8])
    up8 = TimeDistributed(UpSampling2D(size=2))(concat8)
    
    convlstm9_1 = convlstm_layer(x=up8, filters=32, kernel_size=7)
    concat9 = Concatenate(axis=-1)([convlstm1, convlstm9_1])
    convlstm9_2 = convlstm_layer(x=concat9, filters=32, kernel_size=7, return_sequences=False)

    conv10 = conv_layer(x=convlstm9_2, filters=final_filter, kernel_size=1, activation=final_activation)

    convlstm_output = conv10

    model = Model(inputs=convlstm_input, outputs=convlstm_output)

    return model

def convlstm_model_simple(input_size, 
                          scale, 
                          input_frames,
                          final_filter,
                          final_activation, 
                          batch_norm):

    scaled_input = (input_frames, int(input_size[0] * scale), int(input_size[1] * scale), input_size[2])       
    
    convlstm_input = Input(shape=(scaled_input))
    
    convlstm1 = convlstm_layer(x=convlstm_input, filters=64, kernel_size=5)
    convlstm2 = convlstm_layer(x=convlstm1, filters=64, kernel_size=5)
    convlstm3 = convlstm_layer(x=convlstm2, filters=64, kernel_size=5)
    convlstm4 = convlstm_layer(x=convlstm3, filters=64, kernel_size=5)
    convlstm5 = convlstm_layer(x=convlstm4, filters=64, kernel_size=5, return_sequences=False)

    conv6 = conv_layer(x=convlstm5, filters=final_filter, kernel_size=1, activation=final_activation)

    convlstm_output = conv6

    model = Model(inputs=convlstm_input, outputs=convlstm_output)

    return model

# Get ConvLSTM model.

def get_convlstm_skip():

    params = {'input_size': (288, 288, 1),
              'scale': 0.5,
              'input_frames': 4,
              'final_filter': 1,
              'final_activation': "sigmoid",
              'batch_norm': True}

    model = convlstm_model_skip(**params)

    return model

def get_convlstm_simple():

    params = {'input_size': (288, 288, 1),
              'scale': 0.5,
              'input_frames': 4,
              'final_filter': 1,
              'final_activation': "sigmoid",
              'batch_norm': True}

    model = convlstm_model_simple(**params)

    return model

def get_convlstm():

    params = {'input_size': (288, 288, 2),
              'scale': 0.5,
              'input_frames': 4,
              'final_filter': 1,
              'final_activation': "tanh",
              'dropout': 0.0,
              'batch_norm': True}

    model = convlstm_model(**params)

    return model