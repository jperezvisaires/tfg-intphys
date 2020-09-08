import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose  
from tensorflow.keras.layers import MaxPool2D, UpSampling2D, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout

# Unet Layers.

def unet_conv_layer(x, filters, kernel_size, kernel_initializer, activation, batch_norm):
    
    x = Conv2D(filters=filters, 
               kernel_size=kernel_size, 
               kernel_initializer=kernel_initializer,
               padding="same")(x)

    x = Activation(activation)(x)

    if batch_norm:

        x = BatchNormalization()(x)
    
    return x

def unet_max_layer(x):

    x = MaxPool2D(pool_size=2, 
                  padding="same")(x)

    return x

def unet_up_layer(x):

    x = UpSampling2D(size=2)(x)

    return x

def unet_downconv_layer(x, filters, kernel_size, kernel_initializer, activation, batch_norm):

    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               kernel_initializer=kernel_initializer,
               strides=2,
               padding="same")(x)

    x = Activation(activation)(x)

    if batch_norm:

        x = BatchNormalization()(x)

    return x

def unet_transconv_layer(x, filters, kernel_size, kernel_initializer, activation, batch_norm):

    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer,
                        strides=2,
                        padding = "same")(x)
    
    x = Activation(activation)(x)

    if batch_norm:

        x = BatchNormalization()(x)

    return x

def unet_final_layer(x, final_filters, final_activation, kernel_size, kernel_initializer):

    x = Conv2D(filters=final_filters,
               kernel_size=1,
               kernel_initializer=kernel_initializer,
               activation=final_activation)(x)
    
    return x


# Unet Blocks.

def unet_conv_block(x, filters, kernel_size, kernel_initializer, activation, batch_norm):
    
    conv1 = unet_conv_layer(x, filters, kernel_size, kernel_initializer, activation, batch_norm)
    conv2 = unet_conv_layer(conv1, filters, kernel_size, kernel_initializer, activation, batch_norm)

    return conv2, filters

def unet_convpool_block(x, dropout, filters, kernel_size, kernel_initializer, activation, batch_norm):

    conv1 = unet_conv_layer(x, filters, kernel_size, kernel_initializer, activation, batch_norm)
    conv2 = unet_conv_layer(conv1, filters, kernel_size, kernel_initializer, activation, batch_norm)
    pool = unet_max_layer(conv2)    
    drop = Dropout(dropout)(pool)
  
    return drop, conv2, filters

def unet_downconv_block(x, dropout, filters, kernel_size, kernel_initializer, activation, batch_norm):
    
    conv1 = unet_conv_layer(x, filters, kernel_size, kernel_initializer, activation, batch_norm)
    conv2 = unet_conv_layer(conv1, filters, kernel_size, kernel_initializer, activation, batch_norm)
    downconv = unet_downconv_layer(conv2, filters, kernel_size, kernel_initializer, activation, batch_norm)    
    drop = Dropout(dropout)(downconv)
  
    return drop, conv2, filters

def unet_upconv_block(x, dropout, filters, kernel_size, kernel_initializer, activation, batch_norm):

    up = unet_up_layer(x)
    kernel_size = 2
    conv = unet_conv_layer(up, filters, kernel_size, kernel_initializer, activation, batch_norm)

    return conv

def unet_transconv_block(x, dropout, filters, kernel_size, kernel_initializer, activation, batch_norm):

    transconv = unet_transconv_layer(x, filters, kernel_size, kernel_initializer, activation, batch_norm)
    kernel_size = 2
    conv = unet_conv_layer(transconv, filters, kernel_size, kernel_initializer, activation, batch_norm)  

    return conv

def unet_concat_block(x1, x2, dropout, filters, kernel_size, kernel_initializer, activation, batch_norm):

    concat = Concatenate(axis=3)([x1,x2])
    conv1 = unet_conv_layer(concat, filters, kernel_size, kernel_initializer, activation, batch_norm)
    conv2 = unet_conv_layer(conv1, filters, kernel_size, kernel_initializer, activation, batch_norm)
    drop = Dropout(dropout)(conv2)

    return drop, filters


# Generic Unet model.

def unet_model_standard(input_size, 
                        scale, 
                        filters,
                        kernel_size, 
                        kernel_initializer, 
                        activation, 
                        final_filters,
                        final_activation,
                        dropout, 
                        batch_norm,
                        use_input):

    params = {"kernel_size": kernel_size,
              "kernel_initializer": kernel_initializer,
              "activation": activation,
              "batch_norm": batch_norm}

    scaled_input = (int(input_size[0] * scale), int(input_size[1] * scale), input_size[2])       
    
    unet_input = Input(shape=(scaled_input))
    
    drop1, conv1, filters = unet_convpool_block(x=unet_input, filters=filters*2, dropout=dropout*0.5, **params)
    
    drop2, conv2, filters = unet_convpool_block(x=drop1, filters=filters*2, dropout=dropout, **params)
    
    drop3, conv3, filters = unet_convpool_block(x=drop2, filters=filters*2, dropout=dropout, **params)
    
    conv4, filters = unet_conv_block(x=drop3, filters=filters*2, **params)
    
    upconv5 = unet_upconv_block(x=conv4, filters=filters/2, dropout=dropout, **params)
    concat5, filters = unet_concat_block(x1=conv3, x2=upconv5, filters=filters/2, dropout=dropout, **params)
    
    upconv6 = unet_upconv_block(x=concat5, filters=filters/2, dropout=dropout, **params)
    concat6, filters = unet_concat_block(x1=conv2, x2=upconv6, filters=filters/2, dropout=dropout, **params)
    
    upconv7 = unet_upconv_block(x=concat6, filters=filters/2, dropout=dropout, **params)
    concat7, filters = unet_concat_block(x1=conv1, x2=upconv7, filters=filters/2, dropout=dropout, **params)
    
    unet_output = unet_final_layer(concat7, final_filters, final_activation, kernel_size, kernel_initializer)

    if use_input:

        unet_output = Concatenate(axis=3)([unet_input, unet_output])

    model = Model(inputs=unet_input, outputs=unet_output)

    return model

def unet_model_convolutional(input_size, 
                             scale, 
                             filters,
                             kernel_size, 
                             kernel_initializer, 
                             activation, 
                             final_filters,
                             final_activation,
                             dropout, 
                             batch_norm,
                             use_input):

    params = {"kernel_size": kernel_size,
              "kernel_initializer": kernel_initializer,
              "activation": activation,
              "batch_norm": batch_norm}

    scaled_input = (int(input_size[0] * scale), int(input_size[1] * scale), input_size[2])       
    
    unet_input = Input(shape=(scaled_input))
    
    drop1, conv1, filters = unet_downconv_block(x=unet_input, filters=filters*2, dropout=dropout*0.5, **params)
    
    drop2, conv2, filters = unet_downconv_block(x=drop1, filters=filters*2, dropout=dropout, **params)
    
    drop3, conv3, filters = unet_downconv_block(x=drop2, filters=filters*2, dropout=dropout, **params)
    
    conv4, filters = unet_conv_block(x=drop3, filters=filters*2, **params)
    
    transconv5 = unet_transconv_block(x=conv4, filters=filters/2, dropout=dropout, **params)
    concat5, filters = unet_concat_block(x1=conv3, x2=transconv5, filters=filters/2, dropout=dropout, **params)
    
    transconv6 = unet_transconv_block(x=concat5, filters=filters/2, dropout=dropout, **params)
    concat6, filters = unet_concat_block(x1=conv2, x2=transconv6, filters=filters/2, dropout=dropout, **params)
    
    transconv7 = unet_transconv_block(x=concat6, filters=filters/2, dropout=dropout, **params)
    concat7, filters = unet_concat_block(x1=conv1, x2=transconv7, filters=filters/2, dropout=dropout, **params)
    
    unet_output = unet_final_layer(concat7, final_filters, final_activation, kernel_size, kernel_initializer)

    if use_input:

        unet_output = Concatenate(axis=3)([unet_input, unet_output])

    model = Model(inputs=unet_input, outputs=unet_output)

    return model

def get_unet_segmentation():

    params = {'input_size': (288, 288, 3),
              'scale': 0.5,
              'filters': 8,
              'kernel_size': 3,
              'kernel_initializer': "he_normal",
              'activation': "relu",
              'final_filters': 1,
              'final_activation': "sigmoid",
              'use_input': False,
              'dropout': 0.05,
              'batch_norm': True}

    model = unet_model_standard(**params)
    
    return model