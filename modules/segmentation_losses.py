import tensorflow as tf 
import numpy as np 

# Dice Loss + Cross Entropy Loss.
def entropy_dice():

    def dice_loss(y_true, y_pred):

        numerator = 2 * tf.math.reduce_sum(y_true * y_pred, axis=(1,2,3))
        denominator = tf.math.reduce_sum(y_true + y_pred, axis=(1,2,3))

        return tf.reshape(1 - numerator / denominator, (-1,1,1))
    
    def entropy_dice_loss(y_true, y_pred):

        return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

    return entropy_dice_loss

# Dice Loss (F1 Score).
def dice():

    def dice_loss(y_true, y_pred):

        numerator = 2 * tf.math.reduce_sum(y_true * y_pred, axis=(1,2,3))
        denominator = tf.math.reduce_sum(y_true + y_pred, axis=(1,2,3))

        return 1 - numerator / denominator

    def dice_coefficient(y_true, y_pred):

        return 1 - dice_loss(y_true, y_pred)
    
    return dice_loss, dice_coefficient

# Jaccard Loss (IoU).
def jaccard():

    def dice_loss(y_true, y_pred):

        numerator = 2 * tf.math.reduce_sum(y_true * y_pred, axis=(1,2,3))
        denominator = tf.math.reduce_sum(y_true + y_pred, axis=(1,2,3))

        return 1 - numerator / denominator

    def dice_coefficient(y_true, y_pred):

        return 1 - dice_loss(y_true, y_pred)

    def jaccard_loss(y_true, y_pred):

        return 1 - (dice_coefficient(y_true, y_pred)/(2 - dice_coefficient(y_true, y_pred)))

    def jaccard_coefficient(y_true, y_pred):

        return 1 - jaccard_loss(y_true, y_pred)

    return jaccard_loss, jaccard_coefficient

# Intensity Loss (L1 and L2 combined implementation).
def intensity(l_num=1):

    """
    Computes L1 and L2 losses (MAE and MSE)
    Adapted from https://github.com/dyelax/Adversarial_Video_Generation/blob/master/Code/loss_functions.py
    """
    def intensity_loss(y_true, y_pred):

        return tf.math.reduce_mean(tf.math.abs(y_true - y_pred)**l_num)

    return intensity_loss