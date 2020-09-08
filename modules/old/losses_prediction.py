import tensorflow as tf 
import numpy as np 

def flow(optflow_model):

    def flow_loss(y_true, y_pred):

        batch_size = y_true.shape[0]
        base_frame = y_true[:, :, :, :3]
        true_frame = y_true[:, :, :, -3:]
        pred_frame = y_pred[:, :, :, -3:]

        list_true = []
        list_pred = []

        for i in range(batch_size):

            img1 = base_frame[i, :, :, :]
            img2 = true_frame[i, :, :, :]
            img3 = pred_frame[i, :, :, :]

            list_true.append((img1, img2))
            list_pred.append((img1, img3))

        true_flow = optflow_model.predict_from_img_pairs(list_true, batch_size=batch_size, verbose=False)
        pred_flow = optflow_model.predict_from_img_pairs(list_pred, batch_size=batch_size, verbose=False)

        return tf.math.reduce_mean(tf.math.abs(true_flow - pred_flow))

    return flow_loss

# Segmentation model loss (CE+Dice loss implemented)
def segmentation(segmentation_model):

    def entropy_dice(y_true, y_pred):

        def dice_loss(y_true, y_pred):

            numerator = 2 * tf.math.reduce_sum(y_true * y_pred, axis=(1,2,3))
            denominator = tf.math.reduce_sum(y_true + y_pred, axis=(1,2,3))

            return tf.reshape(1 - numerator / denominator, (-1,1,1))

        return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

    def segmentation_loss(y_true, y_pred):

        return entropy_dice(segmentation_model(y_true), segmentation_model(y_pred))

    return segmentation_loss


def intensity(l_num=1):

    """
    Computes L1 and L2 losses (MAE and MSE)
    Adapted from https://github.com/dyelax/Adversarial_Video_Generation/blob/master/Code/loss_functions.py
    """
    def intensity_loss(y_true, y_pred):

        return tf.math.reduce_mean(tf.math.abs(y_true - y_pred)**l_num)

    return intensity_loss

def gradient_difference(alpha=1):

    """
    Computes the Gradient Difference Loss (GDL)
    Adapted from https://github.com/dyelax/Adversarial_Video_Generation/blob/master/Code/loss_functions.py
    """
    def gradient_loss(y_true, y_pred):

        pred_dx, pred_dy = tf.image.image_gradients(y_pred)
        true_dx, true_dy = tf.image.image_gradients(y_true)

        grad_diff_x = tf.math.abs(true_dx - pred_dx)
        grad_diff_y = tf.math.abs(true_dy - pred_dy)

        return tf.math.reduce_mean(grad_diff_x ** alpha + grad_diff_y ** alpha)

    return gradient_loss

def intensity_gradient(l_num=1, alpha=1):

    intensity_loss = intensity(l_num)
    gradient_loss = gradient_difference(alpha)

    def intensity_gradient_loss(y_true, y_pred):

        return (intensity_loss(y_true, y_pred) + gradient_loss(y_true, y_pred))
    
    return intensity_gradient_loss

# Combination Loss
def combination(segmentation_model_object, segmentation_model_occlu, l_num=1, alpha=1):

    intensity_loss = intensity(l_num)
    gradient_loss = gradient_difference(alpha)
    segmentation_loss_object = segmentation(segmentation_model_object)
    segmentation_loss_occlu = segmentation(segmentation_model_occlu)

    def combination_loss(y_true, y_pred):

        return (intensity_loss(y_true, y_pred) + gradient_loss(y_true, y_pred) + segmentation_loss_object(y_true, y_pred) + segmentation_loss_occlu(y_true, y_pred))
    
    return combination_loss

def log10(t):

    """
    Computes Log10
    """

    numerator = tf.math.log(t)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    
    return numerator / denominator

def psnr_error(y_true, y_pred):

    """
    Computes the PSNR error
    Adapted from https://github.com/dyelax/Adversarial_Video_Generation/blob/master/Code/utils.py
    """

    shape = tf.shape(y_pred)
    num_pixels = tf.dtypes.cast((shape[1] * shape[2] * shape[3]), dtype=tf.float32)
    square_diff = tf.math.square(y_true - y_pred)

    batch_errors = 10 * log10(1 / ((1 / num_pixels) * tf.math.reduce_sum(square_diff, [1, 2, 3])))
    
    return tf.math.reduce_mean(batch_errors)

def sharp_diff_error(y_true, y_pred):

    """
    Computes the Sharpness Difference error
    Adapted from https://github.com/dyelax/Adversarial_Video_Generation/blob/master/Code/utils.py
    """

    shape = tf.shape(y_pred)
    num_pixels = tf.dtypes.cast((shape[1] * shape[2] * shape[3]), dtype=tf.float32)

    pred_dx, pred_dy = tf.image.image_gradients(y_pred)
    true_dx, true_dy = tf.image.image_gradients(y_true)

    pred_grad_sum = pred_dx + pred_dy
    true_grad_sum = true_dx + true_dy

    grad_diff = tf.math.abs(true_grad_sum - pred_grad_sum)

    batch_errors = 10 * log10(1 / ((1 / num_pixels) * tf.math.reduce_sum(grad_diff, [1, 2, 3])))

    return tf.math.reduce_mean(batch_errors)
