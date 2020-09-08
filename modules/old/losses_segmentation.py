import tensorflow as tf

# Weighted Cross Entropy (WCE).
def weighted_crossentropy(beta=1):

    def convert_to_logits(y_pred):

        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        return tf.math.log(y_pred / (1 - y_pred))
    
    def weighted_crossentropy_loss(y_true, y_pred):

        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred,
                                                        labels=y_true,
                                                        pos_weight=beta)
        
        return tf.math.reduce_mean(loss)

    return weighted_crossentropy_loss

# Balanced Cross Entropy (BCE).
def balanced_crossentropy(beta=0.5):

    def convert_to_logits(y_pred):

        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        return tf.math.log(y_pred / (1 - y_pred))

    def balanced_crossentropy_loss(y_true, y_pred):

        y_pred = convert_to_logits(y_pred)
        pos_weight = beta / (1 - beta)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, 
                                                        labels=y_true, 
                                                        pos_weight=pos_weight)
        
        return tf.math.reduce_mean(loss * (1 - beta))
    
    return balanced_crossentropy_loss

# Focal Loss (FL).
def focal(alpha=0.5, gamma=0):

    def convert_to_logits(y_pred):

        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        return tf.math.log(y_pred / (1 - y_pred))

    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
    
        return tf.math.log1p(tf.math.exp(-tf.math.abs(logits))) + tf.nn.relu(-logits) * (weight_a + weight_b) + logits * weight_b

    def focal_loss(y_true, y_pred):

        logits = convert_to_logits(y_pred)
        loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

        return tf.math.reduce_mean(loss)
    
    return focal_loss

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

# Tversky Loss.
def tversky(beta=0.5):
    
    def tversky_loss(y_true, y_pred):

        numerator = tf.math.reduce_sum(y_true * y_pred, axis=-1)
        denominator = tf.math.reduce_sum(y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred), axis=-1)
    
        return 1 - (numerator + 1) / (denominator + 1)
    
    return tversky_loss

# Cross Entropy + Dice Loss
def entropy_dice(y_true, y_pred):

    def dice_loss(y_true, y_pred):

        numerator = 2 * tf.math.reduce_sum(y_true * y_pred, axis=(1,2,3))
        denominator = tf.math.reduce_sum(y_true + y_pred, axis=(1,2,3))

        return tf.reshape(1 - numerator / denominator, (-1,1,1))

    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

# Focal Loss + Dice Loss
def focal_dice(alpha=0.5, gamma=0):

    def dice_loss(y_true, y_pred):

        numerator = 2 * tf.math.reduce_sum(y_true * y_pred, axis=(1,2,3))
        denominator = tf.math.reduce_sum(y_true + y_pred, axis=(1,2,3))

        return 1 - numerator / denominator
    
    def focal(alpha=0.5, gamma=0):

        def convert_to_logits(y_pred):

            y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

            return tf.math.log(y_pred / (1 - y_pred))

        def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        
            weight_a = alpha * (1 - y_pred) ** gamma * targets
            weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
    
            return tf.math.log1p(tf.math.exp(-tf.math.abs(logits))) + tf.nn.relu(-logits) * (weight_a + weight_b) + logits * weight_b

        def focal_loss(y_true, y_pred):

            logits = convert_to_logits(y_pred)
            loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

            return tf.math.reduce_mean(loss)
    
        return focal_loss
    
    def focal_dice_loss(y_true, y_pred):

        focal_loss = focal(alpha, gamma)

        return focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)
    
    return focal_dice_loss

        