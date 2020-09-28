import tensorflow as tf


def dice_loss(y_true, y_pred, smooth=1e-6):
    numerator = tf.reduce_sum(y_true * y_pred, axis=0)
    # numerator = tf.reduce_sum(weights * numerator)
    denominator = tf.reduce_sum(y_true + y_pred, axis=0)
    # denominator = tf.reduce_sum(weights * denominator)
    loss = 1.0 - 2.0 * (numerator + smooth) / (denominator + smooth)
    return loss



