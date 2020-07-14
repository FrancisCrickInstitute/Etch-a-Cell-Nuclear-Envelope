import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)


# Hausdorff distance loss functions based on:
# https://github.com/N0vel/weighted-hausdorff-distance-tensorflow-keras-loss
# https://github.com/vnbot2/object-locator


def cdist(a, b):
    """
        Caculate the euclid distance of every point in a to every point in b
        |a| = n, |b|=m
        the expected returned matrix => [mxn]
    """
    a = tf.cast(a, tf.float32)
    b = tf.cast(b, tf.float32)

    diff = tf.expand_dims(a, 1) - tf.expand_dims(b, 0)
    distance = tf.sqrt(tf.reduce_sum(diff ** 2, -1))
    return distance


def get_notnull_indices(tensor):

    #y_pred_flat = tf.reshape(y_pred, [-1])

    zero = K.constant(0, dtype=tf.float32)
    where = K.not_equal(tensor, zero)
    indices = tf.where(where)
    return indices


def hausdorff_loss(y_true, y_pred):
    print("tensor shapes: ", K.int_shape(y_true), K.int_shape(y_pred))

    y_true_flat = get_notnull_indices(y_true)
    y_pred_flat = get_notnull_indices(y_pred)

    print("index shapes: ", K.int_shape(y_true_flat), K.int_shape(y_pred_flat))

    d_matrix = cdist(y_true_flat, y_pred_flat)
    print("matrix shape: ", K.int_shape(d_matrix))

    mins_XA = K.min(d_matrix, axis=0)
    mins_XB = K.min(d_matrix, axis=1)

    avg_XA = K.mean(mins_XA)
    avg_XB = K.mean(mins_XB)

    avg_hausdorff = (avg_XA + avg_XB) / 2.
    print("result shape: ", K.int_shape(avg_hausdorff))
    print("result: ", avg_hausdorff)

    return avg_hausdorff
