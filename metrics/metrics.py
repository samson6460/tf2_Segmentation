# Copyright 2021 Samson Woof. All Rights Reserved.
# =============================================================================

"""Metrics for Segmentation.
"""

import tensorflow as tf
from tensorflow.keras.metrics import binary_accuracy


def be_binary_accuracy(y_true, y_pred):
    class_nums = y_pred.shape[-1]//2

    y_true = y_true[..., class_nums:]
    y_pred = y_pred[..., class_nums:]
    bi_acc = binary_accuracy(y_true, y_pred)

    return bi_acc


def dice_coef_func(smooth=1):
    def dice_coef(y_true, y_pred):
        ground_truth_area = tf.reduce_sum(
            y_true, axis=(1, 2, 3))
        prediction_area = tf.reduce_sum(
            y_pred, axis=(1, 2, 3))
        intersection_area = tf.reduce_sum(
            y_true*y_pred, axis=(1, 2, 3))
        combined_area = ground_truth_area + prediction_area
        dice = tf.reduce_mean(
            (2*intersection_area + smooth)/(combined_area + smooth))
        return dice
    return dice_coef