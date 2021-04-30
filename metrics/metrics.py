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


def dice_coef_func(smooth=1, threshold=0.5):
    def dice_coef(y_true, y_pred):
        prediction = tf.where(y_pred > threshold, 1, 0)
        prediction = tf.cast(prediction, dtype=y_true.dtype)
        ground_truth_area = tf.reduce_sum(
            y_true, axis=(1, 2, 3))
        prediction_area = tf.reduce_sum(
            prediction, axis=(1, 2, 3))
        intersection_area = tf.reduce_sum(
            y_true*y_pred, axis=(1, 2, 3))
        combined_area = ground_truth_area + prediction_area
        dice = tf.reduce_mean(
            (2*intersection_area + smooth)/(combined_area + smooth))
        return dice
    return dice_coef


def jaccard_index_func(smooth=1, threshold=0.5):
    def jaccard_index(y_true, y_pred):
        prediction = tf.where(y_pred > threshold, 1, 0)
        prediction = tf.cast(prediction, dtype=y_true.dtype)
        ground_truth_area = tf.reduce_sum(
            y_true, axis=(1, 2, 3))
        prediction_area = tf.reduce_sum(
            prediction, axis=(1, 2, 3))
        intersection_area = tf.reduce_sum(
            y_true*y_pred, axis=(1, 2, 3))
        union_area = (ground_truth_area
                      + prediction_area
                      - intersection_area)
        jaccard = tf.reduce_mean(
            (intersection_area + smooth)/(union_area + smooth))
        return jaccard
    return jaccard_index