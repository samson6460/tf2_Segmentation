# Copyright 2020 Samson Woof. All Rights Reserved.
# =============================================================================

"""Loss functions for Segmentation.
"""

import tensorflow as tf

epsilon = 1e-07


def balanced_categorical_crossentropy(class_weight=1):
    def _balanced_categorical_crossentropy(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        ce = -tf.reduce_sum(
            (y_true*tf.math.log(y_pred)*class_weight), axis=-1)
        return ce
    return _balanced_categorical_crossentropy


def balanced_binary_crossentropy(class_weight=1, binary_weight=1):
    def _balanced_binary_crossentropy(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        bce = -tf.reduce_mean(
            (y_true*tf.math.log(y_pred)
            + binary_weight*(1 - y_true)
            *tf.math.log(1 - y_pred))*class_weight, axis=-1)
        return bce
    return _balanced_binary_crossentropy


def categorical_be_crossentropy(ce_class_weight=1,
                                bece_class_weight=1,
                                alpha=0.5, beta=0.1):
    def _categorical_be_crossentropy(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

        class_nums = y_pred.shape[-1]//2
        B = y_true[..., :class_nums]
        p_b = y_pred[..., :class_nums]

        G = y_true[..., class_nums:-1]
        p_m = y_pred[..., class_nums:-1]

        G_bg = y_true[..., -1]
        p_m_bg = y_pred[..., -1]

        ce = -tf.reduce_mean(
            tf.reduce_sum(
                B*tf.math.log(p_b)*ce_class_weight, axis=-1))

        b = alpha*tf.maximum(beta - p_b[..., :-1], 0)

        bece = -tf.reduce_mean(
            tf.reduce_sum(
                (1 + b)*G*tf.math.log(p_m)
                *bece_class_weight[:-1], axis=-1))
        bece_bg = -tf.reduce_mean(
            G_bg*tf.math.log(p_m_bg)
            *bece_class_weight[-1])

        return ce + bece + bece_bg
    return _categorical_be_crossentropy


def binary_be_crossentropy(ce_class_weight=1,
                           bece_class_weight=1,
                           ce_binary_weight=1,
                           bece_binary_weight=1,
                           alpha=0.5, beta=0.1):
    def _binary_be_crossentropy(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

        class_nums = y_pred.shape[-1]//2
        B = y_true[..., :class_nums]
        p_b = y_pred[..., :class_nums]
        G = y_true[..., class_nums:]
        p_m = y_pred[..., class_nums:]

        ce = -tf.reduce_mean(
            (B*tf.math.log(p_b)
            + ce_binary_weight
            *(1 - B)*tf.math.log(1 - p_b))*ce_class_weight)
        b = alpha*(tf.maximum(beta - p_b, 0))
        bece = -tf.reduce_mean(
            ((1 + b)*G*tf.math.log(p_m)
            + bece_binary_weight
            *(1 - G)*tf.math.log(1 - p_m))*bece_class_weight)
        return ce + bece
    return _binary_be_crossentropy


def _dice_coef_func(y_true, y_pred, smooth=1):
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


def dice_loss_func(smooth=1):
    def dice_loss(y_true, y_pred):
        dice_coef = _dice_coef_func(
            y_true, y_pred, smooth=smooth)
        dice_loss = 1 - dice_coef
        return dice_loss
    return dice_loss