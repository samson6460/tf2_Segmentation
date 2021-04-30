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

        bece_fg = tf.reduce_sum(
            (1 + b)*G*tf.math.log(p_m)
            *bece_class_weight[:-1], axis=-1)
        bece_bg = G_bg*tf.math.log(p_m_bg)*bece_class_weight[-1]
        bece = -tf.reduce_mean(bece_fg + bece_bg)

        return ce + bece
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


def _dice_coef_func(y_true, y_pred,
                    smooth=1, class_weight=1,
                    beta=1):
    """
    tp: True positive
    fp: False positive
    fn: False negative
    """
    tp = tf.reduce_sum(
        y_true*y_pred, axis=(0, 1, 2))
    fp = tf.reduce_sum(
        y_pred, axis=(0, 1, 2)) - tp
    fn = tf.reduce_sum(
        y_true, axis=(0, 1, 2)) - tp

    dice = (((1 + beta**2)*tp + smooth)
            /((1 + beta**2)*tp + beta**2*fn + fp + smooth))
    dice = tf.reduce_mean(dice*class_weight)
    return dice


def dice_loss_func(smooth=1, class_weight=1, beta=1):
    def dice_loss(y_true, y_pred):
        dice_coef = _dice_coef_func(
            y_true, y_pred,
            smooth=smooth,
            class_weight=class_weight,
            beta=beta)
        dice_loss = 1 - dice_coef
        return dice_loss
    return dice_loss


def binary_dice_loss_func(smooth=1, binary_weight=1, beta=1):
    def binary_dice_loss(y_true, y_pred):
        pos_dice_coef = _dice_coef_func(
            y_true, y_pred,
            smooth=smooth,
            beta=beta)
        neg_dice_coef = _dice_coef_func(
            1 - y_true, 1 - y_pred,
            smooth=smooth,
            beta=beta)
        pos_dice_loss = 1 - pos_dice_coef
        neg_dice_loss = 1 - neg_dice_coef
        binary_dice = (pos_dice_loss + neg_dice_loss*binary_weight)/2
        return binary_dice
    return binary_dice_loss