# Copyright 2020 Samson Woof. All Rights Reserved.
# =============================================================================

"""Measurement tools for Segmentation.
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings(
    "once",
    category=PendingDeprecationWarning)


def get_jaccard(ground_truth, prediction, class_names, classifi_mode='one'):
    """Get Jaccard index table.

    Args:
        ground_truth: ndarray.
        prediction: ndarray.
        class_names: A list of string,   
            corresponding name of label.
        classifi_mode: A string,
            one of 'one'、'binary'、'multi'.          
            If the label encode is one-hot,
            please specify as 'one'.

    Return:
        A pandas.Series.
    """
    ground_truth_area = np.sum(ground_truth, axis=(0, 1, 2))
    prediction_area = np.sum(prediction, axis=(0, 1, 2))
    intersection_area = np.sum(ground_truth*prediction, axis=(0, 1, 2))
    union_area = (ground_truth_area
                  + prediction_area
                  - intersection_area)
    jaccard_list = intersection_area/union_area

    if classifi_mode == "one":
        jaccard_list = jaccard_list[:-1]

    jaccard_series = pd.Series(jaccard_list, class_names)

    return jaccard_series


def get_dice(ground_truth, prediction, class_names, classifi_mode='one'):
    """Get Dice coefficient table.

    Args:
        ground_truth: ndarray.
        prediction: ndarray.
        class_names: A list of string,   
            corresponding name of label.
        classifi_mode: A string,
            one of 'one'、'binary'、'multi'.          
            If the label encode is one-hot,
            please specify as 'one'.

    Return:
        A pandas.Series.
    """
    ground_truth_area = np.sum(ground_truth, axis=(0, 1, 2))
    prediction_area = np.sum(prediction, axis=(0, 1, 2))
    intersection_area = np.sum(ground_truth*prediction, axis=(0, 1, 2))
    combined_area = ground_truth_area + prediction_area
    dice_list = 2*intersection_area/combined_area

    if classifi_mode == "one":
        dice_list = dice_list[:-1]

    dice_series = pd.Series(dice_list, class_names)

    return dice_series


def get_iou(ground_truth, prediction, class_names, classifi_mode='one'):
    """Get IOU table.

    Args:
        ground_truth: ndarray.
        prediction: ndarray.
        class_names: A list of string,   
            corresponding name of label.
        classifi_mode: A string,
            one of 'one'、'binary'、'multi'.          
            If the label encode is one-hot,
            please specify as 'one'.

    Return:
        A pandas.Series.
    """
    warnings.warn(
    ("The function will be deprecated. "
     "Replace it with `get_jaccard()`.")
    , PendingDeprecationWarning)

    return get_jaccard(ground_truth,
                       prediction,
                       class_names,
                       classifi_mode=classifi_mode)


def create_confusion_mat(ground_truth,
                         prediction,
                         class_names,
                         groundtruth_name="groundtruth",
                         prediction_name="prediction",
                         nothing_name="nothing",
                         classifi_mode='one'):
    """Create a confusion matrix for multi-category segmentation.

    Args:
        ground_truth: A ndarray.
        prediction: A ndarray.
        class_names: A list of string,   
            corresponding names of classes.
        groundtruth_name: A string.
        prediction_name: A string.
        nothing_name: A string.
        classifi_mode: A string,
            one of 'one'、'binary'、'multi'.          
            If the label encode is one-hot,
            please specify as 'one'.

    Return:
        A pandas.Dataframe.
    """
    if classifi_mode=="one":
        ground_truth = ground_truth.argmax(axis=-1).flatten()
        prediction = prediction.argmax(axis=-1).flatten()
    else:
        class_num = len(class_names)
        ground_truth = np.where(
            (ground_truth >= 0.5).all(axis=-1),
            ground_truth.argmax(axis=-1),
            class_num
            ).flatten()
        prediction = np.where(
            (prediction >= 0.5).all(axis=-1),
            prediction.argmax(axis=-1),
            class_num
            ).flatten() 
    confus_m = pd.crosstab(
        ground_truth,
        prediction
        )
    class_names_arr = np.array(class_names + [nothing_name])
    confus_m.index = pd.Index(
        class_names_arr,
        name=groundtruth_name)
    confus_m.columns = pd.Index(
        class_names_arr[confus_m.columns],
        name=prediction_name)
    return confus_m


def create_score_mat(confusion_mat):
    """Create score matrix table.

    Args:
        confusion_mat: Pandas.Dataframe,
            you can get this from `create_confusion_mat()`.

    Return:
        A pandas.Dataframe.
    """
    out = pd.DataFrame(index=confusion_mat.index,
                       columns=['precision',
                                'recall',
                                'specificity',
                                'F1-score'])
    out.index.name = None
    ptotal = confusion_mat.sum(axis=0)
    gtotal = confusion_mat.sum(axis=1)
    index_range = range(len(confusion_mat.index))
    TP = [confusion_mat[confusion_mat.index[i]][confusion_mat.columns[i]]
          for i in index_range]
    N = gtotal.sum()
    negative = N - gtotal
    FP = ptotal - TP
    TN = negative - FP

    precision = TP/ptotal
    recall = TP/gtotal
    specificity = TN/negative
    F1_score = (2*precision*recall)/(precision + recall)

    out['precision'] = precision
    out['recall'] = recall
    out['specificity'] = specificity
    out['F1-score'] = F1_score

    return out