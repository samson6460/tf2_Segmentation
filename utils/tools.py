# Copyright 2020 Samson Woof. All Rights Reserved.
# =============================================================================

"""Presentation tools for Segmentation.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import cv2


def vis_img_mask(img, label,
                 color=['r', 'lime', 'b', 'c', 'm', 'y', 'pink', 'w'],
                 label_alpha=0.3,
                 classifi_mode='one',
                 return_array=False):
    """Visualize images and annotations.

    Args:
        img: A ndarry of shape(img heights, img widths, color channels).
        label: A ndarray of shape(mask heights, mask widths, classes).
        color: A list of color string or RGB tuple of float.
            Example of color string:
                ['r', 'lime', 'b', 'c', 'm', 'y', 'pink', 'w'](Default).
                check for more info about color string by the following url:
                https://matplotlib.org/tutorials/colors/colors.html
            Example of RGB tuple of float:
                [(1, 0, 0), (0, 0, 1)](which means Red、Blue).
        label_alpha: A float,
            transparency of annotation mask.
        classifi_mode: A string,
            one of 'one'、'binary'、'multi'.          
            If the label encode is one-hot,
            please specify as 'one'.
        return_array: A boolean,
            Default is False.
    """
    color = list(map(to_rgb, color))
    nimg = np.array(img)
    if nimg.shape != label.shape:
        label = cv2.resize(label, (nimg.shape[1], nimg.shape[0]))
    
    if classifi_mode == "one":
        class_num = label.shape[-1] - 1
    else:
        class_num = label.shape[-1]
    for i in range(class_num):
        is_the_class = label[:, :, i:i+1] == 1
        nimg_R = nimg[:, :, 0:1]
        nimg_G = nimg[:, :, 1:2]
        nimg_B = nimg[:, :, 2:3]
        nimg_R[is_the_class] = ((1 - label_alpha)*nimg_R[is_the_class]
                                + label_alpha*color[i][0])
        nimg_G[is_the_class] = ((1 - label_alpha)*nimg_G[is_the_class]
                                + label_alpha*color[i][1])
        nimg_B[is_the_class] = ((1 - label_alpha)*nimg_B[is_the_class]
                                + label_alpha*color[i][2])
    plt.imshow(nimg)
    plt.show()
    if return_array:
        return nimg


def plot_history(history_dict, keys,
                 title=None, xyLabel=[],
                 ylim=(), size=()):
    """Draw history line graph.
    
    The historical records such as the loss value
    or accuracy rate returned during training
    can be drawn into a line graph.

    Args:
        history: A dictionary,
            containing one or more data to be drawn,
            for example: {'loss': [4,2,1,…], 'acc': [2,3,5,…]}.
        keys: A tuple or list,
            specifying the key value to be drawn in history,
            for example: ('loss', 'acc').
        title: A string,
            specifying the title text of the chart.
        xyLabel: A tuple or list,
            specifying the description text of x, y axis,
            for example:('epoch', 'Accuracy').
        ylim: A tuple or list,
            specifying the minimum and maximum values ​​of the y-axis,
            for example: (1, 3),
            values ​​outside the range will be ignored.
        size: A tuple,
            specifying the size of the picture,
            the default is (6, 4)
            (representing width 6 and height 4 inches)
    """
    lineType = ('-', '--', '.', ':')
    if len(ylim) == 2: plt.ylim(*ylim)
    if len(size) == 2: plt.gcf().set_size_inches(*size)
    epochs = range(1, len(history_dict[keys[0]]) + 1)
    for i in range(len(keys)):
        plt.plot(epochs, history_dict[keys[i]], lineType[i])
    if title:
        plt.title(title)
    if len(xyLabel)==2:
        plt.xlabel(xyLabel[0])
        plt.ylabel(xyLabel[1])
    plt.legend(keys, loc='best')
    plt.show()


def get_class_weight(label_data, method="alpha"):
    """Get the weight of the category.

    Args:
        label_data: A ndarray of shape(batch_size, grid_num, grid_num, info).
        method: A string,
            one of "alpha"、"log"、"effective"、"binary".

    Returns:
        A list containing the weight of each category.
    """
    class_weight = []
    if method != "alpha":
        total = 1
        for i in label_data.shape[:-1]:
            total *= i
        if method == "effective":
            beta = (total - 1)/total
    for i in range(label_data.shape[-1]):
        samples_per_class = label_data[:,:,:,i].sum()
        if method == "effective":
            effective_num = 1 - np.power(beta, samples_per_class)
            class_weight.append((1 - beta)/effective_num)
        elif method == "binary":
            class_weight.append(samples_per_class/(total - samples_per_class))
        else:
            class_weight.append(1/samples_per_class)
    class_weight = np.array(class_weight)
    if method == "log":
        class_weight = np.log(total*class_weight)
 
    if method != "binary":
        class_weight = class_weight/np.sum(class_weight)*len(class_weight)

    return class_weight