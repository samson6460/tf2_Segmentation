# Copyright 2020 Samson Woof. All Rights Reserved.
# =============================================================================

"""Data processing tools for Segmentation.
"""

import json
from io import BytesIO
from PIL import Image
import base64
import numpy as np
import cv2
import os
import threading
import imgaug.augmenters as iaa
import threading
from math import ceil
from matplotlib.colors import to_rgb


def _process_img(img, size):
    size = size[1], size[0]
    zoom_r = np.array(img.size)/np.array(size)
    img = img.resize(size)
    img = img.convert("RGB")
    img = np.array(img)
    return img, zoom_r


class Segdata_reader:
    """Read the images and annotations for segmentation.

    Args:
        rescale: A float or None,
            specifying how the image value should be scaled.
            If None, no scaled.
        augmenter: A `imgaug.augmenters.meta.Sequential` instance.
        aug_times: An integer,
            The default is 1, which means no augmentation.

    Attributes:
        rescale
        augmenter
        aug_times
        file_names: A list of string
            with all file names that have been read.

    Return:
        A reader instance for images and annotations.

    """
    def __init__(self, rescale=None, augmenter=None, aug_times=1):
        self.rescale = rescale
        self.augmenter = augmenter
        self.aug_times = aug_times
        self.file_names = None

        if augmenter is None or type(augmenter) is not iaa.meta.Sequential:
            self.aug_times = 1

    def labelme_json_to_dataset(
        self, img_path=None, label_path=None,
        class_names=[], size=(512, 512),
        padding=True, line_thickness=5,
        shuffle=True, seed=None,
        classifi_mode="one",
        encoding="big5",
        thread_num=10):
        """Convert the JSON file generated by `labelme` into ndarray.

        Args:
            img_path: A string, 
                file path of images.
                if JSON files include images,
                just specify one of args(img_path、label_path)
                with JSON files path.
            label_path: A string,
                file path of JSON files.
            class_names: A list of string,
                the class names of the category in `labelme`.
                For example: ["g", "t", "v", "bg"],
                    the format of the returned label is one-hot,
                    and the channel is the class.
                    For example, the channel is [1, 0, 0, 0, 0],
                    which means the pixel is "g";
                    [0, 0, 0, 0, 1] means Nothing.
                The following format is also acceptable:
                ["g", ("t", "v"), "bg"],
                which means to treat "t" and "v" as a group.
            size: A tuple of 2 integers: (heights, widths),
                images will be resized to this arg.
            padding: A boolean,
                whether to fill the mark, default: True.
            line_thickness: A integer,
                the width of the drawn line, default: 5.
            shuffle: Boolean, default: True.
            seed: An integer, random seed, default: None.
            classifi_mode: A string,
                one of 'one'、'binary'、'multi',
                which means one-hot encode、binary encode
                and multi-hot encode respectively.
            encoding: A string,
                encoding format of JSON file,
                default: "big5".
            thread_num: An integer,
                specifying the number of threads to read files.

        Return:
            A tuple of Numpy arrays: (train data, label data)
            train data:
                shape (batches, img heights, img widths, color channels).
            label data:
                shape (batches, mask heights, mask widths, classes).
        """
        return self._file_to_array(
            img_path=img_path, label_path=label_path,
            class_names=class_names, size=size,
            padding=padding, line_thickness=line_thickness,
            shuffle=shuffle, seed=seed,
            classifi_mode=classifi_mode,
            encoding=encoding,
            thread_num=thread_num, format="json")

    def imglayer_to_dataset(
        self, img_path=None, label_path=None,
        class_colors=["r", "b"],
        size=(512, 512), shuffle=True, seed=None,
        classifi_mode="one",
        thread_num=10):
        """Convert the images and image layers into ndarray.

        Args:
            img_path: A string, 
                file path of images.
                Or specify one of args(img_path、label_path)
                with a folder path includes images folder(should name as img)
                and masks folder(should name as label).
            label_path: A string,
                file path of segmentation masks.
            class_colors: A list of color string or RGB tuple of integer,
                Example of color string:
                    ['r', 'b'](Default).
                    check for more info about color string
                    by the following url:
                    https://matplotlib.org/tutorials/colors/colors.html
                Example of RGB tuple of integer:
                    [(255, 0, 0), (0, 0, 255)](which means Red、Blue).
            size: A tuple of integer: (heights, widths),
                images will be resized to this arg.
            shuffle: A boolean, default: True.
            seed: An integer, random seed, default: None.
            classifi_mode: A string,
                one of 'one'、'binary'、'multi',
                which means one-hot encode、binary encode
                and multi-hot encode respectively.
            thread_num: An integer,
                specifying the number of threads to read files.

        Return:
            A tuple of Numpy arrays: (train data, label data)
            train data:
                shape (batches, img heights, img widths, color channels).
            label data:
                shape (batches, mask heights, mask widths, classes).
        """
        return self._file_to_array(
            img_path=img_path, label_path=label_path,
            class_names=class_colors, size=size,
            padding=None, line_thickness=None,
            shuffle=shuffle, seed=seed,
            classifi_mode=classifi_mode,
            encoding=None,
            thread_num=thread_num, format="layer")  

    def _str_to_rgb(self, c):
        if type(c)==str:
            return tuple(map(lambda x:int(round(x*255)), to_rgb(c)))
        else:
            return c

    def _process_paths(self, path_list):
        path_list = np.array(path_list)
        U_num = path_list.dtype.itemsize//4 + 5
        dtype = "<U" + str(U_num)
        filepaths = np.empty((len(path_list),
                             self.aug_times),
                             dtype = dtype)
        filepaths[:, 0] = path_list
        filepaths[:, 1:] = np.char.add(filepaths[:, 0:1], "(aug)")
        path_list = filepaths.flatten()
        return path_list

    def _file_to_array(self, img_path, label_path,
                       class_names,
                       size, padding, line_thickness,
                       shuffle, seed,
                       classifi_mode,
                       encoding,
                       thread_num, format):
        def _read_json(_path_list, _pos):
            for path_i, name in enumerate(_path_list):
                pos = (_pos + path_i)*self.aug_times

                with open(os.path.join(
                        label_path,
                        name[:name.rfind(".")] + ".json"),
                        encoding=encoding) as f:
                    jdata = f.read()
                    data = json.loads(jdata)

                if img_path is None:
                    img64 = data['imageData']
                    img = Image.open(BytesIO(base64.b64decode(img64)))
                else:
                    img = Image.open(os.path.join(img_path, name))

                img, zoom_r = _process_img(img, size)

                if classifi_mode == "one":
                    y = np.zeros((*size, len(class_names) + 1),
                                 dtype="int8")
                    y[:, :, -1] = 1
                else:
                    y = np.zeros((*size, len(class_names)),
                                 dtype="int8")
            
                for data_i in range(len(data["shapes"])):
                    label = data["shapes"][data_i]["label"]

                    if label in class_names:
                        index = class_names.index(label)
                    else:
                        for key_i, key in enumerate(class_names):
                            if isinstance(key, tuple):
                                if label in key:
                                    index = key_i
                                    break
                        else:
                            index = -1

                    if index >= 0:
                        point = np.array(data["shapes"][data_i]["points"])
                        point = (point/zoom_r).astype(int)

                        label_im = np.zeros((*size, 1))
                        shape_type = data["shapes"][data_i]["shape_type"]
                        if shape_type == "linestrip":
                            cv2.polylines(label_im,
                                          [point], False,
                                          (1, 0, 0), line_thickness)
                        elif padding:
                            cv2.drawContours(label_im,
                                             [point],
                                             -1, (1, 0, 0), -1)
                        else:
                            cv2.polylines(label_im,
                                          [point], True,
                                          (1, 0, 0), line_thickness)

                        label_shape = label_im[..., 0:1].astype(bool)
                        y[..., index:index + 1][label_shape] = 1   
                        if classifi_mode == "one":
                            y[..., -1:][label_shape] = 0
                
                label_data[pos] = y

                if self.augmenter is not None:
                    y = np.expand_dims(y, axis=0)
                    for aug_i in range(1, self.aug_times):
                        img_aug_i, label_aug_i = self.augmenter(
                            image=img,
                            segmentation_maps=y)
                        if self.rescale is not None:
                            img_aug_i = img_aug_i*self.rescale
                        train_data[pos + aug_i] = img_aug_i
                        label_data[pos + aug_i] = label_aug_i[0]

                if self.rescale is not None:
                    img = img*self.rescale
                train_data[pos] = img

        def _read_layer(_path_list, _pos):
            for i,  name in enumerate(_path_list):
                pos = (_pos + i)*self.aug_times

                try:
                    label = Image.open(os.path.join(
                        label_path,
                        name[:name.rfind(".")] + ".png"))
                except:
                    label = Image.open(os.path.join(
                        label_path,
                        name[:name.rfind(".")] + ".jpg"))  

                label, _ = _process_img(label, size)

                img = Image.open(os.path.join(img_path, name))
                img, _ = _process_img(img, size)

                for color_i, color in enumerate(class_colors):
                    mask = (label == color).all(axis=-1)
                    if classifi_mode == "one":
                        label_data[pos][mask, -1] = 0
                    label_data[pos][mask, color_i] = 1
                label = label_data[pos:pos+1].astype("int8")
                
                if self.augmenter is not None:
                    for aug_i in range(1, self.aug_times):
                        img_aug_i, label_aug_i = self.augmenter(
                            image=img,
                            segmentation_maps=label)
                        if self.rescale:
                            img_aug_i = img_aug_i*self.rescale
                        train_data[pos + aug_i] = img_aug_i
                        label_data[pos + aug_i] = label_aug_i[0]
                if self.rescale is not None:
                    img = img*self.rescale
                train_data[pos] = img 

        if label_path is None:
            img_path, label_path = None, img_path

        if format == "json":
            if img_path is None:
                file_path = label_path
            else:
                file_path = img_path
            thread_func = _read_json         
        elif format == "layer":
            if img_path is None:
                img_path = label_path + os.sep + "img"
                label_path = label_path + os.sep + "label"
            file_path = img_path
            thread_func = _read_layer
            class_colors = list(map(self._str_to_rgb, class_names))
        
        path_list = os.listdir(file_path)
        path_list = [f for f in path_list if not f.startswith(".")]
        len_path_list = len(path_list)

        train_data = np.empty((len_path_list*self.aug_times, *size, 3))
        if classifi_mode == "one":
            label_data = np.zeros((len_path_list*self.aug_times,
                                  *size, len(class_names) + 1))
            label_data[..., -1] = 1
        else:
            label_data = np.zeros((len_path_list*self.aug_times,
                                   *size, len(class_names)))

        threads = []
        workers = ceil(len_path_list/thread_num)

        for worker_i in range(0, len_path_list, workers):
            threads.append(
                threading.Thread(target=thread_func,
                args=(path_list[worker_i : worker_i+workers],
                      worker_i)))
        for thread in threads:
            thread.start()                
        for thread in threads:
            thread.join()

        path_list = self._process_paths(path_list)

        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            shuffle_index = np.arange(len(train_data))
            np.random.shuffle(shuffle_index)
            train_data = train_data[shuffle_index]
            label_data = label_data[shuffle_index]
            path_list = path_list[shuffle_index]
            
        path_list = path_list.tolist()
        self.file_names = path_list

        return train_data, label_data


def read_img(path, size=(512, 512), rescale=None):
    """Read images as ndarray.

    Args:
        size: A tuple of 2 integers,
            (heights, widths).
        rescale: A float or None,
            specifying how the image value should be scaled.
            If None, no scaled.
    """
    img_list = [f for f in os.listdir(path) if not f.startswith(".")]
    data = np.empty((len(img_list), *size, 3))
    size = size[1], size[0]

    for img_i, _path in enumerate(img_list):
        img = Image.open(path + os.sep + _path)
        img = img.resize(size)
        img = img.convert("RGB")  
        img = np.array(img)
        if rescale:
            img = img*rescale
        data[img_i] = img
        
    return data