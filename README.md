# tf2_Segmentation

It's a framework of image segmentation implemented by tensorflow 2.x.
There're U-Net、DeepLabV3、BESNet in this framework.

U-Net: Convolutional Networks for Biomedical Image Segmentation by Olaf Ronneberger, Philipp Fischer, Thomas Brox (https://arxiv.org/abs/1505.04597).

DeepLabV3: Rethinking Atrous Convolution for Semantic Image Segmentation by Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam (https://arxiv.org/abs/1706.05587).

BESNet: Boundary-Enhanced Segmentation of Cells in Histopathological Images by Hirohisa Oda, Holger R. Roth et al. (https://link.springer.com/chapter/10.1007/978-3-030-00934-2_26).

# Table of Contents

- [tf2_Segmentation](#tf2_segmentation)
- [Table of Contents](#table-of-contents)
- [Usage](#usage)
- [API reference](#api-reference)
  - [<font color=#0000FF>models</font>](#font-color0000ffmodelsfont)
    - [<font color=#FF00FF>model_predict</font> function](#font-colorff00ffmodel_predictfont-function)
    - [<font color=#FF00FF>unet</font> function](#font-colorff00ffunetfont-function)
    - [<font color=#FF00FF>deeplabv3</font> function](#font-colorff00ffdeeplabv3font-function)
    - [<font color=#FF00FF>besnet</font> function](#font-colorff00ffbesnetfont-function)
    - [<font color=#FF00FF>mbesnet</font> function](#font-colorff00ffmbesnetfont-function)
  - [<font color=#0000FF>losses</font>](#font-color0000fflossesfont)
    - [<font color=#FF00FF>balanced_categorical_crossentropy</font> function](#font-colorff00ffbalanced_categorical_crossentropyfont-function)
    - [<font color=#FF00FF>balanced_binary_crossentropy</font> function](#font-colorff00ffbalanced_binary_crossentropyfont-function)
    - [<font color=#FF00FF>categorical_be_crossentropy</font> function](#font-colorff00ffcategorical_be_crossentropyfont-function)
    - [<font color=#FF00FF>binary_be_crossentropy</font> function](#font-colorff00ffbinary_be_crossentropyfont-function)
    - [<font color=#FF00FF>dice_loss_func</font> function](#font-colorff00ffdice_loss_funcfont-function)
  - [<font color=#0000FF>metrics</font>](#font-color0000ffmetricsfont)
    - [<font color=#FF00FF>be_binary_accuracy</font> function](#font-colorff00ffbe_binary_accuracyfont-function)
    - [<font color=#FF00FF>dice_coef_func</font> function](#font-colorff00ffdice_coef_funcfont-function)
  - [<font color=#0000FF>utils</font>](#font-color0000ffutilsfont)
    - [The Segdata_reader class](#the-segdata_reader-class)
      - [<font color=#FF00FF>Segdata_reader</font> class](#font-colorff00ffsegdata_readerfont-class)
      - [<font color=#FF00FF>labelme_json_to_dataset</font> method](#font-colorff00fflabelme_json_to_datasetfont-method)
      - [<font color=#FF00FF>imglayer_to_dataset</font> method](#font-colorff00ffimglayer_to_datasetfont-method)
    - [<font color=#FF00FF>read_img</font> function](#font-colorff00ffread_imgfont-function)
    - [<font color=#FF00FF>vis_img_mask</font> function](#font-colorff00ffvis_img_maskfont-function)
    - [<font color=#FF00FF>plot_history</font> function](#font-colorff00ffplot_historyfont-function)
    - [<font color=#FF00FF>get_class_weight</font> function](#font-colorff00ffget_class_weightfont-function)
    - [<font color=#FF00FF>get_jaccard</font> function](#font-colorff00ffget_jaccardfont-function)
    - [<font color=#FF00FF>get_dice</font> function](#font-colorff00ffget_dicefont-function)
    - [<font color=#FF00FF>create_confusion_mat</font> function](#font-colorff00ffcreate_confusion_matfont-function)
    - [<font color=#FF00FF>create_score_mat</font> function](#font-colorff00ffcreate_score_matfont-function)

# Usage

1. Clone or download
    - Use the command bellow in terminal to git clone:    
    ```git clone https://github.com/samson6460/tf2_Segmentation.git```

    - Or just download whole files using the **[clone or download]** button in the upper right corner.
    
2. Install dependent packages: 
    ```pip install -r requirements.txt```

3. Import tf2_Segmentation:
   ```import tf2_Segmentation```


# API reference

## <font color=#0000FF>models</font>

### <font color=#FF00FF>model_predict</font> function
```
tf2_Segmentation.models.model_predict(
    model, intput_tensor,
    classifi_mode="one", **kargs)
```
A function like `model.predict_classes()`.

Call `model_predict()` to predict and convert the prediction from probabilities to one-hot encoding or binary encoding.

**Arguments**
- **model**: tf.kears model instance.
- **intput_tensor**: tf.keras tensor.
- **classifi_mode**: A string, Specifying the encoding method.
Default: "one", which means one-hot encoding.
    
**Returns**
Prediction of one-hot encoding or binary encoding.

---

### <font color=#FF00FF>unet</font> function
```
tf2_Segmentation.models.unet(
    pretrained_weights=None,
    input_shape=(512, 512, 3),
    activation='relu',
    categorical_num=4,
    classifi_mode='one')
```
Create U-Net network architecture.
    
**Arguments**
- **pretrained_weights**: A string, file path of pretrained model.
- **input_shape**: A tuple of 3 integers, shape of input image.
- **activation**: A string, activation function for convolutional layer.
- **categorical_num**: An integer, number of categories.
- **classifi_mode**: A string, one of 'one'、'binary'、'multi'.
If specified as 'one', it means that the activation function of the output layer is softmax, and the label  should be one-hot encoding.

**Returns**
A tf.keras Model.

---

### <font color=#FF00FF>deeplabv3</font> function
```
tf2_Segmentation.models.deeplabv3(
    pretrained_weights='pascal_voc',
    input_tensor=None,
    input_shape=(512, 512, 3),
    classes=None,
    categorical_num=4,
    backbone='xception',
    OS=16, alpha=1.,
    activation="softmax")
```
Instantiates the Deeplabv3+ architecture.

Optionally loads weights pre-trained on PASCAL VOC or Cityscapes.
This model is available for TensorFlow only.

**Arguments**
- **pretrained_weights**: one of 'pascal_voc' (pre-trained on pascal voc), 'cityscapes' (pre-trained on cityscape) or local file path of pretrained model or None (random initialization).
- **input_tensor**: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.
- **input_shape**: shape of input image. format HxWxC PASCAL VOC model was trained on (512, 512, 3) images.
    None is allowed as shape/width.
- **classes**: number of desired classes <font color=#FF0000>(will be deprecated)</font>.
    PASCAL VOC has 21 classes, Cityscapes has 19 classes.
    If number of classes not aligned with the weights used,
    last layer is initialized randomly.
- **categorical_num**: An integer,
    number of categories without background.
- **backbone**: backbone to use. one of {'xception','mobilenetv2'}
- **OS**: determines input_shape/feature_extractor_output ratio.
    One of {8,16}.
    Used only for xception backbone.
- **alpha**: controls the width of the MobileNetV2 network.
    This is known as the width multiplier in the MobileNetV2 paper.
  - If `alpha` < 1.0, proportionally decreases the number of filters in each layer.
  - If `alpha` > 1.0, proportionally increases the number of filters in each layer.
  - If `alpha` = 1, default number of filters from the paper are used at each layer.
    Used only for mobilenetv2 backbone.
    Pretrained is only available for alpha=1.
- **activation**: optional activation to add to the top of the network.
    One of 'softmax', 'sigmoid' or None.

**Returns**
A tf.keras model instance.

**Raises**
- **RuntimeError**: If attempting to run this model with a backend that does not support separable convolutions.
- **ValueError**: in case of invalid argument for `backbone`.

---

### <font color=#FF00FF>besnet</font> function
```
tf2_Segmentation.models.besnet(
    pretrained_weights=None,
    input_shape=(512, 512, 3),
    activation='selu',
    categorical_num=4,
    classifi_mode='one')
```
Create BES-Net network architecture.

**Arguments**

- **pretrained_weights**: A string, file path of pretrained model.
- **input_shape**: A tuple of 3 integers, shape of input image.
- **categorical_num**: An integer, number of categories without background.
- **activation**: A string, activation function for convolutional layer.
- **class_weight**:  A list, when the category is unbalanced, you can pass in the category weight list.
- **classifi_mode**: A string, one of 'one'、'binary'、'multi'.
    If specified as 'one', it means that the activation function of the output layer is softmax, and the label should be one-hot encoding.

**Returns**
A tf.keras Model.

---

### <font color=#FF00FF>mbesnet</font> function
```
tf2_Segmentation.models.mbesnet(
    pretrained_weights=None,
    input_shape=(512, 512, 3),
    activation='relu',
    categorical_num=4,
    classifi_mode='one')
```
Create mBES-Net network architecture.

**Arguments**
- **pretrained_weights**: A string, file path of pretrained model.
- **input_shape**: A tuple of 3 integers, shape of input image.
- **categorical_num**: An integer, number of categories
- **activation**: A string, activation function for convolutional layer.
- **class_weight**: A list, when the category is unbalanced, you can pass in the category weight list.
- **classifi_mode**: A string, one of 'one'、'binary'、'multi'.
    If specified as 'one', it means that the activation function  of the output layer is softmax, and the label should be one-hot encoding.

**Returns**
A tf.keras Model.

------

## <font color=#0000FF>losses</font>

### <font color=#FF00FF>balanced_categorical_crossentropy</font> function
```
tf2_Segmentation.losses.balanced_categorical_crossentropy(class_weight=1)
```

**Arguments**
- **class_weight**: Optional `class_weight` acts as reduction weighting coefficient for the per-class losses. If a scalar is provided, then the loss is simply scaled by the given value.

**Returns**
A tf2 loss function.

---

### <font color=#FF00FF>balanced_binary_crossentropy</font> function
```
tf2_Segmentation.losses.balanced_binary_crossentropy(class_weight=1, binary_weight=1)
```

**Arguments**
- **class_weight**: Optional `class_weight` acts as reduction weighting coefficient for the per-class losses. If a scalar is provided, then the loss is simply scaled by the given value.
- **binary_weight**: Optional `binary_weight` acts as reduction weighting coefficient for the positive and negative losses.

**Returns**
A tf2 loss function.

---

### <font color=#FF00FF>categorical_be_crossentropy</font> function
```
tf2_Segmentation.losses.categorical_be_crossentropy(
    ce_class_weight=1,
    bece_class_weight=1,
    alpha=0.5, beta=0.1)
```
Loss function for BESNet.

**Arguments**
- **ce_class_weight**: Class weight of crossentropy.
- **bece_class_weight**: Class weight of boundary enhanced crossentropy.
- **alpha**: A float.
- **beta**: A float.

**Returns**
A tf2 loss function.

---

### <font color=#FF00FF>binary_be_crossentropy</font> function
```
tf2_Segmentation.losses.binary_be_crossentropy(
    ce_class_weight=1,
    bece_class_weight=1,
    ce_binary_weight=1,
    binary_weight=1,
    alpha=0.5, beta=0.1)
```
Loss function for BESNet.

**Arguments**
- **ce_class_weight**: Class weight of crossentropy.
- **bece_class_weight**: Class weight of boundary enhanced crossentropy.
- **ce_binary_weight**: Class weight of binary crossentropy.
- **binary_weight**: Class weight of boundary enhanced binary crossentropy.
- **alpha**: A float.
- **beta**: A float.

**Returns**
A tf2 loss function.

---

### <font color=#FF00FF>dice_loss_func</font> function
```
tf2_Segmentation.losses.dice_loss_func(smooth=1)
```
Loss function for BESNet.

**Arguments**
- **smooth**: An integer.

**Returns**
A tf2 loss function.

------

## <font color=#0000FF>metrics</font>

### <font color=#FF00FF>be_binary_accuracy</font> function
```
tf2_Segmentation.metrics.be_binary_accuracy(y_true, y_pred)
```
Accuracy function for BESNet.

**Arguments**
- **y_true**: Ground truth values.
- **y_pred**: The predicted values.

**Returns**
Accuracy values.

---

### <font color=#FF00FF>dice_coef_func</font> function
```
tf2_Segmentation.metrics.dice_coef_func(smooth=1)
```
Dice coefficient function.

**Arguments**
- **smooth**: An integer.

**Returns**
Dice coefficient function.

------

## <font color=#0000FF>utils</font>

### The Segdata_reader class

#### <font color=#FF00FF>Segdata_reader</font> class
```
tf2_Segmentation.utils.Segdata_reader(
    rescale=None,
    augmenter=None,
    aug_times=1)
```
Read the images and annotations for segmentation.

**Arguments**
- **rescale**: A float or None, specifying how the image value should be scaled.
    If None, no scaled.
- **augmenter**: A `imgaug.augmenters.meta.Sequential` instance.
- **aug_times**: An integer.
  The default is 1, which means no augmentation.

**Attributes**
- **rescale**
- **augmenter**
- **aug_times**
- **file_names**: A list of string with all file names that have been read.

**Return**
A reader instance for images and annotations.

#### <font color=#FF00FF>labelme_json_to_dataset</font> method

```
Segdata_reader.labelme_json_to_dataset(
    img_path=None, label_path=None,
    class_names=[], size=(512, 512),
    padding=True, line_thickness=5,
    shuffle=True, seed=None,
    classifi_mode="one",
    encoding="big5",
    thread_num=10)
```
Convert the JSON file generated by `labelme` into ndarray.

**Arguments**
- **img_path**: A string, file path of images.
    If JSON files include images, just specify one of arg (img_path、label_path) with JSON files path.
- **label_path**: A string, file path of JSON files.
- **class_names**: A list of string, the class names of the category in `labelme`.
    For example: ["g", "t", "v", "bg"], the format of the returned label is one-hot, and the channel is the class.
    For example, the channel is [1, 0, 0, 0, 0], which means the pixel is "g"; [0, 0, 0, 0, 1] means Nothing.
    The following format is also acceptable: ["g", ("t", "v"), "bg"], which means to treat "t" and "v" as a group.
- **size**: A tuple of 2 integers: (heights, widths), images will be resized to this arg.
- **padding**: A boolean, whether to fill the mark, default: True.
- **line_thickness**: A integer, the width of the drawn line, default: 5.
- **shuffle**: Boolean, default: True.
- **seed**: An integer, random seed, default: None.
- **classifi_mode**: A string, one of 'one'、'binary'、'multi', which means one-hot encode、binary encode and multi-hot encode respectively.
- **encoding**: A string, encoding format of JSON file, default: "big5".
- **thread_num**: An integer, specifying the number of threads to read files.

**Return**
- A tuple of Numpy arrays: (train data, label data)
  - train data:
      shape (batches, img heights, img widths, color channels).
  - label data:
      shape (batches, mask heights, mask widths, classes).

#### <font color=#FF00FF>imglayer_to_dataset</font> method

```
Segdata_reader.imglayer_to_dataset(
    img_path=None, label_path=None,
    class_colors=["r", "b"],
    size=(512, 512), shuffle=True, seed=None,
    classifi_mode="one",
    thread_num=10)
```
Convert the images and image layers into ndarray.

**Arguments**

- **img_path**: A string, file path of images.
    Or specify one of args (img_path、label_path) with a folder path includes images folde (should name as img) and masks folder (should name as label).
- **label_path**: A string, file path of segmentation masks.
- **class_colors**: A list of color string or RGB tuple of integer,
    Example of color string: `['r', 'b']`(Default).
        Check for more info about color string by the following url:
        https://matplotlib.org/tutorials/colors/colors.html
    Example of RGB tuple of integer:
        `[(255, 0, 0), (0, 0, 255)]`(which means Red、Blue).
- **size**: A tuple of integer: (heights, widths), images will be resized to this arg.
- **shuffle**: A boolean, default: True.
- **seed**: An integer, random seed, default: None.
- **classifi_mode**: A string, one of 'one'、'binary'、'multi', which means one-hot encode、binary encode and multi-hot encode respectively.
- **thread_num**: An integer, specifying the number of threads to read files.

**Return**
- A tuple of Numpy arrays: (train data, label data)
  - train data:
      shape (batches, img heights, img widths, color channels).
  - label data:
      shape (batches, mask heights, mask widths, classes).

---

### <font color=#FF00FF>read_img</font> function

```
tf2_Segmentation.utils.read_img(
    path,
    size=(512, 512),
    rescale=None)
```
Read images as ndarray

**Arguments**
- **path**: A string, path of images.
- **size**: A tuple of 2 integers, (heights, widths).
- **rescale**: A float or None, specifying how the image value should be scaled.
    If None, no scaled.

**Return**
A tuple of Numpy arrays (batches, img heights, img widths, color channels).

---

### <font color=#FF00FF>vis_img_mask</font> function

```
tf2_Segmentation.utils.vis_img_mask(
    img, label,
    color=['r', 'lime', 'b', 'c', 'm', 'y', 'pink', 'w'],
    label_alpha=0.3,
    classifi_mode='one',
    return_array=False)
```
Visualize images and annotations.

**Arguments**
- img: A ndarry of shape(img heights, img widths, color channels).
- label: A ndarray, shape should be the same as `img`.
- color: A list of color string or RGB tuple of float.
    Example of color string:
        `['r', 'lime', 'b', 'c', 'm', 'y', 'pink', 'w']`(Default).
        check for more info about color string by the following url:
        https://matplotlib.org/tutorials/colors/colors.html
    Example of RGB tuple of float:
        `[(1, 0, 0), (0, 0, 1)]`(which means red、blue).
- label_alpha: A float, transparency of annotation mask.
- classifi_mode: A string, one of 'one'、'binary'、'multi'.          
    If the label encode is one-hot, please specify as 'one'.
- return_array: A boolean, default is False.

**Return**
None or an image ndarray.

---

### <font color=#FF00FF>plot_history</font> function

```
tf2_Segmentation.utils.plot_history(
    history_dict, keys,
    title=None, xyLabel=[],
    ylim=(), size=())
```
Draw history line graph.
The historical records such as the loss value or accuracy rate returned during training can be drawn into a line graph.

**Arguments**
- **history**: A dictionary, containing one or more data to be drawn,
    for example: `{'loss': [4,2,1,…], 'acc': [2,3,5,…]}`.
- **keys**: A tuple or list, specifying the key value to be drawn in history, for example: `('loss', 'acc')`.
- **title**: A string, specifying the title text of the chart.
- **xyLabel**: A tuple or list, specifying the description text of x, y axis, for example:`('epoch', 'Accuracy')`.
- **ylim**: A tuple or list, specifying the minimum and maximum values ​​of the y-axis, for example: `(1, 3)`, values ​​outside the range will be ignored.
- **size**: A tuple, specifying the size of the picture, the default is `(6, 4)` (representing width 6 and height 4 inches).

---

### <font color=#FF00FF>get_class_weight</font> function

```
tf2_Segmentation.utils.get_class_weight(label_data, method="alpha")
```
Get the weight of the category.

**Arguments**
- **label_data**: A ndarray of shape (batch_size, grid_num, grid_num, info).
- **method**: A string, one of "alpha"、"log"、"effective"、"binary".

**Returns**
A list containing the weight of each category.

---

### <font color=#FF00FF>get_jaccard</font> function
```
tf2_Segmentation.utils.get_jaccard(
    ground_truth,
    prediction,
    class_names,
    classifi_mode='one')
```
Get Jaccard index table.

**Arguments**
- **ground_truth**: A ndarray.
- **prediction**: A ndarray.
- **class_names**: A list of string, corresponding names of classes.
- **classifi_mode**: A string, one of 'one'、'binary'、'multi'.          
    If the label encode is one-hot, please specify as 'one'.

**Returns**
A pandas.Series.

---

### <font color=#FF00FF>get_dice</font> function
```
tf2_Segmentation.utils.get_dice(
    ground_truth,
    prediction,
    class_names,
    classifi_mode='one')
```
Get Dice coefficient table.

**Arguments**
- **ground_truth**: A ndarray.
- **prediction**: A ndarray.
- **class_names**: A list of string, corresponding names of classes.
- **classifi_mode**: A string, one of 'one'、'binary'、'multi'.          
    If the label encode is one-hot, please specify as 'one'.

**Returns**
A pandas.Series.

---

### <font color=#FF00FF>create_confusion_mat</font> function
```
tf2_Segmentation.utils.create_confusion_mat(
    ground_truth,
    prediction,
    class_names,
    groundtruth_name="groundtruth",
    prediction_name="prediction",
    nothing_name="nothing")
```
Create a confusion matrix for multi-category segmentation.

**Arguments**
- **ground_truth**: A ndarray.
- **prediction**: A ndarray.
- **class_names**: A list of string, corresponding names of classes.
- **groundtruth_name**: A string.
- **prediction_name**: A string.
- **nothing_name**: A string.

**Returns**
A pandas.Dataframe.

---

### <font color=#FF00FF>create_score_mat</font> function
```
tf2_Segmentation.utils.create_score_mat(confusion_mat)
```
Create score matrix table.

**Arguments**
- **confusion_mat**: pandas.Dataframe, you can get this from `create_confusion_mat()`.

**Return**
A pandas.Dataframe.