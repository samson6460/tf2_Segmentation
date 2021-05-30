# Copyright 2020 Samson Woof. All Rights Reserved.
# =============================================================================

"""Unet model.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import BatchNormalization as BN
from tensorflow.python.keras.utils.data_utils import get_file


def Conv2D_Acti_BN(input_tensor, activation, *args):
    output_tensor = Conv2D(*args,
                           activation=activation,
                           padding='same',
                           kernel_initializer='he_normal')(input_tensor)
    output_tensor = BN()(output_tensor)
    return output_tensor


def UpConv2D_Acti_BN(input_tensor, activation, *args):
    output_tensor = UpSampling2D(size = (2, 2))(input_tensor)
    output_tensor = Conv2D(*args,
                           activation=activation,
                           padding='same',
                           kernel_initializer='he_normal')(output_tensor)
    output_tensor = BN()(output_tensor)
    return output_tensor


def unet(pretrained_weights=None,
         input_shape=(512, 512, 3),
         activation='relu',
         categorical_num=4,
         classifi_mode='one'):
    """Create U-Net architecture.
    
    Args:
        pretrained_weights: A string, 
            file path of pretrained model.
        input_shape: A tuple of 3 integers,
            shape of input image.
        activation: A string,
            activation function for convolutional layer.
        categorical_num:  An integer,
            number of categories without background.
        classifi_mode: A string,
            one of 'one'、'binary'、'multi'.
            If specified as 'one', it means that the activation function
            of the output layer is softmax, and the label 
            should be one-hot encoding.

    Returns:
        A tf.keras Model.
    """
    inputs = Input(input_shape)
    conv1 = Conv2D_Acti_BN(inputs, activation, 64, 3)
    conv1 = Conv2D_Acti_BN(conv1, activation, 64, 3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D_Acti_BN(pool1, activation, 128, 3)
    conv2 = Conv2D_Acti_BN(conv2, activation, 128, 3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D_Acti_BN(pool2, activation, 256, 3)
    conv3 = Conv2D_Acti_BN(conv3, activation, 256, 3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D_Acti_BN(pool3, activation, 512, 3)
    conv4 = Conv2D_Acti_BN(conv4, activation, 512, 3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D_Acti_BN(pool4, activation, 1024, 3)
    conv5 = Conv2D_Acti_BN(conv5, activation, 1024, 3)

    up6 = UpConv2D_Acti_BN(conv5, activation, 512, 2)
    merge6 = concatenate([conv4, up6], axis = 3)
    conv6 = Conv2D_Acti_BN(merge6, activation, 512, 3)
    conv6 = Conv2D_Acti_BN(conv6, activation, 512, 3)

    up7 = UpConv2D_Acti_BN(conv6, activation, 256, 2)
    merge7 = concatenate([conv3, up7], axis = 3)
    conv7 = Conv2D_Acti_BN(merge7, activation, 256, 3)
    conv7 = Conv2D_Acti_BN(conv7, activation, 256, 3)

    up8 = UpConv2D_Acti_BN(conv7, activation, 128, 2)
    merge8 = concatenate([conv2, up8], axis = 3)
    conv8 = Conv2D_Acti_BN(merge8, activation, 128, 3)
    conv8 = Conv2D_Acti_BN(conv8, activation, 128, 3)

    up9 = UpConv2D_Acti_BN(conv8, activation, 64, 2)
    merge9 = concatenate([conv1, up9], axis = 3)
    conv9 = Conv2D_Acti_BN(merge9, activation, 64, 3)
    conv9 = Conv2D_Acti_BN(conv9, activation, 64, 3)

    if classifi_mode=='one':
        conv10 = Conv2D(categorical_num+1, 1, activation='softmax')(conv9)
    else:
        conv10 = Conv2D(categorical_num, 1, activation='sigmoid')(conv9)

    model = Model(inputs, conv10)
    
    if pretrained_weights is not None:    
        model.load_weights(pretrained_weights)

    return model