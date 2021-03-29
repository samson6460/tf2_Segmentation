# Copyright 2020 Samson Woof. All Rights Reserved.
# =============================================================================

"""BES-Net model and mBES-Net.

Based on:
https://link.springer.com/chapter/10.1007/978-3-030-00934-2_26
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization as BN


def Conv2D_Acti_BN(input_tensor, activation, *args):
    output_tensor = Conv2D(*args,
                           activation=activation,
                           padding='same',
                           kernel_initializer='he_normal')(input_tensor)
    output_tensor = BN()(output_tensor)
    return output_tensor


def Conv2DTrans_Acti_BN(input_tensor, activation, *args):
    output_tensor = Conv2DTranspose(
                        *args,
                        activation=activation,
                        padding='same',
                        kernel_initializer='he_normal')(input_tensor)
    output_tensor = BN()(output_tensor)
    return output_tensor


def besnet(pretrained_weights=None,
           input_shape=(512, 512, 3),
           activation='selu',
           categorical_num=4,
           classifi_mode='one'):
    """Create BES-Net network architecture.
    
    Args:
        pretrained_weights: A string, 
            file path of pretrained model.
        input_shape: A tuple of 3 integers,
            shape of input image.
        categorical_num:  An integer,
            number of categories without background.
        activation: A string,
            activation function for convolutional layer.
        class_weight:  A list,
            when the category is unbalanced, 
            you can pass in the category weight list.
        classifi_mode: A string,
            one of 'one'、'binary'、'multi'.
            If specified as 'one', it means that the activation function 
            of the output layer is softmax, and the label 
            should be one-hot encoding.

    Returns:
        A tf.keras Model.
    """
    inputs = Input(input_shape)
    ENP_conv1 = Conv2D_Acti_BN(inputs, activation, 32, 3)
    ENP_conv1 = Conv2D_Acti_BN(ENP_conv1, activation, 64, 3)
    ENP_pool1 = MaxPooling2D(pool_size=(2, 2))(ENP_conv1)

    ENP_conv2 = Conv2D_Acti_BN(ENP_pool1, activation, 64, 3)
    ENP_conv2 = Conv2D_Acti_BN(ENP_conv2, activation, 128, 3)
    ENP_pool2 = MaxPooling2D(pool_size=(2, 2))(ENP_conv2)

    ENP_conv3 = Conv2D_Acti_BN(ENP_pool2, activation, 128, 3)
    ENP_conv3 = Conv2D_Acti_BN(ENP_conv3, activation, 256, 3)
    ENP_pool3 = MaxPooling2D(pool_size=(2, 2))(ENP_conv3)

    ENP_conv4 = Conv2D_Acti_BN(ENP_pool3, activation, 256, 3)
    ENP_conv4 = Conv2D_Acti_BN(ENP_conv4, activation, 512, 3)
    ENP_pool4 = MaxPooling2D(pool_size=(2, 2))(ENP_conv4)

    ENP_conv5 = Conv2D_Acti_BN(ENP_pool4, activation, 512, 3)
    ENP_conv5 = Conv2D_Acti_BN(ENP_conv5, activation, 1024, 3)

    BDP_tconv1 = Conv2DTrans_Acti_BN(ENP_conv5, activation, 512, 3, 2)
    BDP_sum1 = BDP_tconv1 + ENP_conv4
    BDP_conv1 = Conv2D_Acti_BN(BDP_sum1, activation, 512, 3)
    BDP_conv1 = Conv2D_Acti_BN(BDP_conv1, activation, 256, 3)

    BDP_tconv2 = Conv2DTrans_Acti_BN(BDP_conv1, activation, 256, 3, 2)
    BDP_sum2 = BDP_tconv2 + ENP_conv3
    BDP_conv2 = Conv2D_Acti_BN(BDP_sum2, activation, 256, 3)
    BDP_conv2 = Conv2D_Acti_BN(BDP_conv2, activation, 256, 3)

    BDP_tconv3 = Conv2DTrans_Acti_BN(BDP_conv2, activation, 128, 3, 2)
    BDP_sum3 = BDP_tconv3 + ENP_conv2
    BDP_conv3 = Conv2D_Acti_BN(BDP_sum3, activation, 128, 3)
    BDP_conv3 = Conv2D_Acti_BN(BDP_conv3, activation, 128, 3)

    BDP_tconv4 = Conv2DTrans_Acti_BN(BDP_conv3, activation, 64, 3, 2)
    BDP_sum4 = BDP_tconv4 + ENP_conv1
    BDP_conv4 = Conv2D_Acti_BN(BDP_sum4, activation, 64, 3)
    BDP_conv4 = Conv2D_Acti_BN(BDP_conv4, activation, 64, 3)

    MDP_tconv1 = Conv2DTrans_Acti_BN(ENP_conv5, activation, 512, 3, 2)
    MDP_sum1 = MDP_tconv1 + ENP_conv4
    MDP_conv1 = Conv2D_Acti_BN(MDP_sum1, activation, 512, 3)
    MDP_conv1 = Conv2D_Acti_BN(MDP_conv1, activation, 256, 3)
    MDP_merge1 = concatenate([BDP_conv1, MDP_conv1], axis = 3)

    MDP_tconv2 = Conv2DTrans_Acti_BN(MDP_merge1, activation, 256, 3, 2)
    MDP_sum2 = MDP_tconv2 + ENP_conv3
    MDP_conv2 = Conv2D_Acti_BN(MDP_sum2, activation, 256, 3)
    MDP_conv2 = Conv2D_Acti_BN(MDP_conv2, activation, 256, 3)
    MDP_merge2 = concatenate([BDP_conv2,MDP_conv2], axis = 3)

    MDP_tconv3 = Conv2DTrans_Acti_BN(MDP_merge2, activation, 128, 3, 2)
    MDP_sum3 = MDP_tconv3 + ENP_conv2
    MDP_conv3 = Conv2D_Acti_BN(MDP_sum3, activation, 128, 3)
    MDP_conv3 = Conv2D_Acti_BN(MDP_conv3, activation, 128, 3)
    MDP_merge3 = concatenate([BDP_conv3, MDP_conv3], axis = 3)

    MDP_tconv4 = Conv2DTrans_Acti_BN(MDP_merge3, activation, 64, 3, 2)
    MDP_sum4 = MDP_tconv4 + ENP_conv1
    MDP_conv4 = Conv2D_Acti_BN(MDP_sum4, activation, 64, 3)
    MDP_conv4 = Conv2D_Acti_BN(MDP_conv4, activation, 64, 3)

    if classifi_mode=='one':
        BDP_output = Conv2D(categorical_num + 1, 1, 
                            activation = 'softmax')(BDP_conv4)
        MDP_output = Conv2D(categorical_num+1, 1,
                            activation = 'softmax')(MDP_conv4)
    else:
        BDP_output = Conv2D(categorical_num, 1,
                            activation = 'sigmoid')(BDP_conv4)
        MDP_output = Conv2D(categorical_num, 1, 
                            activation = 'sigmoid')(MDP_conv4)
    
    outputs = concatenate([BDP_output, MDP_output], axis = 3)
    model = Model(inputs, outputs)
    
    if pretrained_weights is not None:   
        model.load_weights(pretrained_weights)

    return model


def mbesnet(pretrained_weights = None,
            input_shape=(512, 512, 3),
            activation='relu',
            categorical_num=4,
            classifi_mode='one'):
    """Create mBES-Net network architecture.
    
    Args:
        pretrained_weights: A string, 
            file path of pretrained model.
        input_shape: A tuple of 3 integers,
            shape of input image.
        categorical_num:  An integer,
            number of categories
        activation: A string,
            activation function for convolutional layer.
        class_weight:  A list,
            when the category is unbalanced, 
            you can pass in the category weight list.
        classifi_mode: A string,
            one of 'one'、'binary'、'multi'.
            If specified as 'one', it means that the activation function 
            of the output layer is softmax, and the label 
            should be one-hot encoding.

    Returns:
        A tf.keras Model.
    """
    inputs = Input(input_shape)
    ENP_conv1 = Conv2D_Acti_BN(inputs, activation, 32, 3)
    ENP_conv1 = Conv2D_Acti_BN(ENP_conv1, activation, 64, 3)
    ENP_pool1 = MaxPooling2D(pool_size=(2, 2))(ENP_conv1)

    ENP_conv2 = Conv2D_Acti_BN(ENP_pool1, activation, 64, 3)
    ENP_conv2 = Conv2D_Acti_BN(ENP_conv2, activation, 128, 3)
    ENP_pool2 = MaxPooling2D(pool_size=(2, 2))(ENP_conv2)

    ENP_conv3 = Conv2D_Acti_BN(ENP_pool2, activation, 128, 3)
    ENP_conv3 = Conv2D_Acti_BN(ENP_conv3, activation, 256, 3)
    ENP_pool3 = MaxPooling2D(pool_size=(2, 2))(ENP_conv3)

    ENP_conv4 = Conv2D_Acti_BN(ENP_pool3, activation, 256, 3)
    ENP_conv4 = Conv2D_Acti_BN(ENP_conv4, activation, 512, 3)
    ENP_pool4 = MaxPooling2D(pool_size=(2, 2))(ENP_conv4)

    ENP_conv5 = Conv2D_Acti_BN(ENP_pool4, activation, 512, 3)
    ENP_conv5 = Conv2D_Acti_BN(ENP_conv5, activation, 1024, 3)

    BDP_tconv1 = Conv2DTrans_Acti_BN(ENP_conv5, activation, 512, 3, 2)
    BDP_sum1 = concatenate([BDP_tconv1, ENP_conv4], axis = 3)
    BDP_conv1 = Conv2D_Acti_BN(BDP_sum1, activation, 512, 3)
    BDP_conv1 = Conv2D_Acti_BN(BDP_conv1, activation, 256, 3)

    BDP_tconv2 = Conv2DTrans_Acti_BN(BDP_conv1, activation, 256, 3, 2)
    BDP_sum2 = concatenate([BDP_tconv2, ENP_conv3], axis = 3)
    BDP_conv2 = Conv2D_Acti_BN(BDP_sum2, activation, 256, 3)
    BDP_conv2 = Conv2D_Acti_BN(BDP_conv2, activation, 256, 3)

    BDP_tconv3 = Conv2DTrans_Acti_BN(BDP_conv2, activation, 128, 3, 2)
    BDP_sum3 = concatenate([BDP_tconv3, ENP_conv2], axis = 3)
    BDP_conv3 = Conv2D_Acti_BN(BDP_sum3, activation, 128, 3)
    BDP_conv3 = Conv2D_Acti_BN(BDP_conv3, activation, 128, 3)

    BDP_tconv4 = Conv2DTrans_Acti_BN(BDP_conv3, activation, 64, 3, 2)
    BDP_sum4 = concatenate([BDP_tconv4, ENP_conv1], axis = 3)
    BDP_conv4 = Conv2D_Acti_BN(BDP_sum4, activation, 64, 3)
    BDP_conv4 = Conv2D_Acti_BN(BDP_conv4, activation, 64, 3)

    MDP_tconv1 = Conv2DTrans_Acti_BN(ENP_conv5, activation, 512, 3, 2)
    MDP_sum1 = concatenate([MDP_tconv1, ENP_conv4], axis = 3)
    MDP_conv1 = Conv2D_Acti_BN(MDP_sum1, activation, 512, 3)
    MDP_conv1 = Conv2D_Acti_BN(MDP_conv1, activation, 256, 3)
    MDP_merge1 = BDP_conv1 + MDP_conv1

    MDP_tconv2 = Conv2DTrans_Acti_BN(MDP_merge1, activation, 256, 3, 2)
    MDP_sum2 = concatenate([MDP_tconv2, ENP_conv3], axis = 3)
    MDP_conv2 = Conv2D_Acti_BN(MDP_sum2, activation, 256, 3)
    MDP_conv2 = Conv2D_Acti_BN(MDP_conv2, activation, 256, 3)
    MDP_merge2 = BDP_conv2 + MDP_conv2

    MDP_tconv3 = Conv2DTrans_Acti_BN(MDP_merge2, activation, 128, 3, 2)
    MDP_sum3 = concatenate([MDP_tconv3, ENP_conv2], axis = 3)
    MDP_conv3 = Conv2D_Acti_BN(MDP_sum3, activation, 128, 3)
    MDP_conv3 = Conv2D_Acti_BN(MDP_conv3, activation, 128, 3)
    MDP_merge3 = BDP_conv3 + MDP_conv3

    MDP_tconv4 = Conv2DTrans_Acti_BN(MDP_merge3, activation, 64, 3, 2)
    MDP_sum4 = concatenate([MDP_tconv4, ENP_conv1], axis = 3)
    MDP_conv4 = Conv2D_Acti_BN(MDP_sum4, activation, 64, 3)
    MDP_conv4 = Conv2D_Acti_BN(MDP_conv4, activation, 64, 3)

    if classifi_mode=='one':
        BDP_output = Conv2D(categorical_num + 1, 1,
                            activation = 'softmax')(BDP_conv4)
        MDP_output = Conv2D(categorical_num + 1, 1,
                            activation = 'softmax')(MDP_conv4)
    else:
        BDP_output = Conv2D(categorical_num, 1,
                            activation = 'sigmoid')(BDP_conv4)
        MDP_output = Conv2D(categorical_num, 1,
                            activation = 'sigmoid')(MDP_conv4)
    
    outputs = concatenate([BDP_output, MDP_output], axis = 3)
    model = Model(inputs, outputs)
    
    if pretrained_weights is not None:      
        model.load_weights(pretrained_weights)

    return model