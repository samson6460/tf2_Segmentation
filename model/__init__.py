# Copyright 2020 Samson Woof. All Rights Reserved.
# =============================================================================

"""Segmeantation Models.
"""

from .unet_model import unet
from .deep_lab_model import deeplabv3
from .besnet_model import besnet, mbesnet

import numpy as np
from tensorflow.keras.utils import to_categorical


def model_predict(model, intput_tensor, classifi_mode="one", **kargs):
    """A function like model.predict_classes().

    It can call the `model.predict()` and convert
    the prediction from probabilities
    to one-hot encoding or binary encoding.
    
    Args:
        model: tf.kears model instance.
        intput_tensor: tf.keras tensor.
        classifi_mode: A string,
            Specifying the encoding method.
            Default: "one", which means one-hot encoding.
    """
    prediction = model.predict(intput_tensor, **kargs)
    if classifi_mode=="one":
        num_classes = prediction.shape[-1]
        prediction = to_categorical(prediction.argmax(axis=-1),
                                    num_classes=num_classes)
    else:
        prediction=np.where(prediction > 0.5, 1, 0)   
    return prediction