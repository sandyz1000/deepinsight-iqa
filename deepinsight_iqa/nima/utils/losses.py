import numpy as np
from tensorflow.keras import backend as K


def earth_movers_distance(y_true, y_pred):
    cdf_true = K.cumsum(y_true, axis=-1)
    cdf_pred = K.cumsum(y_pred, axis=-1)
    emd = K.sqrt(K.mean(K.square(cdf_true - cdf_pred), axis=-1))
    return K.mean(emd)


def np_earth_movers_distance(y_true, y_pred):
    cdf_true = np.cumsum(y_true, axis=-1)
    cdf_pred = np.cumsum(y_pred, axis=-1)
    emd = np.sqrt(np.mean(np.square(cdf_true - cdf_pred), axis=-1))
    return np.mean(emd)
