import tensorflow as tf
from scipy.stats import spearmanr
import numpy as np
from tensorflow.keras import metrics as KMetric
from ..utils.tf_imgutils import rescale
from functools import partial
import tensorflow_probability as tfp


def error_map(reference: tf.Tensor, distorted: tf.Tensor, p: float = 0.2) -> tf.Tensor:
    """
    # Objective Error Map

    For the first model, objective errors are used as a proxy to take advantage of the effect of increasing data. 
    The loss function is defined by the mean squared error between the predicted and ground-truth error maps.

    # \begin{align*}
    # \mathbf{e}_{gt} = err(\hat{I}_r, \hat{I}_d)
    # \end{align*}

    and *err(·)* is an error function. For this implementation, the authors recommend using

    # \begin{align*}
    # \mathbf{e}_{gt} = | \hat{I}_r -  \hat{I}_d | ^ p
    # \end{align*}

    with *p=0.2*. The latter is to prevent that the values in the error map are small or close to zero.


    Arguments:
        reference {tf.Tensor} -- [description]
        distorted {tf.Tensor} -- [description]

    Keyword Arguments:
        p {float} -- [description] (default: {0.2})

    Returns:
        tf.Tensor -- [description]
    """
    assert reference.dtype == tf.float32 and distorted.dtype == tf.float32, 'dtype must be tf.float32'
    return tf.pow(tf.abs(reference - distorted), p)


def reliability_map(distorted: tf.Tensor, alpha: float) -> tf.Tensor:
    """
    ## Reliability Map

    According to the authors, the model is likely to fail to predict images with homogeneous regions. 
    To prevent it, they propose a reliability function. The assumption is that blurry areas have lower 
    reliability than textured ones. The reliability function is defined as

    # \begin{align*}
    # \mathbf{r} = \frac{2}{1 + exp(-\alpha|\hat{I}_d|)} - 1
    # \end{align*}

    score = n * sigmoid - n/2  (Low sigmoid score for blurry image and vice-versa; range -1 to 1 where n=2)
    where α controls the saturation property of the reliability map. The positive part of a sigmoid is
    used to assign sufficiently large values to pixels with low intensity.


    Arguments:
        distorted {tf.Tensor} -- [description]
        alpha {float} -- [description]

    Returns:
        tf.Tensor -- [description]
    """
    assert distorted.dtype == tf.float32, 'The Tensor must by of dtype tf.float32'
    return 2 / (1 + tf.exp(- alpha * tf.abs(distorted))) - 1


def average_reliability_map(distorted: tf.Tensor, alpha: float) -> tf.Tensor:
    """
    The previous definition might directly affect the predicted score. Therefore, the average reliability map 
    is used instead.

    # \begin{align*}
    # \mathbf{\hat{r}} = \frac{1}{\frac{1}{H_rW_r}\sum_{(i,j)}\mathbf{r}(i,j)}\mathbf{r}
    # \end{align*}

    For the Tensorflow function, we just calculate the reliability map and divide it by its mean.

    Arguments:
        distorted {tf.Tensor} -- [description]
        alpha {float} -- [description]

    Returns:
        tf.Tensor -- [description]
    """
    r = reliability_map(distorted, alpha)
    return r / tf.reduce_mean(r)


@tf.function
def loss_fn(model_fn, x, y_true, r):
    y_pred = model_fn(x)
    return tf.reduce_mean(tf.square((y_true - y_pred) * r))


@tf.function
def gradient(model_fn, x, y_true, r):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(model_fn, x, y_true, r)
    return loss_value, tape.gradient(loss_value, model_fn.trainable_variables)


@tf.function
def calculate_error_map(
    I_d: tf.Tensor,
    I_r: tf.Tensor,
    scaling_factor: float = 1. / 32
):
    r = rescale(average_reliability_map(I_d, 0.2), scaling_factor)
    e_gt = rescale(error_map(I_r, I_d, 0.2), scaling_factor)
    return e_gt, r


class PearsonCorrMetric(KMetric.Metric):
    def __init__(self, name='spearman-corr', dtype=None, **kwargs):
        super(PearsonCorrMetric, self).__init__(name, dtype, **kwargs)
        self.corr_score = self.add_weight(name='score', initializer='zeros')

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        corr = tfp.stats.correlation(y_true, y_pred, sample_axis=None, event_axis=None)
        self.corr_score.assign_add(corr)

    def result(self):
        return self.corr_score


class SpearmanCorrMetric(KMetric.Metric):
    def __init__(self, name='spearman-corr', dtype=None, **kwargs):
        super(SpearmanCorrMetric, self).__init__(name, dtype, **kwargs)
        self.corr_score = self.add_weight(name='corr-score', initializer='zeros')

    def get_rank(self, y_pred):
        rank = tf.argsort(tf.argsort(y_pred, axis=-1, direction="ASCENDING"), axis=-1) + \
            1  # +1 to get the rank starting in 1 instead of 0
        return rank

    def sp_rank(self, x, y):
        cov = tfp.stats.covariance(x, y, sample_axis=0, event_axis=None)
        sd_x = tfp.stats.stddev(x, sample_axis=0, keepdims=False, name=None)
        sd_y = tfp.stats.stddev(y, sample_axis=0, keepdims=False, name=None)
        return 1 - cov / (sd_x * sd_y)  # 1- because we want to minimize loss

    def spearman_correlation(self, y_true, y_pred):
        """
        First we obtain the ranking of the predicted values

        Example:
        --------
        Spearman rank correlation between each pair of samples:
        Sample dim: (1, 8)
        Batch of samples dim: (None, 8) None=batch_size=64
        Output dim: (batch_size, ) = (64, )

        :param _type_ y_true: _description_
        :param _type_ y_pred: _description_
        :return _type_: _description_
        """
        # TODO: FixMe
        to_shape = [y_true.get_shape()[0], np.multiply.reduce(y_true.get_shape()[1:])]
        y_pred = tf.reshape(y_pred, shape=to_shape)
        y_true = tf.reshape(y_true, shape=to_shape)

        y_pred_rank = tf.map_fn(lambda x: self.get_rank(x), y_pred, dtype=tf.float32)

        sp = tf.map_fn(lambda x: self.sp_rank(x[0], x[1]), (y_true, y_pred_rank), dtype=tf.float32)
        # Reduce to a single value
        loss = tf.reduce_mean(sp)
        return loss

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        
        corr = self.spearman_correlation(self, y_true, y_pred)

        # _spearmanr = partial(spearmanr, axis=None)
        # corr = tf.py_function(
        #     _spearmanr,
        #     inp=[tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)],
        #     Tout=tf.float32
        # )

        self.corr_score.assign_add(corr)

    def result(self):
        return self.corr_score
