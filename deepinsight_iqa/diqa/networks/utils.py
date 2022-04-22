import tensorflow as tf
from scipy.stats import spearmanr
from tensorflow.keras import metrics as KMetric
from ..utils.tf_imgutils import rescale
from functools import partial

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
def loss_fn(model_, x, y_true, r):
    y_pred = model_(x, True)
    return tf.reduce_mean(tf.square((y_true - y_pred) * r))


@tf.function
def gradient(model, x, y_true, r):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(model, x, y_true, r)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# def calculate_error_map(features, SCALING_FACTOR=1 / 32):
#     I_d = image_preprocess(features['distorted_image'])
#     I_r = image_preprocess(features['reference_image'])
#     r = rescale(average_reliability_map(I_d, 0.2), SCALING_FACTOR)
#     e_gt = rescale(error_map(I_r, I_d, 0.2), SCALING_FACTOR)
#     return (I_d, e_gt, r)

def calculate_error_map(
    I_d: tf.Tensor,
    I_r: tf.Tensor,
    scaling_factor: float = 1. / 32
):
    r = rescale(average_reliability_map(I_d, 0.2), scaling_factor)
    e_gt = rescale(error_map(I_r, I_d, 0.2), scaling_factor)
    return e_gt, r


class SpearmanCorrMetric(KMetric.Metric):
    def __init__(self, name='spearman-corr', dtype=None, **kwargs):
        super(SpearmanCorrMetric, self).__init__(name, dtype, **kwargs)
        self.corr_score = self.add_weight(name='corr-score', initializer='zeros')

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        _spearmanr = partial(spearmanr, axis=None)
        self.corr_score.assign_add(
            tf.py_function(
                _spearmanr,
                inp=[tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)],
                Tout=tf.float32
            ))

    def result(self):
        return self.corr_score
