import math
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def image_normalization(image: tf.Tensor, new_min=0, new_max=255) -> tf.Tensor:
    original_dtype = image.dtype
    new_min = tf.constant(new_min, dtype=tf.float32)
    new_max = tf.constant(new_max, dtype=tf.float32)
    image_min = tf.cast(tf.reduce_min(image), tf.float32)
    image_max = tf.cast(tf.reduce_max(image), tf.float32)
    image = tf.cast(image, tf.float32)

    normalized_image = (new_max - new_min) / (image_max - image_min) * (image - image_min) + new_min
    return tf.cast(normalized_image, original_dtype)


def normalize_kernel(kernel: tf.Tensor) -> tf.Tensor:
    return kernel / tf.reduce_sum(kernel)


def gaussian_kernel2d(kernel_size: int, sigma: float, dtype=tf.float32) -> tf.Tensor:
    _range = tf.range(kernel_size)
    x, y = tf.meshgrid(_range, _range)
    constant = tf.cast(tf.round(kernel_size / 2), dtype=dtype)
    x = tf.cast(x, dtype=dtype) - constant
    y = tf.cast(y, dtype=dtype) - constant
    kernel = 1 / (2 * math.pi * sigma ** 2) * tf.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return normalize_kernel(kernel)


def gaussian_filter(image: tf.Tensor, kernel_size: int, sigma: float, dtype=tf.float32) -> tf.Tensor:
    kernel = gaussian_kernel2d(kernel_size, sigma)
    if image.get_shape().ndims == 3:
        image = image[tf.newaxis, :, :, :]
    image = tf.cast(image, tf.float32)
    image = tf.nn.conv2d(image, kernel[:, :, tf.newaxis, tf.newaxis], strides=1, padding='SAME')
    return tf.cast(image, dtype)


def image_shape(image: tf.Tensor, dtype=tf.int32) -> tf.Tensor:
    shape = tf.shape(image)
    shape = shape[:2] if image.get_shape().ndims == 3 else shape[1:3]
    return tf.cast(shape, dtype)


def scale_shape(image: tf.Tensor, scale: float) -> tf.Tensor:
    shape = image_shape(image, tf.float32)
    shape = tf.math.ceil(shape * scale)
    return tf.cast(shape, tf.int32)


def rescale(image: tf.Tensor, scale: float, dtype=tf.float32, **kwargs) -> tf.Tensor:
    assert image.get_shape().ndims in (3, 4), 'The tensor must be of dimension 3 or 4'

    image = tf.cast(image, tf.float32)
    rescale_size = scale_shape(image, scale)
    rescaled_image = tf.image.resize(image, size=rescale_size, **kwargs)
    return tf.cast(rescaled_image, dtype)


def read_image(filename: str, **kwargs) -> tf.Tensor:
    stream = tf.io.read_file(filename)
    return tf.image.decode_image(stream, **kwargs)


def random_crop(img, crop_dims):
    h, w = img.shape[0], img.shape[1]
    ch, cw = crop_dims[0], crop_dims[1]
    assert h >= ch, 'image height is less than crop height'
    assert w >= cw, 'image width is less than crop width'
    x = np.random.randint(0, w - cw + 1)
    y = np.random.randint(0, h - ch + 1)
    return img[y:(y + ch), x:(x + cw), :]


def random_vertical_flip(img):
    assert len(img.shape) == 3, 'input tensor must have 3 dimensions (height, width, channels)'
    assert img.shape[2] == 3, 'image not in channels last format'
    if np.random.random() < 0.5:
        img = img[..., ::-1]
    return img


def random_horizontal_flip(img):
    assert len(img.shape) == 3, 'input tensor must have 3 dimensions (height, width, channels)'
    assert img.shape[2] == 3, 'image not in channels last format'
    if np.random.random() < 0.5:
        img = img.swapaxes(1, 0)
        img = img[::-1, ...]
        img = img.swapaxes(0, 1)
    return img


def image_preprocess(image: tf.Tensor, SCALING_FACTOR=1 / 4) -> tf.Tensor:
    """
    #### Image Normalization

    The first step for DIQA is to pre-process the images. The image is converted into grayscale, 
    and then a low-pass filter is applied. The low-pass filter is defined as:

    \begin{align*}
    \hat{I} = I_{gray} - I^{low}
    \end{align*}

    where the low-frequency image is the result of the following algorithm:

    1. Blur the grayscale image.
    2. Downscale it by a factor of SCALING_FACTOR.
    3. Upscale it back to the original size.

    The main reasons for this normalization are (1) the Human Visual System (HVS) is not sensitive to changes 
    in the low-frequency band, and (2) image distortions barely affect the low-frequency component of images.

    Arguments:
        image {tf.Tensor} -- [description]

    Returns:
        tf.Tensor -- [description]
    """
    image = tf.cast(image, tf.float32)
    image = tf.image.rgb_to_grayscale(image)
    image_low = gaussian_filter(image, 16, 7 / 6)
    image_low = rescale(image_low, SCALING_FACTOR, method=tf.image.ResizeMethod.BICUBIC)
    image_low = tf.image.resize(image_low,
                                size=image_shape(image),
                                method=tf.image.ResizeMethod.BICUBIC)
    return image - tf.cast(image_low, image.dtype)


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


def loss(model_, x, y_true, r):
    y_pred = model_(x)
    return tf.reduce_mean(tf.square((y_true - y_pred) * r))


def optimizer():
    return tf.optimizers.Nadam(learning_rate=2 * 10 ** -4)


def gradient(model, x, y_true, r):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, y_true, r)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# def calculate_error_map(features, SCALING_FACTOR=1 / 32):
#     I_d = image_preprocess(features['distorted_image'])
#     I_r = image_preprocess(features['reference_image'])
#     r = rescale(average_reliability_map(I_d, 0.2), SCALING_FACTOR)
#     e_gt = rescale(error_map(I_r, I_d, 0.2), SCALING_FACTOR)
#     return (I_d, e_gt, r)


def calculate_subjective_score(features, key='mos'):
    I_d = image_preprocess(features['distorted_image'])
    mos = features[key]
    return (I_d, mos)


def calculate_error_map(I_d: tf.Tensor, I_r: tf.Tensor, SCALING_FACTOR: float = 1. / 32):
    r = rescale(average_reliability_map(I_d, 0.2), SCALING_FACTOR)
    e_gt = rescale(error_map(I_r, I_d, 0.2), SCALING_FACTOR)
    return (I_d, e_gt, r)
