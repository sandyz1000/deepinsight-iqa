import math
import tensorflow as tf
import cv2
import numpy as np
from scipy import signal


def image_normalization(image: np.ndarray, new_min=0, new_max=255) -> np.ndarray:
    """
    Normalize the input image to a given range set by min and max parameter
    Args:
        image ([type]): [description]
        new_min ([type], optional): [description]. Defaults to 0.
        new_max ([type], optional): [description]. Defaults to 255.

    Returns:
        [np.ndarray]: Normalized image
    """
    original_dtype = image.dtype
    image = image.astype(np.float32)
    image_min, image_max = np.min(image), np.max(image)
    image = tf.cast(image, np.float32)

    normalized_image = (new_max - new_min) / (image_max - image_min) * (image - image_min) + new_min
    return tf.cast(normalized_image, original_dtype)


def normalize_kernel(kernel: np.array) -> np.ndarray:
    return kernel / np.sum(kernel, axis=-1)


def gaussian_kernel2d(kernel_size: int, sigma: float, dtype=np.float32) -> np.ndarray:
    krange = np.arange(kernel_size)
    x, y = np.meshgrid(krange, krange)
    constant = np.round(kernel_size / 2)
    x -= constant
    y -= constant
    kernel = 1 / (2 * math.pi * sigma**2) * np.math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return normalize_kernel(kernel)


def gaussian_filter(
    image: np.ndarray, kernel_size: int,
    sigma: float, dtype=np.float32, strides: int = 1
) -> np.ndarray:
    """
    Apply convolution filter to image with gaussian image kernel
    
    TODO: Verify this methos with tensorflow
    https://stackoverflow.com/questions/48097941/strided-convolution-of-2d-in-numpy

    Args:
        image ([np.ndarray]): [description]
        kernel_size ([int]): [description]
        sigma ([float]): [description]
        dtype ([type], optional): [description]. Defaults to np.float32.
        strides ([int], optional): [description]. Defaults to 1.

    Returns:
        [np.ndarray]: [description]
    """
    kernel = gaussian_kernel2d(kernel_size, sigma)
    if len(image.shape) == 3:
        image = image[np.newaxis, ...]
    image = tf.cast(image, tf.float32)
    image = image.astype(np.float32)
    image = signal.convolve2d(image, kernel[:, :, np.newaxis, np.newaxis], mode='same', )[::strides, ::strides]
    return image.astype(dtype)


def image_shape(image: np.ndarray, dtype=np.int32) -> np.ndarray:
    shape = image.shape
    shape = shape[:2] if len(image.shape) == 3 else shape[1:3]
    return shape


def scale_shape(image: np.ndarray, scale: float):
    shape = image_shape(image, np.float32)
    shape = np.math.ceil(shape * scale)
    return shape.astype(np.float32)


def rescale(image: np.ndarray, scale: float, dtype=np.float32, **kwargs) -> np.ndarray:
    assert len(image.shape) in (3, 4), 'The tensor must be of dimension 3 or 4'

    image = image.astype(np.float32)
    rescale_size = scale_shape(image, scale)
    interpolation = kwargs.pop('interpolation', cv2.INTER_CUBIC)
    rescaled_image = cv2.resize(image, rescale_size, interpolation=interpolation)
    return rescaled_image.astype(dtype)


def read_image(filename: str, **kwargs) -> np.ndarray:
    mode = kwargs.pop('mode', cv2.IMREAD_UNCHANGED)
    return cv2.imread(filename, flags=mode)


def image_preprocess(image: np.ndarray, SCALING_FACTOR=1 / 4) -> np.ndarray:
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
        image {np.ndarray} -- [description]

    Returns:
        np.ndarray -- [description]
    """
    image = tf.cast(image, tf.float32)
    image = tf.image.rgb_to_grayscale(image)
    image_low = gaussian_filter(image, 16, 7 / 6)
    image_low = rescale(image_low, SCALING_FACTOR, method=tf.image.ResizeMethod.BICUBIC)
    image_low = tf.image.resize(image_low,
                                size=image_shape(image),
                                method=tf.image.ResizeMethod.BICUBIC)
    return image - tf.cast(image_low, image.dtype)
