from typing import List
from PIL import Image as IMG
import numpy as np
import tensorflow as tf
import csv
from skimage import feature
import matplotlib.pyplot as plt
from threading import Lock
import logging
import sys


def plot_image(img_path):
    im1 = IMG.open(img_path)
    im2 = im1.convert(mode='L')
    im = np.asarray(im2)

    edges1 = feature.canny(im, sigma=1)
    edges2 = feature.canny(im, sigma=3)

    # display results
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)

    ax1.imshow(im, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('noisy image', fontsize=20)

    ax2.imshow(edges1, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

    ax3.imshow(edges2, cmap=plt.cm.gray)
    ax3.axis('off')
    ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

    fig.tight_layout()

    plt.show()


def load_samples(sample_file):
    with open(sample_file) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for row in csv_reader:
            yield row


def show_images(images: List[tf.Tensor], **kwargs):
    fig, axs = plt.subplots(1, len(images), figsize=(19, 10))
    for image, ax in zip(images, axs):
        assert image.get_shape().ndims in (3, 4), 'The tensor must be of dimension 3 or 4'
        if image.get_shape().ndims == 4:
            image = tf.squeeze(image)

        _ = ax.imshow(image, **kwargs)
        ax.axis('off')
    fig.tight_layout()


def thread_safe_memoize(func):
    cache = {}
    session_lock = Lock()

    def memoizer(*args, **kwargs):
        with session_lock:
            key = str(args) + str(kwargs)
            if key not in cache:
                cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoizer


class thread_safe_singleton(type):
    _instances = {}
    session_lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls.session_lock:
            if cls not in cls._instances:
                cls._instances[cls] = super(thread_safe_singleton, cls).__call__(*args, **kwargs)
            return cls._instances[cls]


def set_gpu_limit(limit=2):
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * limit)]
            )
            return f"GPU Limit set to: {1024*limit} MB"
        except RuntimeError as e:
            raise e


def get_stream_handler():
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(name)-12s %(levelname)-8s %(message)s", "%Y-%m-%dT%H:%M:%S"
    ))
    return stdout_handler