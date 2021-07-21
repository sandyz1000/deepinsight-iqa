import tensorflow as tf
from .utils import (show_images, image_normalization,
                    image_preprocess, error_map, average_reliability_map)
from typing import List, Dict


def show_distorted_reference_image(ds: List[Dict], do_gray=False):
    """Show distorted and reference image

    Arguments:
        features {[type]} -- [description]
    """
    for features in ds:
        distorted_image = features['distorted_image']
        reference_image = features['reference_image']
        I_d = None
        if do_gray:
            I_d = image_preprocess(distorted_image)
            I_d = tf.image.grayscale_to_rgb(distorted_image)
            I_d = image_normalization(distorted_image, 0, 1)

        dmos = tf.round(features['dmos'][0], 2)
        distortion = features['distortion'][0]
        print(f'The distortion of the image is {dmos} with'
              f' a distortion {distortion} and shape {distorted_image.shape}')
        show_images([reference_image, distorted_image, I_d])


def show_error_map_image(ds: List[Dict]):
    """Show images on applying image normalization to error_map and distorted image 

    Arguments:
        features {[type]} -- [description]
    """
    for features in ds:
        reference_image = features['reference_image']
        I_r = image_preprocess(reference_image)
        I_d = image_preprocess(features['distorted_image'])
        e_gt = error_map(I_r, I_d, 0.2)
        I_d = image_normalization(tf.image.grayscale_to_rgb(I_d), 0, 1)
        e_gt = image_normalization(tf.image.grayscale_to_rgb(e_gt), 0, 1)
        show_images([reference_image, I_d, e_gt])


def show_average_reliability_map_image(ds: List[Dict]):
    """Show images after applying average reliability transformation on distorted images 

    Arguments:
        features {[type]} -- [description]
    """
    for features in ds:
        reference_image = features['reference_image']
        I_d = image_preprocess(features['distorted_image'])
        r = average_reliability_map(I_d, 1)
        r = image_normalization(tf.image.grayscale_to_rgb(r), 0, 1)
        show_images([reference_image, r], cmap='gray')
