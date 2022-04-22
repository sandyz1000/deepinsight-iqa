import pandas as pd
from typing import Union, Tuple
from typing import Callable
import numpy as np
import tensorflow as tf
from functools import partial


def get_tfdataset(
    image_dir: str,
    samples: Union[np.ndarray, pd.DataFrame],
    generator_fn: Callable,
    batch_size: int = 32,
    img_preprocessing: Callable = None,
    input_size: Tuple[int] = (256, 256),
    img_crop_dims: Tuple[int] = (224, 224),
    do_train: bool = False,
    do_augment: bool = False,
    channel_dim: int = 3,
    **kwds
) -> tf.data.Dataset:
    """Function to convert Keras Sequence to tensorflow dataset

    Args:
        image_dir ([str]): [description]
        samples ([Union[np.ndarray, pd.DataFrame]]): [description]
        generator_fn ([type]): [description]
        batch_size ([type], optional): [description]. Defaults to 32.
        img_preprocessing ([type], optional): [description]. Defaults to None.
        input_size ([type], optional): [description]. Defaults to (256, 256).
        img_crop_dims ([type], optional): [description]. Defaults to (224, 224).
        shuffle ([type], optional): [description]. Defaults to False.
        do_augment ([type], optional): [description]. Defaults to False.
        channel_dim ([type], optional): [description]. Defaults to 3.

    Returns:
        [tf.data.Dataset]: [description]
    """

    image_gen = partial(
        generator_fn,
        image_dir,
        samples,
        img_preprocessing=img_preprocessing,
        input_size=input_size,
        img_crop_dims=img_crop_dims,
        batch_size=batch_size,
        repeat=True,
        do_train=do_train,
        do_augment=do_augment,
        channel_dim=channel_dim,
    )

    steps_per_epoch = np.floor(len(samples) / batch_size)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if do_train:
        dataset = tf.data.Dataset.from_generator(
            image_gen,
            output_types=(tf.float32, tf.float32, tf.float32),
            output_shapes=([None, *input_size, channel_dim], [None, *input_size, channel_dim], [None]),
        )
    else:
        dataset = tf.data.Dataset.from_generator(
            image_gen,
            output_types=(tf.float32, ),
            output_shapes=([None, *input_size, channel_dim], ),
        )

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset, steps_per_epoch


def get_tfds_v2(
    image_dir: str,
    samples: Union[np.ndarray, pd.DataFrame],
    batch_size: int = 32,
    generator_fn: Callable = None,
    output_types: Tuple = None,
    output_shapes: Tuple = None,
    img_preprocessing: Callable = None,
    input_size: Tuple[int] = (256, 256),
    img_crop_dims: Tuple[int] = (224, 224),
    do_train: bool = False,
    epochs: int = 300,
    do_augment: bool = False, 
    channel_dim: int = 3,
):
    steps_per_epoch = np.floor(len(samples) / batch_size)

    image_gen = generator_fn(
        image_dir,
        samples,
        img_preprocessing=img_preprocessing,
        input_size=input_size,
        img_crop_dims=img_crop_dims,
        batch_size=batch_size,
        shuffle=do_train,
        do_augment=do_augment,
        channel_dim=channel_dim
    )
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def _ds_from_sequence(func, output_types, output_shapes):
        """
        Dataset From Sequence Class Eager Context

        Args:
            func ([type]): [description]
        """
        def _wrapper(batch_idx):
            # Use a tf.py_function to prevent auto-graph from compiling the method
            tensors = tf.py_function(func, inp=[batch_idx], Tout=output_types)
            # set the shape of the tensors - assuming channels last
            return [tensors[idx].set_shape(shape) for idx, shape in enumerate(output_shapes)]
        return _wrapper

    @_ds_from_sequence(output_types=output_types, output_shapes=output_shapes,)
    def train_batch_from_sequence(batch_idx):
        batch_idx = batch_idx.numpy()
        # zero-based index for what batch of data to load; 
        # i.e. goes to 0 at stepsPerEpoch and starts cound over
        zeroBatch = batch_idx % steps_per_epoch
        X_ref, X_dst, score = image_gen[zeroBatch]
        return X_ref, X_dst, score

    @_ds_from_sequence(output_types=output_types, output_shapes=output_shapes,)
    def valid_batch_from_sequence(batch_idx):
        batch_idx = batch_idx.numpy()
        # zero-based index for what batch of data to load;
        # i.e. goes to 0 at stepsPerEpoch and starts cound over
        zeroBatch = batch_idx % steps_per_epoch
        X_dst = image_gen[zeroBatch]
        return X_dst

    # create our data set for how many total steps of training we have
    dataset = tf.data.Dataset.range(steps_per_epoch * epochs)
    dataset.map(
        train_batch_from_sequence if do_train else valid_batch_from_sequence,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset, steps_per_epoch
