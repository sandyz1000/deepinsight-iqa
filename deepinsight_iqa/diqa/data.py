import os
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Tuple
from deepinsight_iqa.data_pipeline.diqa_gen.datagenerator import (
    TID2013DataRowParser,
    CSIQDataRowParser,
    LiveDataRowParser,
    AVADataRowParser,
    DiqaCombineDataGen,
    get_train_datagenerator
)
from deepinsight_iqa.data_pipeline.diqa_gen.tfdataset import get_tfdataset, get_tfds_v2
from deepinsight_iqa.common.utility import get_stream_handler
from typing import Callable
import logging


logger = logging.getLogger(__name__)
stdout_handler = get_stream_handler()
logger.addHandler(stdout_handler)
_DATAGEN_MAPPING = {
    "tid2013": TID2013DataRowParser,
    "csiq": CSIQDataRowParser,
    "live": LiveDataRowParser,
    "ava": AVADataRowParser
}


def get_iqa_tfds(
    image_dir: str,
    csv_path: str,
    dataset_type: str,
    image_preprocess: Callable = None,
    image_normalization: Callable = None,
    do_augment: bool = False,
    input_size: Tuple = (256, 256),
    batch_size: int = 8,
    channel_dim: int = 3,
    split_dataset: bool = True,
    split_prop: float = 0.7,
):
    if not Path(csv_path).exists():
        raise FileNotFoundError("Csv/Json file not found")

    if dataset_type in _DATAGEN_MAPPING:
        data_gen_cls = _DATAGEN_MAPPING[dataset_type]
        tf_dataset_func = get_tfds_v2
        output_types = (tf.float32, tf.float32, tf.float32),
        output_shapes = ([None, *input_size, channel_dim], [None, *input_size, channel_dim], [None]),
    else:
        data_gen_cls = get_train_datagenerator
        tf_dataset_func = get_tfdataset
        output_types = (tf.float32, ),
        output_shapes = ([None, *input_size, channel_dim]),

    df = pd.read_csv(csv_path)
    if split_dataset:
        samples_train = df.iloc[int(len(df) * split_prop):, ].to_numpy()  # type: np.ndarray
        samples_test = df.iloc[:int(len(df) * split_prop), ].to_numpy()  # type: np.ndarray
    else:
        samples_train = df.to_numpy()

    train_tfds, train_steps = tf_dataset_func(
        image_dir,
        samples_train,
        generator_fn=data_gen_cls,
        batch_size=batch_size,
        output_shapes=output_shapes,
        output_types=output_types,
        img_preprocessing=image_preprocess,
        image_normalization=image_normalization,
        do_augment=do_augment,
        input_size=input_size,
        channel_dim=channel_dim,
        do_train=True
    )

    valid_tfds, valid_steps = tf_dataset_func(
        image_dir,
        samples_test,
        generator_fn=data_gen_cls,
        batch_size=batch_size,
        output_shapes=output_shapes,
        output_types=output_types,
        img_preprocessing=image_preprocess,
        image_normalization=image_normalization,
        do_augment=False,
        input_size=input_size,
        channel_dim=channel_dim,
        do_train=False
    )

    logger.info(f"Train Step: {train_steps} -- Valid Steps: {valid_steps}")
    return train_tfds, valid_tfds


def get_iqa_datagen(
    image_dir: str,
    csv_path: str,
    dataset_type: str = None,
    image_preprocess: Callable = None,
    image_normalization: Callable = None,
    do_augment=False,
    input_size=None,
    batch_size=8,
    channel_dim=3,
    split_dataset=True,
    split_prop=0.7,
    **kwds
):
    if not Path(csv_path).exists():
        raise FileNotFoundError("Csv/Json file not found")

    if dataset_type in _DATAGEN_MAPPING:
        data_gen_cls = _DATAGEN_MAPPING[dataset_type]
    else:
        data_gen_cls = DiqaCombineDataGen

    df = pd.read_csv(csv_path)
    if split_dataset:
        samples_train = df.iloc[int(len(df) * split_prop):, ].to_numpy()  # type: np.ndarray
        samples_test = df.iloc[:int(len(df) * split_prop), ].to_numpy()  # type: np.ndarray
    else:
        samples_train = df.to_numpy()

    do_train = kwds.pop('do_train', False)

    train_datagen = data_gen_cls(
        image_dir,
        samples_train,
        batch_size=batch_size,
        img_preprocessing=image_preprocess,
        image_normalization=image_normalization,
        do_augment=do_augment,
        input_size=input_size,
        channel_dim=channel_dim,
        do_train=do_train,
        **kwds
    )

    valid_datagen = data_gen_cls(
        image_dir,
        samples_test,
        batch_size=batch_size,
        img_preprocessing=image_preprocess,
        image_normalization=image_normalization,
        do_augment=False,
        input_size=input_size,
        channel_dim=channel_dim,
        do_train=do_train,
        **kwds
    ) if split_dataset else None

    return train_datagen, valid_datagen


def save_json(data, target_file):
    import json
    with Path(target_file).open('w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
