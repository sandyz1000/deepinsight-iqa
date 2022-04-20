import os
import pandas as pd
import numpy as np
import tensorflow as tf
from deepinsight_iqa.data_pipeline.diqa_gen import diqa_datagen
from deepinsight_iqa.common.utility import get_stream_handler
from typing import Callable
import logging


logger = logging.getLogger(__name__)
stdout_handler = get_stream_handler()
logger.addHandler(stdout_handler)
_DATAGEN_MAPPING = {
    "tid2013": diqa_datagen.TID2013DataRowParser,
    "csiq": diqa_datagen.CSIQDataRowParser,
    "live": diqa_datagen.LiveDataRowParser,
    "ava": diqa_datagen.AVADataRowParser
}


def get_iqa_tfds(
    image_dir: str,
    csv_path: str,
    dataset_type: str,
    image_preprocess: Callable = None,
    do_augment=False,
    input_size=(256, 256), batch_size=8, channel_dim=3
):
    if dataset_type is None:
        assert os.path.exists(csv_path), FileNotFoundError("Csv/Json file not found")
        df = pd.read_csv(csv_path)
        samples_train, samples_test = (
            df.iloc[:int(len(df) * 0.7), ].to_numpy(),
            df.iloc[int(len(df) * 0.7):, ].to_numpy()
        )

        train_tfds, train_steps = diqa_datagen.get_tfdataset(
            image_dir,
            samples_train,
            generator_fn=diqa_datagen.get_train_datagenerator,
            batch_size=batch_size,
            img_preprocessing=image_preprocess,
            do_augment=do_augment,
            input_size=input_size,
            channel_dim=channel_dim,
            shuffle=True,
            is_training=True,
        )

        valid_tfds, valid_steps = diqa_datagen.get_tfdataset(
            image_dir,
            [dist for _, dist, ref, mos in samples_test],
            generator_fn=diqa_datagen.get_eval_datagenerator,
            batch_size=batch_size,
            img_preprocessing=image_preprocess,
            do_augment=do_augment,
            input_size=input_size,
            channel_dim=channel_dim,
            is_training=False,
        )
    else:
        assert dataset_type in _DATAGEN_MAPPING.keys(), "Invalid dataset_type, unable to use generator"
        assert os.path.splitext(csv_path)[-1] == '.csv' and os.path.exists(csv_path), \
            FileNotFoundError("Csv/Json file not found or not a valid file ext")

        df = pd.read_csv(csv_path)
        samples_train, samples_test = df.iloc[:int(len(df) * 0.7), ], df.iloc[int(len(df) * 0.7):, ]

        data_gen_cls = _DATAGEN_MAPPING[dataset_type]

        train_tfds, train_steps = diqa_datagen.get_tfds_v2(
            image_dir,
            samples_train,
            generator_fn=data_gen_cls,
            output_types=(tf.float32, tf.float32, tf.float32),
            output_shapes=([None, *input_size, channel_dim], [None, *input_size, channel_dim], [None]),
            batch_size=batch_size,
            img_preprocessing=image_preprocess,
            do_augment=do_augment,
            input_size=input_size,
            channel_dim=channel_dim,
            shuffle=True,
        )

        valid_tfds, valid_steps = diqa_datagen.get_tfds_v2(
            image_dir,
            [dist for _, dist, ref, mos in samples_test],
            generator_fn=data_gen_cls,
            output_types=(tf.float32, ),
            output_shapes=([None, *input_size, channel_dim]),
            batch_size=batch_size,
            img_preprocessing=image_preprocess,
            do_augment=do_augment,
            input_size=input_size,
            channel_dim=channel_dim,
            shuffle=False,
        )

    logger.info(f"Train Step: {train_steps} -- Valid Steps: {valid_steps}")
    return train_tfds, valid_tfds


def get_iqa_datagen(
    image_dir: str,
    csv_path: str,
    dataset_type: str = None,
    image_preprocess: Callable = None,
    do_augment=False,
    input_size=(256, 256),
    batch_size=8,
    channel_dim=3,
    **kwds
):
    if dataset_type is None:
        assert os.path.exists(csv_path), \
            FileNotFoundError("Csv/Json file not found")
        
        df = pd.read_csv(csv_path)
        samples_train = df.iloc[int(len(df) * 0.7):, ].to_numpy()  # type: np.ndarray
        samples_test = df.iloc[:int(len(df) * 0.7), ].to_numpy()  # type: np.ndarray
        
        train_datagen = diqa_datagen.DiqaCombineDataGen(
            image_dir,
            samples_train,
            batch_size=batch_size,
            img_preprocessing=image_preprocess,
            do_augment=do_augment,
            input_size=input_size,
            channel_dim=channel_dim,
            shuffle=True
        )

        valid_datagen = diqa_datagen.DiqaCombineDataGen(
            image_dir,
            [dist for _, dist, ref, mos in samples_test],
            batch_size=batch_size,
            img_preprocessing=image_preprocess,
            do_augment=do_augment,
            input_size=input_size,
            channel_dim=channel_dim,
            shuffle=False
        )
    else:
        assert os.path.splitext(csv_path)[-1] == '.csv' and os.path.exists(csv_path), \
            FileNotFoundError("Csv/Json file not found or not a valid file ext")

        df = pd.read_csv(csv_path)
        samples_train, samples_test = df.iloc[:int(len(df) * 0.7), ], df.iloc[int(len(df) * 0.7):, ]

        data_gen_cls = _DATAGEN_MAPPING[dataset_type]

        train_datagen = data_gen_cls(
            image_dir,
            samples_train,
            batch_size=batch_size,
            img_preprocessing=image_preprocess,
            do_augment=do_augment,
            input_size=input_size,
            channel_dim=channel_dim,
            shuffle=True,
        )

        valid_datagen = data_gen_cls(
            image_dir,
            [dist for _, dist, ref, mos in samples_test],
            batch_size=batch_size,
            img_preprocessing=image_preprocess,
            do_augment=do_augment,
            input_size=input_size,
            channel_dim=channel_dim,
            shuffle=False,
        )

    return train_datagen, valid_datagen
