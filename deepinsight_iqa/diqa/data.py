import os
import sys
from deepinsight_iqa.data_pipeline.diqa_gen import diqa_datagen
from typing import Callable
import pandas as pd
import logging
logger = logging.getLogger(__name__)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(logging.Formatter(
    "%(asctime)s %(name)-12s %(levelname)-8s %(message)s", "%Y-%m-%dT%H:%M:%S"
))
logger.addHandler(stdout_handler)
_DATAGEN_MAPPING = {
    "tid2013": diqa_datagen.TID2013DataRowParser,
    "csiq": diqa_datagen.CSIQDataRowParser,
    "live": diqa_datagen.LiveDataRowParser,
    "ava": diqa_datagen.AVADataRowParser
}


def get_iqa_combined_datagen(
    image_dir: str, csv_path: str,
    image_preprocess: Callable = None,
    do_augment=False, input_size=(256, 256), batch_size=8, channel_dim=3
):
    assert os.path.exists(csv_path), FileNotFoundError("Csv/Json file not found")
    df = pd.read_csv(csv_path)
    samples_train, samples_test = df.iloc[:int(len(df) * 0.7), ].to_numpy(), df.iloc[int(len(df) * 0.7):, ].to_numpy()
    train_datagen = diqa_datagen.get_deepiqa_datagenerator(
        image_dir,
        samples_train,
        batch_size=batch_size,
        img_preprocessing=image_preprocess,
        do_augment=do_augment,
        input_size=input_size,
        channel_dim=channel_dim,
        shuffle=True, repeat=True
    )

    valid_datagen = diqa_datagen.get_deepiqa_datagenerator(
        image_dir,
        samples_test,
        batch_size=batch_size,
        img_preprocessing=image_preprocess,
        do_augment=do_augment,
        input_size=input_size,
        channel_dim=channel_dim,
        shuffle=False, repeat=True
    )

    return train_datagen, valid_datagen


def get_iqa_combine_tfds(
    image_dir: str, csv_path: str,
    image_preprocess: Callable = None,
    do_augment=False, input_size=(256, 256), batch_size=8, channel_dim=3
):
    assert os.path.exists(csv_path), FileNotFoundError("Csv/Json file not found")
    df = pd.read_csv(csv_path)
    samples_train, samples_test = df.iloc[:int(len(df) * 0.7), ].to_numpy(), df.iloc[int(len(df) * 0.7):, ].to_numpy()

    train_tfdataset, train_steps = diqa_datagen.get_tfdataset(
        image_dir,
        samples_train,
        generator_fn=diqa_datagen.get_deepiqa_datagenerator,
        batch_size=batch_size,
        img_preprocessing=image_preprocess,
        do_augment=do_augment,
        input_size=input_size,
        channel_dim=channel_dim,
        shuffle=True,
        is_training=True,
    )

    valid_tfdataset, valid_steps = diqa_datagen.get_tfdataset(
        image_dir,
        samples_test,
        generator_fn=diqa_datagen.get_deepiqa_datagenerator,
        batch_size=batch_size,
        img_preprocessing=image_preprocess,
        do_augment=do_augment,
        input_size=input_size,
        channel_dim=channel_dim,
        is_training=False,
    )
    logger.info(f"Train Step: {train_steps} -- Valid Steps: {valid_steps}")
    return train_tfdataset, valid_tfdataset


def get_iqa_tfds(
    image_dir: str, csv_path: str, dataset_type: str,
    image_preprocess: Callable = None,
    do_augment=False, input_size=(256, 256), batch_size=8, channel_dim=3
):

    assert dataset_type in _DATAGEN_MAPPING.keys(), "Invalid dataset_type, unable to use generator"
    assert os.path.splitext(csv_path)[-1] == '.csv' and os.path.exists(csv_path), \
        FileNotFoundError("Csv/Json file not found or not a valid file ext")

    df = pd.read_csv(csv_path)
    samples_train, samples_test = df.iloc[:int(len(df) * 0.7), ], df.iloc[int(len(df) * 0.7):, ]

    data_gen_cls = _DATAGEN_MAPPING[dataset_type]

    train_tfds, train_steps = diqa_datagen.get_tfdataset(
        image_dir,
        samples_train,
        generator_fn=data_gen_cls,
        batch_size=batch_size,
        img_preprocessing=image_preprocess,
        do_augment=do_augment,
        input_size=input_size,
        channel_dim=channel_dim,
        shuffle=True,
    )

    valid_tfds, valid_steps = diqa_datagen.get_tfdataset(
        image_dir,
        samples_test,
        generator_fn=data_gen_cls,
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
    image_dir: str, csv_path: str, dataset_type: str,
    image_preprocess: Callable = None,
    do_augment=False, input_size=(256, 256), batch_size=8, channel_dim=3
):

    assert dataset_type in _DATAGEN_MAPPING.keys(), "Invalid dataset_type, unable to use generator"
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
        samples_test,
        batch_size=batch_size,
        img_preprocessing=image_preprocess,
        do_augment=do_augment,
        input_size=input_size,
        channel_dim=channel_dim,
        shuffle=False,
    )

    return train_datagen, valid_datagen