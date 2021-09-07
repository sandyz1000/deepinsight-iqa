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


def get_combine_datagen(
    image_dir: str, csv_path: str,
    image_preprocess: Callable = None,
    do_augment=False,
    batch_size=8, **kwargs
):
    assert os.path.exists(csv_path), FileNotFoundError("Csv/Json file not found") 
    df = pd.read_csv(csv_path)
    samples_train, samples_test = df.iloc[:int(len(df) * 0.7), ].to_numpy(), df.iloc[int(len(df) * 0.7):, ].to_numpy()

    train_tfdataset, train_steps = diqa_datagen.get_tfdataset(
        image_dir, samples_train,
        generator_fn=diqa_datagen.get_deepiqa_datagenerator,
        batch_size=batch_size,
        img_preprocessing=image_preprocess,
        do_augment=do_augment,
        **kwargs
    )

    valid_tfdataset, valid_steps = diqa_datagen.get_tfdataset(
        image_dir, samples_test,
        generator_fn=diqa_datagen.get_deepiqa_datagenerator,
        batch_size=batch_size,
        img_preprocessing=image_preprocess,
        do_augment=do_augment,
        **kwargs
    )
    logger.info(f"Train Step: {train_steps} -- Valid Steps: {valid_steps}")
    return train_tfdataset, valid_tfdataset


def get_iqa_datagen(
    image_dir: str, csv_path: str, dataset_type: str,
    image_preprocess: Callable = None,
    do_augment=False,
    batch_size=8, **kwargs
):
    _DATAGEN_MAPPING = {
        "tid2013": diqa_datagen.TID2013DataRowParser,
        "csiq": diqa_datagen.CSIQDataRowParser,
        "live": diqa_datagen.LiveDataRowParser,
        "ava": diqa_datagen.AVADataRowParser
    }

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
        shuffle=True,
        **kwargs
    )

    valid_tfds, valid_steps = diqa_datagen.get_tfdataset(
        image_dir,
        samples_test,
        generator_fn=data_gen_cls,
        batch_size=batch_size,
        img_preprocessing=image_preprocess,
        do_augment=do_augment,
        shuffle=False,
        **kwargs
    )
    logger.info(f"Train Step: {train_steps} -- Valid Steps: {valid_steps}")
    return train_tfds, valid_tfds
