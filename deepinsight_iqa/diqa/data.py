import os
from deepinsight_iqa.data_pipeline.diqa_gen import diqa_datagen
from typing import Callable
import pandas as pd


def get_combine_datagen(
    image_dir: str, csv_path: str,
    image_preprocess: Callable = None,
    do_augment=False,
    batch_size=8, **kwargs
):
    df = pd.read_csv(os.path.join(image_dir, csv_path))
    samples_train, samples_test = df.iloc[:int(len(df) * 0.7), ].to_numpy(), df.iloc[int(len(df) * 0.7):, ].to_numpy()

    train_generator = diqa_datagen.get_deepiqa_datagenerator(
        image_dir, samples_train,
        batch_size=batch_size,
        img_preprocessing=image_preprocess,
        do_augment=do_augment,
        **kwargs
    )

    valid_generator = diqa_datagen.get_deepiqa_datagenerator(
        image_dir, samples_test,
        batch_size=batch_size,
        img_preprocessing=image_preprocess,
        do_augment=do_augment,
        **kwargs
    )

    return train_generator, valid_generator


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
    csv_path = os.path.join(image_dir, csv_path)
    assert os.path.splitext(csv_path)[-1] == '.csv' and os.path.exists(csv_path), \
        "Not a valid file extension"

    df = pd.read_csv(csv_path)
    samples_train, samples_test = df.iloc[:int(len(df) * 0.7), ], df.iloc[int(len(df) * 0.7):, ]

    data_gen_cls = _DATAGEN_MAPPING[dataset_type]

    train_generator = data_gen_cls(
        samples_train,
        image_dir,
        batch_size,
        img_preprocessing=image_preprocess,
        do_augment=do_augment,
        shuffle=True,
        **kwargs
    )

    valid_generator = data_gen_cls(
        samples_test,
        image_dir,
        batch_size,
        img_preprocessing=image_preprocess,
        do_augment=do_augment,
        shuffle=False,
        **kwargs
    )

    return train_generator, valid_generator
