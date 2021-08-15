import itertools
from typing import Tuple, Callable, Dict, List
import pandas as pd
from deepinsight_iqa.common import image_aug
import tensorflow as tf
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from six import add_metaclass
from abc import ABCMeta, abstractmethod
import os
IMG_EXT = "jpg"
_AVA_OUTYPE = Tuple[Tuple[np.ndarray, str, str], float, List[int]]


def load_image(img_file, target_size=None):
    return np.asarray(tf.keras.preprocessing.image.load_img(img_file, target_size=target_size))


def read_image(filename: str, **kwargs) -> tf.Tensor:
    stream = tf.io.read_file(filename)
    return tf.image.decode_image(stream, **kwargs)


def _augmentation(img, rand_crop_dims=[200, 200]):
    sequential = [
        # (lambda img: image_aug.random_crop(img, rand_crop_dims)),
        image_aug.random_horizontal_flip,
        image_aug.random_vertical_flip
    ]
    for _func in sequential:
        img = _func(img)
    return img


class InvalidParserError(Exception):
    """Raise parser error incase of any exception

    Arguments:
        Exception {[type]} -- [description]
    """


@add_metaclass(ABCMeta)
class DiqaDataGenerator(tf.keras.utils.Sequence):
    '''inherits from Keras Sequence base object, allows to use multiprocessing in .fit_generator'''

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        batch_size: int = 32,
        img_preprocessing: Callable = None,
        input_size: Tuple[int] = (256, 256),
        img_crop_dims: Tuple[int] = (224, 224),
        shuffle: bool = False,
        check_dir_availability=True,
        do_augment=False,
        channel_dim=3,
    ):
        self.channel_dim = channel_dim
        self.do_augment = do_augment
        self.shuffle = shuffle
        df = df.sample(frac=1)
        self.samples = df.to_numpy()
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_preprocessing = img_preprocessing
        self.input_size = input_size  # dimensions that images get resized into when loaded
        self.img_crop_dims = img_crop_dims  # dimensions that images get randomly cropped to
        self.data_generator = self.__train_datagen__ if shuffle else self.__eval_datagen__
        if check_dir_availability:
            self._validate()  # All Image location not avaliable for the given dataset type
        self.on_epoch_end()  # call ensures that samples are shuffled in first epoch if shuffle is set to True

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))  # number of batches per epoch

    @abstractmethod
    def data_parser(self, row: Dict):
        """ Parse CSV ROW
        :param row: dataframe rows
        :type row: pd.Dataframe
        """
        pass

    @abstractmethod
    def _validate(self):
        """
        Check all files and folder exists for the given dataset,
        this will do a basic sanity check before generating the dataset
        """
        pass

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]  # get batch indexes
        batch_samples = [self.samples[i] for i in batch_indexes]  # get batch samples
        X_dist, X_ref, y = self.data_generator(batch_samples)
        return X_dist, X_ref, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __train_datagen__(self, batch_samples):

        X_dist, X_ref, mos = [], [], []
        for i, sample in enumerate(batch_samples):
            # load and randomly augment image
            i_d, i_r, label = self.data_parser(*sample)
            if i_d is None or i_r is None:
                continue
            distorted_image, reference_image = (load_image(im) for im in [i_d, i_r])
            if reference_image is None or distorted_image is None:
                continue
            if self.img_preprocessing:
                distorted_image, reference_image = [
                    tf.squeeze(self.img_preprocessing(im), axis=0)
                    for im in [distorted_image, reference_image]
                ]

            if self.channel_dim == 3:
                distorted_image, reference_image = [
                    tf.tile(im, (1, 1, self.channel_dim))
                    for im in [distorted_image, reference_image]
                ]

            if self.do_augment:
                distorted_image, reference_image = [
                    # tf.convert_to_tensor(image_aug.augment_img(im, augmentation_name='geometric'))
                    _augmentation(im, rand_crop_dims=self.input_size)
                    for im in [distorted_image, reference_image]
                ]

            X_dist.append(distorted_image)
            X_ref.append(reference_image)
            mos.append(label)
        X_dist, X_ref, mos = [tf.cast(dty, dtype=tf.float32) for dty in [X_dist, X_ref, mos]]
        return X_dist, X_ref, mos

    def __eval_datagen__(self, batch_samples):
        """ initialize images and labels tensors for faster processing """

        X_dist, X_ref, Y = [], [], []
        for i, sample in enumerate(batch_samples):
            i_d, i_r, label = self.data_parser(*sample)
            distorted_image, reference_image = (load_image(im, target_size=self.input_size) for im in [i_d, i_r])
            if self.img_preprocessing:
                distorted_image, reference_image = [
                    tf.squeeze(self.img_preprocessing(im), axis=0)
                    for im in [distorted_image, reference_image]
                ]
            if self.channel_dim == 3:
                distorted_image, reference_image = [
                    tf.tile(im, (1, 1, self.channel_dim))
                    for im in [distorted_image, reference_image]
                ]
            X_dist.append(distorted_image)
            X_ref.append(reference_image)
            Y.append(label)
        
        X_dist, X_ref, Y = [tf.cast(dty, dtype=tf.float32) for dty in [X_dist, X_ref, Y]]
        return X_dist, X_ref, Y


class LiveDataRowParser(DiqaDataGenerator):
    def data_parser(self, *row: Tuple):
        """
        Function parse LIVE csv and return image features and label from csv row

        Arguments:
            df {[type]} -- Dataframes that has ref and distorted image and mos/dmos score
            img_dir {[type]} -- Directory that contains both image
        """
        distortion, index, distorted_path, reference_path, dmos, dmos_realigned, dmos_realigned_std = row
        I_d, I_r, label = os.path.join(self.img_dir, distorted_path), \
            os.path.join(self.img_dir, reference_path), dmos
        return I_d, I_r, label

    def _validate(self) -> bool:
        _DISTPATH = ['fastfading', 'jpeg', 'jp2k', 'refimgs']
        assert all(os.path.exists(self.img_dir + os.sep + _p) for _p in _DISTPATH), \
            "Image location not avaliable for LIVE dataset"
        return True


class TID2013DataRowParser(DiqaDataGenerator):
    def data_parser(self, *row: Tuple):
        """Generator function parse TID2013 csv and return image features and label from csv row

        Arguments:
            df {[type]} -- Dataframes that has ref and distorted image and mos/dmos score
            img_dir {[type]} -- Directory that contains both image
        """
        distorted_path, reference_path, mos = row
        I_d, I_r, label = os.path.join(self.img_dir, distorted_path), \
            os.path.join(self.img_dir, reference_path), mos

        return I_d, I_r, label

    def _validate(self) -> bool:
        # "Image location not avaliable for TID2013 dataset"
        return all(os.path.exists(self.img_dir + os.sep + _p) for _p in [
            "distorted_images", "reference_images"
        ])


class CSIQDataRowParser(DiqaDataGenerator):
    dst_to_dir = {
        "blur": "blur",
        "awgn": "awgn",
        "contrast": "contrast",
        "noise": "fnoise",
        "jpeg 2000": "jpeg2000",
        "jpeg": "jpeg"
    }

    def data_parser(self, *row: Tuple):
        """Generator function parse CSIQ csv and return image features and label from csv row

        Arguments:
            df {[type]} -- Dataframes that has ref and distorted image and mos/dmos score
            img_dir {[type]} -- Directory that contains both image
        """

        image, dst_idx, dst_type, dst_lev, dmos_std, dmos = row

        I_r = os.path.join(
            self.img_dir, "src_imgs", f"{image}.png"
        )
        I_d = os.path.join(
            self.img_dir, 'dst_imgs', dst_type, f"{image}.{dst_type}.{dst_lev}.png"
        )
        label = dmos
        return I_d, I_r, label

    def _validate(self) -> bool:
        # "Image location not avaliable for CSIQ dataset" in case of error
        return all(os.path.exists(self.img_dir + os.sep + _p) for _p in [
            "src_imgs", "dst_imgs/blur", "dst_imgs/awgn", "dst_imgs/contrast",
            "dst_imgs/fnoise", "dst_imgs/jpeg2000", "dst_imgs/jpeg"
        ])


class AVADataRowParser(DiqaDataGenerator):
    def __init__(
            self,
            ava_txt: str,
            img_dir: str,
            challenges_file='challenges.txt',
            tags_file='tags.txt',
            batch_size: int = 32,
            img_preprocessing: Callable = None,
            target_size: Tuple[int] = (256, 256),
            img_crop_dims: Tuple[int] = (224, 224),
            shuffle: bool = False,
            do_augment=False,
    ):
        self.do_augment = do_augment
        self.img_dir = img_dir

        lines = [line.strip().split() for line in tf.io.gfile.GFile(img_dir + os.sep + ava_txt).readlines()]
        df = pd.DataFrame.from_records(lines)

        _init = tf.lookup.TextFileInitializer(
            filename=os.path.join(img_dir, challenges_file),
            key_dtype=tf.string, key_index=0,
            value_dtype=tf.string, value_index=1,
            delimiter=" "
        )
        self.challenges = tf.lookup.StaticHashTable(_init, default_value="")

        _init = tf.lookup.TextFileInitializer(
            filename=os.path.join(img_dir, tags_file),
            key_dtype=tf.string, key_index=0,
            value_dtype=tf.string, value_index=1,
            delimiter=" "
        )
        self.tags = tf.lookup.StaticHashTable(_init, default_value="")

        super(AVADataRowParser, self).__init__(
            df, img_dir,
            batch_size=batch_size,
            img_preprocessing=img_preprocessing,
            input_size=target_size,
            img_crop_dims=img_crop_dims,
            shuffle=shuffle, do_augment=do_augment
        )

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]  # get batch indexes
        batch_samples = [self.samples[i] for i in batch_indexes]  # get batch samples
        X, y, dist = self.data_generator(batch_samples)
        return X, y, dist

    def normalize_labels(self, labels):
        labels_np = np.array(labels)
        return labels_np / labels_np.sum()

    def calc_mean_score(self, score_dist) -> np.float32:
        # Expectation
        score_dist = self.normalize_labels(score_dist)
        return (score_dist * np.arange(1, 11)).sum()

    def data_parser(self, *row: Tuple):
        """Parse AVA Dataset from the csv row

        :param row: [description]
        :type row: [type]
        """
        idx, image_path, mos, score_dist, linked_tags, challenge = (
            int(row[0]),
            f"{os.path.join(self.img_dir, 'images', row[1])}.jpg",
            self.calc_mean_score([int(lab) for lab in row[2:12]]),
            [int(lab) for lab in row[2:12]],
            [self.tags.lookup(tf.cast(_id, tf.string)) for _id in row[12:14]],
            self.challenges.lookup(tf.cast(row[14], tf.string))
        )
        return image_path, mos, score_dist, linked_tags, challenge

    def _validate(self) -> bool:
        return os.path.exists(os.path.join(self.img_dir, "images"))

    def __train_datagen__(self, batch_samples) -> _AVA_OUTYPE:
        features, mos_scores, distributions = [], [], []
        for i, sample in enumerate(batch_samples):
            # load and randomly augment image
            image_path, mos, scoredis, linked_tags, challenge = self.data_parser(*sample)
            img = read_image(image_path, channels=3)
            if img is None:
                continue
            if self.img_preprocessing:
                img = self.img_preprocessing(img)
            if self.do_augment:
                img = _augmentation(img)

            features.append([img, challenge, "|".join(linked_tags)])
            mos_scores.append(mos)
            distributions.append(scoredis)
        
        return features, mos_scores, distributions

    def __eval_datagen__(self, batch_samples):
        """ initialize images and labels tensors for faster processing """
        features, mos_scores, distributions = [], []
        for i, sample in enumerate(batch_samples):
            image_path, mos, scoredis, linked_tags, challenge = self.data_parser(*sample)
            img = read_image(image_path, channels=3)
            if self.img_preprocessing:
                img = self.img_preprocessing(img)
            features.append([img, challenge, "|".join(linked_tags)])
            mos_scores.append(mos)
            distributions.append(scoredis)

        return features, mos_scores, distributions


def get_deepiqa_datagenerator(
    image_dir: str,
    samples: np.ndarray,
    batch_size: int = 32,
    img_preprocessing: Callable = None,
    input_size: Tuple[int] = (256, 256),
    img_crop_dims: Tuple[int] = (224, 224),
    shuffle: bool = False, do_augment: bool = False, channel_dim: int = 3,
):
    """
    Generator that will generate image for AVA, TID2013 and CSIQ dataset
    """
    if shuffle:
        np.random.shuffle(samples)

    zipped = itertools.cycle(samples)
    # zipped = iter(samples)
    # apply_aug = (lambda im: image_aug.augment_img(im, augmentation_name='geometric'))
    apply_aug = _augmentation
    while True:
        X_dist = []
        X_ref = []
        Y = []

        for _ in range(batch_size):
            row = next(zipped)
            _, i_d, i_r, mos_score = row
            i_d, i_r = [load_image(os.path.join(image_dir, im), target_size=input_size) for im in [i_d, i_r]]

            if img_preprocessing:
                i_d, i_r = [
                    tf.squeeze(img_preprocessing(im), axis=0).numpy()
                    for im in [i_d, i_r]
                ]
            if channel_dim == 3:
                i_d, i_r = [
                    tf.tile(im, (1, 1, channel_dim))
                    for im in [i_d, i_r]
                ]
            if do_augment:
                i_d, i_r = [apply_aug(im, rand_crop_dims=img_crop_dims) for im in [i_d, i_r]]

            X_dist.append(i_d)
            X_ref.append(i_r)
            Y.append(mos_score)

        X_dist, X_ref, Y = [tf.cast(dty, dtype=tf.float32) for dty in [X_dist, X_ref, Y]]
        yield X_dist, X_ref, Y


def combine_deepiqa_dataset(data_dir: str, csvspathmap: Dict[str, str], output_csv: str) -> None:
    """ Combine all csv to single csv file that can be used by the data-generator


    """
    from functools import partial

    def _tid2013_data_parser(img_dir, *row):
        """
        Higher value of MOS (0 - minimal, 9 - maximal) corresponds to higher visual
        quality of the image.

        Rescale range to the range [0, 1], where 0 denotes the lowest quality (largest perceived distortion).
        """
        distorted_image, reference_image, mos = row
        return (
            os.path.join(img_dir, distorted_image),
            os.path.join(img_dir, reference_image),
            mos / 10
        )

    def _csiq_data_parser(img_dir, *row):
        """ The ratings were converted to z-scores, realigned, outliers removed, averaged across subjects,
        and then normalized to span the range [0, 1], where 1 denotes the lowest quality (largest perceived distortion).

        Subtract dmos from 1 to convert the score to common scale i.e. 0 denotes the lowest quality and vice-versa
        """
        image, dst_idx, dst_type, dst_lev, dmos_std, dmos = row
        dst_type = "".join(dst_type.split())
        dst_img_path = os.path.join(
            img_dir, 'dst_imgs', dst_type, f"{image}.{dst_type}.{dst_lev}.png"
        )
        ref_img_path = os.path.join(img_dir, 'src_imgs', f"{image}.png")

        return dst_img_path, ref_img_path, 1 - dmos

    def _liveiqa_data_parser(img_dir, *row: Tuple):
        """
        Difference Mean Opinion Score (DMOS) value for each distorted image:
        The raw scores for each subject is the difference scores (between the test and the reference) 
        and then Z-scores and then scaled and shifted to the full range (1 to 100).

        Rescale range to the range [0, 1] and subtract from 1 to covert it to common scale,
        where 0 denotes the lowest quality (largest perceived distortion) and vice-versa
        """
        distortion, index, distorted_path, reference_path, dmos, dmos_realigned, dmos_realigned_std = row
        return (
            os.path.join(img_dir, distorted_path),
            os.path.join(img_dir, reference_path),
            1 - (dmos / 100)
        )

    _FUNC_MAPPING = {
        "tid2013": _tid2013_data_parser,
        "csiq": _csiq_data_parser,
        "live": _liveiqa_data_parser
    }

    assert set(csvspathmap.keys()) == set(_FUNC_MAPPING.keys()), "Invalid csvpath to function map"
    cols = ['index', 'distorted_image', 'reference_image', 'mos']
    dataset = pd.DataFrame(columns=cols)
    for dataset_name, csvpath in csvspathmap.items():
        # csv_name = os.path.basename(csvpath)
        folder_name = os.path.dirname(csvpath)
        ddf = pd.read_csv(os.path.join(data_dir, csvpath))
        funcparser = partial(_FUNC_MAPPING[dataset_name], folder_name)
        current = pd.DataFrame([funcparser(*row) for idx, row in ddf.iterrows()], columns=cols)
        dataset = dataset.append(current, ignore_index=True)
    output_csv = os.path.join(data_dir, output_csv)
    dataset.to_csv(output_csv)
