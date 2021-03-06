import itertools
import random
from typing import Tuple, Callable, Dict, List, Union
import pandas as pd
from deepinsight_iqa.common import image_aug
from deepinsight_iqa.diqa.networks.utils import calculate_error_map
import tensorflow as tf
import numpy as np
from functools import partial
from six import add_metaclass
from abc import ABCMeta, abstractmethod
import os
from functools import partial
IMG_EXT = "jpg"
_AVA_OUTYPE = Tuple[Tuple[np.ndarray, str, str], float, List[int]]


def load_image(img_file, target_size=None):
    """Imahe utility to load from image-path

    :param _type_ img_file: _description_
    :param _type_ target_size: `load_img` Can accept optional arguments None, 
    defaults to None

    :return np.ndarray: N-d tensor
    """
    pilimg = tf.keras.preprocessing.image.load_img(img_file, target_size=target_size)
    return tf.keras.preprocessing.image.img_to_array(pilimg)


def read_image(filename: str, **kwargs) -> tf.Tensor:
    stream = tf.io.read_file(filename)
    return tf.image.decode_image(stream, **kwargs)


def _augmentation(img1, img2, rand_crop_dims=[416, 416], random_crop=False, geometric_augment=False):
    
    augmenter = [image_aug.horizontal_flip, image_aug.vertical_flip]

    if random_crop:
        random_func = partial(image_aug.random_crop, rand_crop_dims)
        augmenter.append(random_func)

    if geometric_augment:
        geometric_func = partial(image_aug.augment_img, augmentation_name='geometric')
        augmenter.append(geometric_func)
    
    random.seed(42)
    fn_idx = random.randint(0, len(augmenter) - 1)
    img1 = augmenter[fn_idx](img1)
    img2 = augmenter[fn_idx](img2)
    return img1, img2


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
        img_dir: str,
        df: pd.DataFrame,
        batch_size: int = 32,
        img_preprocessing: Callable = None,
        image_normalization: Callable = None,
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
        self.image_normalization = image_normalization
        self.input_size = input_size  # dimensions that images get resized into when loaded
        self.img_crop_dims = img_crop_dims  # dimensions that images get randomly cropped to
        self.data_generator = self.__train_datagen__ if shuffle else self.__eval_datagen__
        self.steps_per_epoch = np.floor(len(self.samples) / self.batch_size)
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
        return self.data_generator(batch_samples)

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
            if (
                (reference_image is None or distorted_image is None) or
                (reference_image.shape != distorted_image.shape)
            ):
                continue

            if self.img_preprocessing:
                distorted_image, reference_image = [
                    tf.squeeze(self.img_preprocessing(im), axis=0)
                    for im in [distorted_image, reference_image]
                ]

            distorted_image, reference_image = [
                tf.tile(im, (1, 1, self.channel_dim)) if self.channel_dim == 3 and im.get_shape()[-1] != 3 else im
                for im in [distorted_image, reference_image]
            ]

            if self.do_augment:
                distorted_image, reference_image = [
                    # image_aug.augment_img(im, augmentation_name='geometric'),
                    _augmentation(im, rand_crop_dims=self.input_size)
                    for im in [distorted_image, reference_image]
                ]

            X_dist.append(distorted_image)
            X_ref.append(reference_image)
            mos.append(label)
        
        X_dist, X_ref, mos = (tf.cast(dty, dtype=tf.float32) for dty in [X_dist, X_ref, mos])
        
        X_ref = tf.slice(X_ref, begin=[0, 0, 0, 0], size=X_ref.shape[:-1] + [1])
        dist_gray = tf.slice(X_dist, begin=[0, 0, 0, 0], size=X_dist.shape[:-1] + [1])
        return [X_dist, dist_gray, X_ref], mos

    def __eval_datagen__(self, batch_samples):
        """ initialize images and labels tensors for faster processing """

        X_dist, X_ref, mos = [], [], []
        for i, sample in enumerate(batch_samples):
            i_d, i_r, label = self.data_parser(*sample)
            distorted_image, reference_image = (load_image(im, target_size=self.input_size) for im in [i_d, i_r])
            if self.img_preprocessing:
                distorted_image, reference_image = [
                    tf.squeeze(self.img_preprocessing(im), axis=0)
                    for im in [distorted_image, reference_image]
                ]

            distorted_image, reference_image = [
                tf.tile(im, (1, 1, self.channel_dim)) if self.channel_dim == 3 and im.get_shape()[-1] != 3 else im
                for im in [distorted_image, reference_image]
            ]
            X_dist.append(distorted_image)
            X_ref.append(reference_image)
            mos.append(label)

        X_dist, X_ref, mos = (tf.cast(dty, dtype=tf.float32) for dty in [X_dist, X_ref, mos])
        
        X_ref = tf.slice(X_ref, begin=[0, 0, 0, 0], size=X_ref.shape[:-1] + [1])
        dist_gray = tf.slice(X_dist, begin=[0, 0, 0, 0], size=X_dist.shape[:-1] + [1])
        return [X_dist, dist_gray, X_ref], mos


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
            img_dir: str,
            ava_txt: str,
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
            df,
            img_dir,
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


class DiqaCombineDataGen(tf.keras.utils.Sequence):
    def __init__(
        self,
        image_dir: str,
        samples: np.ndarray,
        batch_size: int = 32,
        img_preprocessing: Callable = None,
        image_normalization: Callable = None,
        input_size: Tuple[int] = (256, 256),
        img_crop_dims: Tuple[int] = (224, 224),
        do_train: bool = False,
        do_augment: bool = False,
        channel_dim: int = 3,
        scaling_factor: float = 1 / 38
    ) -> None:
        """ Predict Generator that will generate batch of images from a folder

        # NOTE: If do_train == False and do_augment == False, it will return eval generator
        Args:
            image_dir ([type]): [description]
            samples ([type]): [description]
            batch_size ([type], optional): [description]. Defaults to 32.
            img_preprocessing ([type], optional): [description]. Defaults to None.
            input_size ([type], optional): [description]. Defaults to (256, 256).
            channel_dim ([type], optional): [description]. Defaults to 3.
        """

        self.do_train = do_train
        self.samples = samples
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_preprocessing = img_preprocessing
        self.image_normalization = image_normalization
        self.input_size = input_size
        self.channel_dim = channel_dim
        self.img_crop_dims = img_crop_dims
        self.do_augment = do_augment
        self.scaling_factor = scaling_factor
        self.data_generator = self.__train_generator if self.do_train else self.__batch_generator
        self.steps_per_epoch = int(np.floor(len(self.samples) / self.batch_size))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))  # number of batches per epoch

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.samples))
        if self.do_train:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]  # get batch indexes
        batch_samples = [self.samples[i] for i in batch_indexes]  # get batch samples
        return self.data_generator(batch_samples)

    def __train_generator(self, batch_samples):
        X_dist = []
        X_ref = []
        mos = []

        for row in batch_samples:
            _, dist_img, ref_img, mos_score = row
            dist_img, ref_img = [
                load_image(os.path.join(self.image_dir, im), target_size=self.input_size) 
                for im in [dist_img, ref_img]
            ]
            
            if ref_img is None or dist_img is None:
                continue

            if self.img_preprocessing:
                dist_img, ref_img = [
                    tf.squeeze(self.img_preprocessing(im), axis=0)
                    for im in [dist_img, ref_img]
                ]
            
            if self.image_normalization:
                dist_img = self.image_normalization(dist_img)
                ref_img = self.image_normalization(ref_img)

            dist_img, ref_img = [
                tf.tile(im, (1, 1, self.channel_dim))
                if self.channel_dim == 3 and im.get_shape()[-1] != 3 else im
                for im in [dist_img, ref_img]
            ]
            if self.do_augment:
                dist_img, ref_img = _augmentation(dist_img, ref_img, rand_crop_dims=self.img_crop_dims, random_crop=False)

            if ref_img.shape != dist_img.shape:
                continue
            
            X_dist.append(dist_img)
            X_ref.append(ref_img)
            mos.append(mos_score)

        X_dist, X_ref, mos = [tf.cast(dty, dtype=tf.float32) for dty in [X_dist, X_ref, mos]]
        
        X_ref = tf.slice(X_ref, begin=[0, 0, 0, 0], size=X_ref.shape[:-1] + [1])
        dist_gray = tf.slice(X_dist, begin=[0, 0, 0, 0], size=X_dist.shape[:-1] + [1])
        e_gt, r = calculate_error_map(dist_gray, X_ref, scaling_factor=self.scaling_factor)
        if self.image_normalization:
            e_gt, r = [self.image_normalization(im) for im in [e_gt, r]]
        
        return [X_dist, X_ref, e_gt, r], mos

    def __batch_generator(self, batch_samples):
        X = []
        for dist_im in batch_samples:
            dist_im = load_image(os.path.join(self.image_dir, dist_im), target_size=self.input_size)
            if dist_im is None:
                continue

            if self.img_preprocessing:
                dist_im = tf.squeeze(self.img_preprocessing(dist_im), axis=0)

            if self.image_normalization:
                dist_im = self.image_normalization(dist_im)

            dist_im = tf.tile(
                dist_im, (1, 1, self.channel_dim)
            ) if self.channel_dim == 3 and dist_im.get_shape()[-1] != 3 else dist_im
            X.append(dist_im)

        X = tf.cast(X, dtype=tf.float32)
        return X


class get_train_datagenerator:
    def __init__(
        self,
        image_dir: str,
        samples: np.ndarray,
        batch_size: int = 32,
        img_preprocessing: Callable = None,
        input_size: Tuple[int] = (256, 256),
        img_crop_dims: Tuple[int] = (224, 224),
        do_train: bool = False,
        do_augment: bool = False,
        channel_dim: int = 3,
        repeat: bool = False,
    ):
        """
        Generator that will generate shuffle image for AVA, TID2013 and CSIQ dataset combined

        Args:
            image_dir ([type]): [description]
            samples ([type]): [description]
            batch_size ([type], optional): [description]. Defaults to 32.
            img_preprocessing ([type], optional): [description]. Defaults to None.
            input_size ([type], optional): [description]. Defaults to (256, 256).
            img_crop_dims ([type], optional): [description]. Defaults to (224, 224).
            shuffle ([type], optional): [description]. Defaults to False.
            do_augment ([type], optional): [description]. Defaults to False.
            channel_dim ([type], optional): [description]. Defaults to 3.

        Yields:
            [type]: [description]
        """
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.channel_dim = channel_dim
        self.input_size = input_size
        self.do_augment = do_augment
        self.img_preprocessing = img_preprocessing
        self.img_crop_dims = img_crop_dims

        if do_train:
            np.random.shuffle(samples)
        self.zipped = itertools.cycle(samples) if repeat else iter(samples)

    def __iter__(self):
        return self

    def __next__(self):
        X_dist = []
        X_ref = []
        Y = []

        for _ in range(self.batch_size):
            try:
                row = next(self.zipped)
                _, dist_img, ref_img, mos_score = row
                dist_img, ref_img = [
                    load_image(os.path.join(self.image_dir, im), target_size=self.input_size)
                    for im in [dist_img, ref_img]
                ]

                if (
                    not ref_img or not dist_img or
                    ref_img.shape != dist_img.shape
                ):
                    continue

                if self.img_preprocessing:
                    dist_img, ref_img = [
                        tf.squeeze(self.img_preprocessing(im), axis=0)
                        for im in [dist_img, ref_img]
                    ]

                dist_img, ref_img = [
                    tf.tile(im, (1, 1, self.channel_dim))
                    if self.channel_dim == 3 and im.get_shape()[-1] != 3 else im
                    for im in [dist_img, ref_img]
                ]
                if self.do_augment:
                    dist_img, ref_img = _augmentation(dist_img, ref_img, rand_crop_dims=self.img_crop_dims)

                X_dist.append(dist_img)
                X_ref.append(ref_img)
                Y.append(mos_score)
            except StopIteration:
                break

        X_dist, X_ref, Y = [tf.cast(dty, dtype=tf.float32) for dty in [X_dist, X_ref, Y]]
        X_ref = tf.slice(X_ref, begin=[0, 0, 0, 0], size=X_ref.shape[:-1] + [1])
        dist_gray = tf.slice(X_dist, begin=[0, 0, 0, 0], size=X_dist.shape[:-1] + [1])
        e_gt, r = calculate_error_map(dist_gray, X_ref, scaling_factor=self.scaling_factor)
        if self.image_normalization:
            e_gt, r = [self.image_normalization(im) for im in [e_gt, r]]
        
        return [X_dist, X_ref, e_gt, r], Y


class get_batch_datagenerator:
    def __init__(
        self,
        image_dir: str,
        samples: np.ndarray,
        batch_size: int = 32,
        img_preprocessing: Callable = None,
        input_size: Tuple[int] = (256, 256),
        channel_dim: int = 3,
    ):
        """
        Generator that will generate image for evaluation

        Args:
            image_dir ([type]): [description]
            samples ([type]): [description]
            batch_size ([type], optional): [description]. Defaults to 32.
            img_preprocessing ([type], optional): [description]. Defaults to None.
            input_size ([type], optional): [description]. Defaults to (256, 256).
            channel_dim ([type], optional): [description]. Defaults to 3.

        Yields:
            [type]: [description]
        """
        self.image_dir = image_dir
        self.img_preprocessing = img_preprocessing
        self.steps_per_epoch = np.floor(len(samples) / batch_size)
        self.zipped = itertools.cycle(samples)
        self.batch_size = batch_size
        self.channel_dim = channel_dim
        self.input_size = input_size

    def __iter__(self):
        return self

    def __next__(self):
        X_dist = []
        for _ in range(self.batch_size):
            try:
                dist_img = next(self.zipped)
                dist_img = load_image(os.path.join(self.image_dir, dist_img), target_size=self.input_size)
                if dist_img is None:
                    continue
                
                if self.img_preprocessing:
                    dist_img = tf.squeeze(self.img_preprocessing(dist_img), axis=0)

                dist_img = tf.tile(dist_img, (1, 1, self.channel_dim)) \
                    if self.channel_dim == 3 and dist_img.get_shape()[-1] != 3 else dist_img

                X_dist.append(dist_img)
            except StopIteration:
                break

        return tf.cast(X_dist, dtype=tf.float32)
