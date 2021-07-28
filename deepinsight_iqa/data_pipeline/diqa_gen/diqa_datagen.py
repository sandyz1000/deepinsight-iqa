import itertools
from typing import Tuple, Callable, Dict
import pandas as pd
from deepinsight_iqa.common import image_aug
import tensorflow as tf
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from six import add_metaclass
from abc import ABCMeta, abstractmethod
import os
IMG_EXT = "jpg"


def load_image(img_file, target_size):
    return np.asarray(tf.keras.preprocessing.image.load_img(img_file, target_size=target_size))


def read_image(filename: str, **kwargs) -> tf.Tensor:
    stream = tf.io.read_file(filename)
    return tf.image.decode_image(stream, **kwargs)


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
        target_size: Tuple[int] = (256, 256),
        img_crop_dims: Tuple[int] = (224, 224),
        shuffle: bool = False,
        check_dir_availability=True,
    ):
        self.shuffle = shuffle
        df = df.sample(frac=1)
        self.samples = df.to_records()
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_preprocessing = img_preprocessing
        self.target_size = target_size  # dimensions that images get resized into when loaded
        self.img_crop_dims = img_crop_dims  # dimensions that images get randomly cropped to
        self.__data_generator = self.__train_datagen__ if shuffle else self.__eval_datagen__
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
        X, y = self.__data_generator(batch_samples)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _augmentation(self, img, rand_crop_dims=200):
        sequential = [
            (lambda img: image_aug.random_crop(img, rand_crop_dims)),
            image_aug.random_horizontal_flip, image_aug.random_vertical_flip
        ]
        # TODO: Apply augmentation based on paramenter supplied by the user
        for _func in sequential:
            img = _func(img)
        return img

    def __train_datagen__(self, batch_samples):

        dist_images, ref_images, labels = [], [], []
        for i, sample in enumerate(batch_samples):
            # load and randomly augment image
            i_d, i_r, label = self.data_parser(sample)
            distorted_image = self.img_preprocessing(read_image(i_d, channels=len(self.target_size)))
            reference_image = self.img_preprocessing(read_image(i_r, channels=len(self.target_size)))

            if reference_image is None or distorted_image is None:
                continue

            reference_image = self._augmentation(reference_image)
            distorted_image = self._augmentation(distorted_image)

            dist_images.append(distorted_image)
            ref_images.append(reference_image)
            labels.append(label)

        return dist_images, ref_images, labels

    def __eval_datagen__(self, batch_samples):
        """ initialize images and labels tensors for faster processing """

        features, labels = [], []
        for i, sample in enumerate(batch_samples):
            i_d, i_r, label = self.data_parser(sample)
            img = self.img_preprocessing(read_image(i_d, channels=len(self.target_size)))
            features.append(img)
            labels.append(label)

        return features, labels


class LiveDataRowParser(DiqaDataGenerator):
    def data_parser(self, row: Dict):
        """
        Function parse LIVE csv and return image features and label from csv row

        Arguments:
            df {[type]} -- Dataframes that has ref and distorted image and mos/dmos score
            img_dir {[type]} -- Directory that contains both image
        """
        I_d, I_r, label = os.path.join(self.img_dir, row['distorted_path']), \
            os.path.join(self.img_dir, row['reference_images']), row['dmos']
        return I_d, I_r, label

    def _validate(self) -> bool:
        _DISTPATH = ['fastfading', 'jpeg', 'jp2k', 'refimgs']
        assert all(os.path.exists(self.img_dir + os.sep + _p) for _p in _DISTPATH), \
            "Image location not avaliable for LIVE dataset"
        return True


class TID2013DataRowParser(DiqaDataGenerator):
    def data_parser(self, row: Dict):
        """Generator function parse TID2013 csv and return image features and label from csv row

        Arguments:
            df {[type]} -- Dataframes that has ref and distorted image and mos/dmos score
            img_dir {[type]} -- Directory that contains both image
        """
        I_d, I_r, label = os.path.join(self.img_dir, row['distorted_path']), \
            os.path.join(self.img_dir, row['reference_images']), row['mos']

        return I_d, I_r, label

    def _validate(self) -> bool:
        # "Image location not avaliable for TID2013 dataset"
        return all(os.path.exists(self.img_dir + os.sep + _p) for _p in [
            "distorted_images", "reference_images"
        ])


class CSIQDataRowParser(DiqaDataGenerator):
    IMG_EXT = "png"

    def data_parser(self, row: Dict):
        """Generator function parse CSIQ csv and return image features and label from csv row

        Arguments:
            df {[type]} -- Dataframes that has ref and distorted image and mos/dmos score
            img_dir {[type]} -- Directory that contains both image
        """
        I_d = os.path.join(
            self.img_dir, "dst_imgs",
            f"{row['image']}.{row['dst_type'].upper()}.{row['dst_idx']}.{self.IMG_EXT}"
        )
        I_r = os.path.join(
            self.img_dir, "src_imgs", f"{row['image']}.{self.IMG_EXT}"
        )
        label = row['dmos']
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
            shuffle: bool = False):

        lines = tf.io.gfile.GFile(ava_txt).readlines().split()
        # TODO: Fix below dataframe code

        df = pd.DataFrame.from_records(lines)
        self.challenges = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
            os.path.join(img_dir, challenges_file), tf.int64, 0, tf.string, 1, delimiter=" "), -1)

        self.tags = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
            os.path.join(img_dir, tags_file), tf.int64, 0, tf.string, 1, delimiter=" "), -1)

        super(AVADataRowParser, self).__init__(df, img_dir,
                                               batch_size=batch_size,
                                               img_preprocessing=img_preprocessing,
                                               target_size=target_size,
                                               img_crop_dims=img_crop_dims, shuffle=shuffle)

    def normalize_labels(self, labels):
        labels_np = np.array(labels)
        return labels_np / labels_np.sum()

    def calc_mean_score(self, score_dist) -> np.float32:
        # Expectation
        score_dist = self.normalize_labels(score_dist)
        return (score_dist * np.arange(1, 11)).sum()

    def data_parser(self, row: Dict):
        """Parse AVA Dataset from the csv row

        :param row: [description]
        :type row: [type]
        """
        return os.path.join(self.img_dir, row['image']), row['mos'], row['tags'], row["challenge"]

    def _validate(self) -> bool:
        return all(os.path.exists(os.path.join(self.img_dir, p)) for p in [
            "dataset/test", "dataset/train"
        ])

    def __train_datagen__(self, batch_samples):

        features, mos_scores, imgtags, challanges = [], [], [], []
        for i, sample in enumerate(batch_samples):
            # load and randomly augment image
            img_path, mos, tags, challange = self.data_parser(sample)
            img = self.img_preprocessing(read_image(img_path, channels=len(self.target_size)))
            if img is None:
                continue
            img = self._augmentation(img)

            features.append(img)
            mos_scores.append(mos)
            imgtags.append(tags)
            challanges.append(challange)

        return features, mos_scores, imgtags, challanges

    def __eval_datagen__(self, batch_samples):
        """ initialize images and labels tensors for faster processing """
        features, labels = [], []
        for i, sample in enumerate(batch_samples):
            i_d, i_r, label = self.data_parser(sample)
            img = self.img_preprocessing(read_image(i_d, channels=len(self.target_size)))
            features.append(img)
            labels.append(label)

        return features, labels


def get_deepiqa_datagenerator(
    df: pd.DataFrame,
    img_dir: str,
    batch_size: int = 32,
    img_preprocessing: Callable = None,
    target_size: Tuple[int] = (256, 256),
    img_crop_dims: Tuple[int] = (224, 224),
    dataset_type: str = "tid2013",
    shuffle: bool = False, do_augment: bool = False
):
    """
    Generator that will generate image for AVA, TID2013 and CSIQ dataset
    """
    from functools import partial

    def _tid2013_data_parser(row):
        return (
            os.path.join(img_dir, row['distorted_image']),
            os.path.join(img_dir, row['reference_image']),
            row['mos']
        )

    def _csiq_data_parser(row):
        image, dst_idx, dst_type, dst_lev, dmos_std, dmos = row.keys()
        dst_type = "".join(dst_type.split())
        dst_img_path = os.path.join(
            img_dir, 'dst_imgs', dst_type, f"{image}.{dst_type}.{dst_lev}.png"
        )
        ref_img_path = os.path.join(img_dir, 'src_imgs', f"{image}.png")
        if not os.path.exists(os.path.basename(dst_img_path)):
            return (None, None, None)

        return dst_img_path, ref_img_path, dmos

    def _liveiqa_data_parser(row):
        return (
            os.path.join(img_dir, row['distorted_image']),
            os.path.join(img_dir, row['reference_image']),
            row['dmos']
        )

    _FUNC_MAPPING = {
        "tid2013": _tid2013_data_parser,
        "csiq": _csiq_data_parser,
        "liva": _liveiqa_data_parser
    }

    assert dataset_type in _FUNC_MAPPING.keys(), "Invalid dataset type"
    data_parser = partial(_FUNC_MAPPING[dataset_type])
    if shuffle:
        df = df.sample(frac=1)
    samples = df.to_records()
    zipped = itertools.cycle(samples)
    apply_aug = (lambda im: image_aug.augment_img(im, augmentation_name='geometric'))
    while True:
        X = []
        Y = []

        for _ in range(batch_size):
            row = next(zipped)
            i_d, i_r, mos_score = data_parser(row)
            im1 = img_preprocessing(load_image(i_d, target_size))
            im2 = img_preprocessing(load_image(i_r, target_size))

            if do_augment:
                im1, im2 = [apply_aug(im) for im in [im1, im2]]

            X.append((im1, im2))
            Y.append(mos_score)

        yield X, Y
