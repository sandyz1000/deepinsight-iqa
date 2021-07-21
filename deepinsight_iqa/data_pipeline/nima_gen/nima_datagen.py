import tensorflow as tf
import numpy as np
import os
import random
import itertools
from pixar_common import image_aug
from tensorflow import keras
from keras.applications.imagenet_utils import preprocess_input
import cv2
from PIL import Image
import json


def load_image(img_file, target_size):
    return np.asarray(tf.keras.preprocessing.image.load_img(img_file, target_size=target_size))


def normalize_labels(labels):
    labels_np = np.array(labels)
    return labels_np / labels_np.sum()


def calc_mean_score(score_dist):
    score_dist = normalize_labels(score_dist)
    return (score_dist * np.arange(1, 11)).sum()


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def img_read_n_resize(img_path, target_size=(112, 112), rescale=1. / 255.):
    assert len(target_size) == 2, "Invalid target size format for resizing the img"
    img = np.array(Image.open(img_path), dtype=np.uint8)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA) \
        if target_size and len(target_size) == 2 else img
    # From BGR to RGB format
    img = img[..., ::-1] * rescale
    return img


class NimaDataGenerator(keras.utils.Sequence):
    '''inherits from Keras Sequence base object, allows to use multiprocessing in .fit_generator'''

    def __init__(self,
                 samples, img_dir, batch_size, n_classes, preprocess_func, img_format,
                 target_size=(256, 256), img_crop_dims=(224, 224), shuffle=False):
        self.samples = samples
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.preprocess_func = preprocess_func  # Keras basenet specific preprocessing function
        self.img_load_dims = target_size  # dimensions that images get resized into when loaded
        self.img_crop_dims = img_crop_dims  # dimensions that images get randomly cropped to
        self.shuffle = shuffle
        self.img_format = img_format
        self.__data_generator__ = self.__train_generator__ if shuffle else self.__eval_generator__
        self.on_epoch_end()  # call ensures that samples are shuffled in first epoch if shuffle is set to True

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))  # number of batches per epoch

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]  # get batch indexes
        batch_samples = [self.samples[i] for i in batch_indexes]  # get batch samples
        X, y = self.__data_generator__(batch_samples)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.samples))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __train_generator__(self, batch_samples):
        # initialize images and labels tensors for faster processing
        X = np.empty((len(batch_samples), *self.img_crop_dims, 3))
        y = np.empty((len(batch_samples), self.n_classes))

        for i, sample in enumerate(batch_samples):
            # load and randomly augment image
            img_file = os.path.join(self.img_dir, '{}.{}'.format(sample['image_id'], self.img_format))
            img = load_image(img_file, self.img_load_dims)
            if img is not None:
                img = self.random_crop(img, self.img_crop_dims)
                img = self.random_horizontal_flip(img)
                img = self.random_vertical_flip(img)
                X[i, ] = img

            # normalize labels
            y[i, ] = normalize_labels(sample['label'])

        # apply basenet specific preprocessing
        # input is 4D numpy array of RGB values within [0, 255]
        X = self.preprocess_func(X)

        return X, y

    def __eval_generator__(self, batch_samples):
        # initialize images and labels tensors for faster processing
        X = np.empty((len(batch_samples), *self.img_load_dims, 3))
        y = np.empty((len(batch_samples), self.n_classes))

        for i, sample in enumerate(batch_samples):
            # load and randomly augment image
            img_file = os.path.join(self.img_dir, '{}.{}'.format(sample['image_id'], self.img_format))
            img = load_image(img_file, self.img_load_dims)
            if img is not None:
                X[i, ] = img

            # normalize labels
            if sample.get('label') is not None:
                y[i, ] = normalize_labels(sample['label'])

        # apply basenet specific preprocessing
        # input is 4D numpy array of RGB values within [0, 255]
        X = self.preprocess_func(X)


def get_train_dataset(pairs_txt: str, img_dir_path: str, generator_fn: tf.keras.utils.Sequence,
                      img_shape=(160, 160, 3), is_train=True, batch_size=64):
    """
    Function to convert Keras Sequence to tensorflow dataset

    Arguments:
        pairs_txt {[type]} -- [description]
        img_dir_path {[type]} -- [description]
        generator_fn {[type]} -- [description]

    Keyword Arguments:
        img_shape {[type]} -- [description] (default: {(160, 160, 3)})
        is_train {[type]} -- [description] (default: {True})
        batch_size {[type]} -- [description] (default: {64})

    Returns:
        [type] -- [description]
    """
    image_gen = generator_fn(
        pairs_txt,
        img_dir_path,
        batch_size=batch_size,
        horizontal_rotation=True,
        preprocess_func=preprocess_input,
    )

    classes = image_gen.nb_classes
    steps_per_epoch = np.floor(len(image_gen.pairs) / batch_size)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = tf.data.Dataset.from_generator(
        image_gen,
        output_types=(tf.float32, tf.int32),
        output_shapes=([None, 3, *img_shape], [None, 1])
    )

    if is_train:
        train_ds = train_ds.repeat()

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    return train_ds, steps_per_epoch, classes


def get_nima_datagenerator(samples,
                           img_dir,
                           batch_size,
                           preprocess_func=preprocess_input,
                           target_size=(256, 256),
                           img_crop_dims=(224, 224),
                           normalize_labels=False,
                           do_augment=False,
                           shuffle=False,):
    if shuffle:
        random.shuffle(samples)
    zipped = itertools.cycle(samples)

    while True:
        X = []
        Y = []

        for _ in range(batch_size):
            row = next(zipped)
            img_path, label = os.path.join(img_dir, row['image_id']), normalize_labels(row['label'])
            im1 = np.array(load_image(img_path, target_size), dtype=np.uint8)

            if do_augment:
                im1 = image_aug.augment_img(im1, augmentation_name='geometric')

            X.append(im1)
            Y.append(label)

        X, Y = preprocess_func(np.array(X)), np.array(Y)

        yield X, Y
