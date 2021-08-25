import os
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import numpy as np
from abc import ABCMeta, abstractmethod
from six import add_metaclass
import tqdm


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class abstractstatic(staticmethod):
    __slots__ = ()

    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True


@add_metaclass(ABCMeta)
class TFRecordDataset:
    """ TF Record dataset that will be used to serialize and write object to file for quick streaming
    """

    @abstractmethod
    def _parse_tfrecord(self, tfrecord):
        pass

    @abstractmethod
    def _serialize_feat(*args, **kwargs):
        """ Create a Features message using tf.train.Example.

        :return: [description]
        :rtype: [type]
        """
        pass

    @abstractmethod
    def write_tfrecord_dataset(
            self, input_dir: str, csv_filename: str, *args, tfrecord_path=None, override=False, **kwrags):
        """ Convert raw input tfrecord dataset and save to locations

        :param input_dir: [description]
        :param csv_filename: CSV filename, where each row hold the reference, distorted and opinion score
        :param tfrecord_path: [description], defaults to None
        :type tfrecord_path: [type], optional
        """
        if not override and os.path.exists(tfrecord_path):
            raise RuntimeError("Cannot override data present in that location, set override=True to prepare tf-record")

    def load_tfrecord_dataset(self, file_pattern):
        assert os.path.exists(file_pattern), "TFRecord path doesn't exists"
        files = tf.data.Dataset.list_files(file_pattern)
        dataset = files.flat_map(tf.data.TFRecordDataset)
        return dataset.map(lambda x: self._parse_tfrecord(x))

    def split_tfrecord_files(self, file_pattern, batch_size=16):
        """ Split tf-records files to multiple tf-records file """
        outfile = os.path.basename(os.path.splitext(file_pattern))
        dirname = os.path.dirname(outfile)
        raw_dataset = self.load_tfrecord_dataset(file_pattern)
        batch_idx = 0
        for batch in raw_dataset.batch(batch_size):
            # Converting `batch` back into a `Dataset`, assuming batch is a `tuple` of `tensors`
            batch_ds = tf.data.Dataset.from_tensor_slices(tuple([*batch]))
            filename = os.path.join(dirname, 'tfchunks', f'{outfile}.tfrecord.{batch_idx:03d}')

            writer = tf.data.experimental.TFRecordWriter(filename)
            writer.write(batch_ds)
            batch_idx += 1


class AVARecordDataset(TFRecordDataset):
    DESCRIPTION = """
    Quality Assessment research strongly depends upon subjective experiments to provide calibration
    data as well as a testing mechanism. After all, the goal of all QA research is to make quality
    predictions that are in agreement with subjective opinion of human observers. In order to calibrate
    QA algorithms and test their performance, a data set of images and videos whose quality has been ranked by
    human subjects is required. The QA algorithm may be trained on part of this data set, and tested on the rest.

    **************************************************************************
    Content of AVA.txt
    **************************************************************************

    Column 1: Index

    Column 2: Image ID

    Columns 3 - 12: Counts of aesthetics ratings on a scale of 1-10. Column 3 
    has counts of ratings of 1 and column 12 has counts of ratings of 10.

    Columns 13 - 14: Semantic tag IDs. There are 66 IDs ranging from 1 to 66.
    The file tags.txt contains the textual tag corresponding to the numerical
    id. Each image has between 0 and 2 tags. Images with less than 2 tags have
    a "0" in place of the missing tag(s).

    Column 15: Challenge ID. The file challenges.txt contains the name of 
    the challenge corresponding to each ID.

    **************************************************************************
    Aesthetics image Lists
    **************************************************************************

    The aesthetics_image_lists directory contains files with the IDs of images
    used for training and testing generic aesthetics classifiers. There were:

    1. small scale (ss) experiments with few training images.
    2. large scale (ls) experiments with many training images.

    The directory also contains lists of training and testing images used for
    content (or category)-dependent classifiers. The categories are: animal,
    architecture, cityscape, floral, food/drink, landscape, portrait, and
    still-life.

    """
    SUPERVISED_KEYS = ("distorted_image", "dmos")
    FILE_EXT = "jpg"

    IMAGE_FEATURE_MAP = {
        "index": tf.io.FixedLenFeature((), tf.int64),
        "image": tf.io.FixedLenFeature((), tf.string),
        "mos": tf.io.FixedLenFeature((), tf.float32),
        "score_dist": tf.io.VarLenFeature(tf.string),
        "tags": tf.io.VarLenFeature(tf.string),
        "challenge": tf.io.FixedLenFeature((), tf.string),
    }

    def _serialize_feat(self, index, image, mos, score_dist, challenge, tags, *args, **kwargs):
        """ Create a Features message using tf.train.Example. """
        example_proto = tf.train.Example(features=tf.train.Features(feature={
            "index": _int64_feature(index),
            "image": _bytes_feature(image),
            "mos": _float_feature(mos),
            "score_dist": _bytes_feature(score_dist),
            "tags": _bytes_feature(tags),
            "challenge": _bytes_feature(challenge),
        }))
        return example_proto.SerializeToString()

    def _parse_tfrecord(self, tfrecord):
        """
        Parse single record and return image in PIL format
        :param tfrecord: TFRecord serialized string
        :type tfrecord: [type]
        :rtype: Tuple(PIL.Image.Image, float, string, string)
        """
        x = tf.io.parse_single_example(tfrecord, self.IMAGE_FEATURE_MAP)
        img = tf.io.parse_tensor(x['image'], out_type=tf.uint8)
        tags = tf.io.parse_tensor(x['tags'], out_type=tf.string)
        score_dist = tf.io.parse_tensor(x['score_dist'], out_type=tf.int32)
        mos = x['mos']
        challenge = x['challenge']
        return img, mos, score_dist, tags, challenge

    def write_tfrecord_dataset(
        self, input_dir, csv_filename,
        tfrecord_path=os.path.expanduser("~/tensorflow_dataset/ava/data.tf.records"),
        challenges_file='challenges.txt',
        tags_file='tags.txt', **kwargs
    ):
        super().write_tfrecord_dataset(input_dir, csv_filename, tfrecord_path=tfrecord_path, **kwargs)
        lines = [line.strip().split() for line in tf.io.gfile.GFile(input_dir + os.sep + csv_filename).readlines()]
        _init = tf.lookup.TextFileInitializer(
            filename=os.path.join(input_dir, challenges_file),
            key_dtype=tf.string, key_index=0,
            value_dtype=tf.string, value_index=1,
            delimiter=" "
        )
        challenges = tf.lookup.StaticHashTable(_init, default_value="")

        _init = tf.lookup.TextFileInitializer(
            filename=os.path.join(input_dir, tags_file),
            key_dtype=tf.string, key_index=0,
            value_dtype=tf.string, value_index=1,
            delimiter=" "
        )
        tags = tf.lookup.StaticHashTable(_init, default_value="")

        # TODO: Get folder from image_id and found in image_lists
        os.makedirs(os.path.dirname(tfrecord_path), exist_ok=True)
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for line in tqdm.tqdm(lines):
                index, image_path, mos, score_dist, linked_tags, challenge = (
                    int(line[0]),
                    f"{os.path.join(input_dir, 'images', line[1])}.jpg",
                    self.calc_mean_score([int(lab) for lab in line[2:12]]),
                    [int(lab) for lab in line[2:12]],
                    [tags.lookup(tf.cast(_id, tf.string)) for _id in line[12:14]],
                    challenges.lookup(tf.cast(line[14], tf.string))
                )
                try:
                    im_arr = tf.io.decode_image(tf.io.read_file(image_path))
                except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError):
                    continue
                # im_arr = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(image))
                # im_arr = tf.keras.preprocessing.image.random_zoom(im_arr, (0.5, 0.5),
                #                                                      row_axis=0,
                #                                                      col_axis=1,
                #                                                      channel_axis=2)
                example = self._serialize_feat(
                    index, tf.io.serialize_tensor(im_arr),
                    mos, tf.io.serialize_tensor(score_dist),
                    challenge, tf.io.serialize_tensor(linked_tags)
                )
                writer.write(example)

        return tfrecord_path

    def normalize_labels(self, labels):
        labels_np = np.array(labels)
        return labels_np / labels_np.sum()

    def calc_mean_score(self, score_dist) -> np.float32:
        # Expectation
        score_dist = self.normalize_labels(score_dist)
        return (score_dist * np.arange(1, 11)).sum()


class Tid2013RecordDataset(TFRecordDataset):
    DESCRIPTIONS = """
    The TID2013 contains 25 reference images and 3000 distorted images
    (25 reference images x 24 types of distortions x 5 levels of distortions).
    Reference images are obtained by cropping from Kodak Lossless True Color Image Suite.
    All images are saved in database in Bitmap format without any compression. File names are
    organized in such a manner that they indicate a number of the reference image,
    then a number of distortion's type, and, finally, a number of distortion's level: "iXX_YY_Z.bmp".
    """

    IMAGE_FEATURE_MAP = {
        "distorted_image": tf.io.FixedLenFeature((), tf.string),
        "reference_image": tf.io.FixedLenFeature((), tf.string),
        "mos": tf.io.FixedLenFeature((), tf.float32)
    }

    def _serialize_feat(self, distorted_image, reference_image, mos):
        """ Create a Features message using tf.train.Example. """
        example_proto = tf.train.Example(features=tf.train.Features(
            feature={
                "distorted_image": _bytes_feature(distorted_image),
                "reference_image": _bytes_feature(reference_image),
                "mos": _float_feature(float(mos))
            }))
        return example_proto.SerializeToString()

    def _parse_tfrecord(self, tfrecord):
        """
        Parse single record and return image in PIL format
        :param tfrecord: TFRecord serialized string
        :type tfrecord: [type]
        :rtype: Tuple(PIL.Image.Image, PIL.Image.Image, float)
        """
        IMAGE_FEATURE_MAP = {
            "distorted_image": tf.io.FixedLenFeature((), tf.string),
            "reference_image": tf.io.FixedLenFeature((), tf.string),
            "mos": tf.io.FixedLenFeature((), tf.float32)
        }
        x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
        distorted_image = tf.io.parse_tensor(x['distorted_image'], out_type=tf.uint8)
        reference_image = tf.io.parse_tensor(x['reference_image'], out_type=tf.uint8)
        return distorted_image, reference_image, x['mos']

    def write_tfrecord_dataset(
        self, input_dir, csv_filename,
        tfrecord_path=os.path.expanduser("~/tensorflow_dataset/tid2013/data.tf.records",),
        **kwargs
    ):
        # TODO: Split record and save into multiple chunks, move writer inside loop
        # https://stackoverflow.com/questions/54519309/split-tfrecords-file-into-many-tfrecords-files
        super().write_tfrecord_dataset(input_dir, csv_filename, tfrecord_path=tfrecord_path, **kwargs)
        lines = [line.strip().split(",") for line in tf.io.gfile.GFile(input_dir + os.sep + csv_filename).readlines()]
        os.makedirs(os.path.dirname(tfrecord_path), exist_ok=True)
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for index, line in tqdm.tqdm(enumerate(lines[1:])):
                mos = line[2]
                distorted_image = tf.io.decode_image(tf.io.read_file(os.path.join(input_dir, line[0])))
                reference_image = tf.io.decode_image(tf.io.read_file(os.path.join(input_dir, line[1])))
                example = self._serialize_feat(tf.io.serialize_tensor(distorted_image),
                                               tf.io.serialize_tensor(reference_image),
                                               mos)
                
                writer.write(example)
                
        return tfrecord_path


class CSIQRecordDataset(TFRecordDataset):
    DESCRIPTION = """
    Quality Assessment research strongly depends upon subjective experiments to provide calibration
    data as well as a testing mechanism. After all, the goal of all QA research is to make quality
    predictions that are in agreement with subjective opinion of human observers. In order to calibrate
    QA algorithms and test their performance, a data set of images and videos whose quality has been ranked by
    human subjects is required. The QA algorithm may be trained on part of this data set, and tested on the rest.

    Details
    Reference Images: Thirty reference images were obtained from public-domain sources (mostly from the U.S.
    National Park Service). The images were chosen to span five categories: Animals, Landscapes, People, Plants, Urban.

    Types of Distortions: The distortions used in CSIQ are: JPEG compression, JPEG-2000 compression, global
    contrast decrements, additive pink Gaussian noise, and Gaussian blurring. In total, there are 866 distorted images.

    Protocol: The CSIQ distorted images were subjectively rated base on a linear displacement of the images.
    Four Sceptre X24WG LCD monitors at resolution of 1920x1200 were calibrated to be as close as possible to the
    sRGB standard. The monitors were placed side-by-side with equal viewing distance to the subject. The subjects
    were instructed to keep a fixed viewing distance stable of approximately 70 cm.

    All of the distorted versions of each reference image were viewed simultaneously across the monitor array.
    Each subject horizontally positioned these images across the monitor array such that the horizontal distance
    between every pair of images reflected the difference in perceived quality between them. As a final step,
    across-image ratings were performed to obtain a "realignment" of the within-image ratings; this realignment
    experiment was a separate, but identical, experiment in which observers placed subsets of all the images linearly
    in space. The ratings were converted to z-scores, realigned, outliers removed, averaged across subjects, and then
    normalized to span the range [0, 1], where 1 denotes the lowest quality (largest perceived distortion).

    Overall the database contains 5000 subjective ratings and are reported in the form of DMOS. Thirty-five total
    subjects participated in this experiment, but each subject only viewed a subset of the images. The subject pool
    consisted of both males and females with normal or corrected-to-normal vision. The subjects' ages ranged from 21
    to 35./Volumes/ExtremeSSD/work-dataset/image-quality-assesement/tid2013/readme
    """
    SUPERVISED_KEYS = ("distorted_image", "dmos")
    IMAGE_FEATURE_MAP = {
        "index": tf.io.FixedLenFeature((), tf.int64),
        "distortion": tf.io.FixedLenFeature((), tf.string),
        "distorted_image": tf.io.FixedLenFeature((), tf.string),
        "reference_image": tf.io.FixedLenFeature((), tf.string),
        "dmos": tf.io.FixedLenFeature((), tf.float32),
        "dmos_std": tf.io.FixedLenFeature((), tf.float32)
    }

    def _serialize_feat(self, index, distortion, distorted_image, reference_image, dmos, dmos_std):
        """ Create a Features message using tf.train.Example. """
        example_proto = tf.train.Example(features=tf.train.Features(feature={
            "index": _int64_feature(int(index)),
            "distortion": _bytes_feature(bytes(distortion, 'utf-8')),
            "distorted_image": _bytes_feature(distorted_image),
            "reference_image": _bytes_feature(reference_image),
            "dmos": _float_feature(float(dmos)),
            "dmos_std": _float_feature(float(dmos_std))
        }))
        return example_proto.SerializeToString()

    def _parse_tfrecord(self, tfrecord):
        """
        Parse single record and return image in PIL format
        :param tfrecord: TFRecord serialized string
        :type tfrecord: [type]
        :rtype: Tuple(PIL.Image.Image, PIL.Image.Image, string, float)
        """
        x = tf.io.parse_single_example(tfrecord, self.IMAGE_FEATURE_MAP)
        distorted_image = tf.io.parse_tensor(x['distorted_image'], out_type=tf.uint8)
        reference_image = tf.io.parse_tensor(x['reference_image'], out_type=tf.uint8)
        return distorted_image, reference_image, x['distortion'], x['dmos'], x['dmos_std']

    def write_tfrecord_dataset(
        self, input_dir, csv_filename,
        tfrecord_path=os.path.expanduser("~/tensorflow_dataset/csiq/data.tf.records"),
        **kwargs
    ):
        super().write_tfrecord_dataset(input_dir, csv_filename, tfrecord_path=tfrecord_path, **kwargs)
        import re

        lines = [line.strip().split(",") for line in tf.io.gfile.GFile(input_dir + os.sep + csv_filename).readlines()]
        os.makedirs(os.path.dirname(tfrecord_path), exist_ok=True)
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for line in tqdm.tqdm(enumerate(lines[1:])):
                image, dst_idx, dst_type, dst_lev, dmos_std, dmos = line
                dst_type = re.sub(re.compile(r'\s+'), '', dst_type)
                dst_img_path = os.path.join(
                    input_dir, 'dst_imgs', dst_type, f"{image}.{dst_type}.{dst_lev}.png"
                )
                ref_img_path = os.path.join(input_dir, 'src_imgs', f"{image}.png")
                if all(os.path.exists(path) for path in [ref_img_path, dst_img_path]):
                    distorted_image = tf.io.decode_image(tf.io.read_file(dst_img_path))
                    reference_image = tf.io.decode_image(tf.io.read_file(ref_img_path))
                    example = self._serialize_feat(dst_idx, dst_type,
                                                   tf.io.serialize_tensor(distorted_image),
                                                   tf.io.serialize_tensor(reference_image),
                                                   dmos, dmos_std)
                    writer.write(example)

        return tfrecord_path


class LiveRecordDataset(TFRecordDataset):
    DESCRIPTION = """
    Quality Assessment research strongly depends upon subjective experiments to provide calibration
    data as well as a testing mechanism. After all, the goal of all QA research is to make quality
    predictions that are in agreement with subjective opinion of human observers. In order to calibrate
    QA algorithms and test their performance, a data set of images and videos whose quality has been ranked by
    human subjects is required. The QA algorithm may be trained on part of this data set, and tested on the rest.

    At LIVE (in collaboration with The Department of Psychology at the University of Texas at Austin),
    an extensive experiment was conducted to obtain scores from human subjects for a number of images
    distorted with different distortion types. These images were acquired in support of a research
    project on generic shape matching and recognition.
    """
    SUPERVISED_KEYS = ("distorted_image", "dmos")

    IMAGE_FEATURE_MAP = {
        "index": tf.io.FixedLenFeature((), tf.int64),
        "distortion": tf.io.FixedLenFeature((), tf.string),
        "distorted_image": tf.io.FixedLenFeature((), tf.string),
        "reference_image": tf.io.FixedLenFeature((), tf.string),
        "dmos": tf.io.FixedLenFeature((), tf.float32),
        "dmos_realigned": tf.io.FixedLenFeature((), tf.float32),
        "dmos_realigned_std": tf.io.FixedLenFeature((), tf.float32)
    }

    def _serialize_feat(self, index, distortion, distorted_image, reference_image, dmos,
                        dmos_realigned, dmos_realigned_std):
        """ Create a Features message using tf.train.Example. """
        example_proto = tf.train.Example(features=tf.train.Features(feature={
            "index": _int64_feature(int(index)),
            "distortion": _bytes_feature(bytes(distortion, 'utf-8')),
            "distorted_image": _bytes_feature(distorted_image),
            "reference_image": _bytes_feature(reference_image),
            "dmos": _float_feature(float(dmos)),
            "dmos_realigned": _float_feature(float(dmos_realigned)),
            "dmos_realigned_std": _float_feature(float(dmos_realigned_std))
        }))
        return example_proto.SerializeToString()

    def _parse_tfrecord(self, tfrecord: tf.data.TFRecordDataset):
        """
        Parse single record and return image in PIL format
        :param tfrecord: TFRecord serialized string
        :type tfrecord: [type]
        :rtype: Tuple(PIL.Image.Image, PIL.Image.Image, string, float, float, float)
        """
        x = tf.io.parse_single_example(tfrecord, LiveRecordDataset.IMAGE_FEATURE_MAP)
        distorted_image = tf.io.parse_tensor(x['distorted_image'], out_type=tf.uint8)
        reference_image = tf.io.parse_tensor(x['reference_image'], out_type=tf.uint8)
        return distorted_image, reference_image, x['distortion'], x['dmos'], \
            x['dmos_realigned'], x['dmos_realigned_std']

    def write_tfrecord_dataset(
        self,
        input_dir: str,
        csv_filename: str,
        tfrecord_path: str = os.path.expanduser("~/tensorflow_dataset/live/data.tf.records"),
        **kwargs
    ) -> tf.data.TFRecordDataset:
        super().write_tfrecord_dataset(input_dir, csv_filename, tfrecord_path=tfrecord_path, **kwargs)
        lines = [line.strip().split(",") for line in tf.io.gfile.GFile(input_dir + os.sep + csv_filename).readlines()]
        os.makedirs(os.path.dirname(tfrecord_path), exist_ok=True)
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for values in tqdm.tqdm(lines[1:]):
                distorted_image = tf.io.decode_image(tf.io.read_file(os.path.join(input_dir, values[2])))
                reference_image = tf.io.decode_image(tf.io.read_file(os.path.join(input_dir, values[3])))
                example = self._serialize_feat(values[1], values[0], tf.io.serialize_tensor(distorted_image),
                                               tf.io.serialize_tensor(reference_image), values[4], values[5],
                                               values[6])
                writer.write(example)

        return tfrecord_path
