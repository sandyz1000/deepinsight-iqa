import sys
import os
from typing import Optional
import tensorflow as tf
from .networks.model import Diqa
from .utils.tf_imgutils import image_preprocess
from time import perf_counter
import logging
import numpy as np
from typing import Union, Tuple
import cv2
import six
from skimage import io
import glob
from deepinsight_iqa.data_pipeline.diqa_gen.diqa_datagen import DiqaCombineDataGen

logger = logging.getLogger(__name__)
from deepinsight_iqa.common.utility import thread_safe_singleton

print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(
    __file__, __name__, str(__package__)))


def image_dir_to_json(img_dir: str, img_type: str = 'jpg'):
    img_paths = glob.glob(os.path.join(img_dir, '*.' + img_type))

    samples = []
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        img_id = img_name.split('.')[0]
        samples.append({'image_id': img_id, "path": img_name})

    return samples


@six.add_metaclass(thread_safe_singleton)
class Prediction:
    def __init__(
        self,
        model_dir: Optional[str] = None,
        subjective_weightfname: Optional[str] = None,
        base_model_name: str = None,
        custom=False
    ):
        try:
            model_path = os.path.join(model_dir, base_model_name, subjective_weightfname)
            self.diqa = Diqa(base_model_name, custom=custom)
            self.diqa._build()
            self.scoring_model = self.diqa.subjective
            self.scoring_model.load_weights(model_path)
        except Exception as e:
            print("Unable to load DIQA model, check model path", str(e))
            sys.exit(1)

    def predict(self, img: Union[np.ndarray, str]) -> float:
        if isinstance(img, str) and not os.path.exists(img):
            raise FileNotFoundError("Invalid path or image type")
        
        if isinstance(img, str):
            img = cv2.imread(img, cv2.IMREAD_UNCHANGED)

        logger.info("Predicting the final score for IQA")
        img = tf.cast(img, dtype=tf.float32)

        start = perf_counter()
        I_d = image_preprocess(img)
        end = perf_counter()
        logger.debug(f"IQA preprocessing function image_preprocess took {end-start} seconds")

        start = perf_counter()
        I_d = tf.tile(I_d, (1, 1, 1, 3))
        prediction = self.scoring_model.predict(I_d)[0][0]
        end = perf_counter()

        logger.debug(f"Keras model took {end-start} seconds to predict the iqa score")
        logger.info(f"final IQA score is {prediction}")
        return prediction

    def predict_batch(self, img_dir, batch_size=3,):
        start = perf_counter()
        samples = image_dir_to_json(img_dir)
        images = [row['path'] for row in samples]
        X_data = DiqaCombineDataGen(
            image_dir=img_dir, samples=images, batch_size=batch_size, img_preprocessing=image_preprocess
        )

        predictions = self.scoring_model.predict_generator(X_data, workers=1, use_multiprocessing=False, verbose=1)
        for i, sample in enumerate(samples):
            sample['score'] = predictions[i]
        end = perf_counter()
        logger.debug(f"Predictions on batch of {batch_size} took {end-start} seconds")
        return samples
