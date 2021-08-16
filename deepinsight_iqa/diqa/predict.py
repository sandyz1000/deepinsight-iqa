import sys
import os
from typing import Optional
import tensorflow as tf
from .handlers.model import Diqa
from .utils.img_utils import image_preprocess
from time import perf_counter
import logging
import numpy as np
from typing import Union, Tuple
import cv2
import six
logger = logging.getLogger(__name__)
from deepinsight_iqa.common.utility import thread_safe_singleton

print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(
    __file__, __name__, str(__package__)))


@six.add_metaclass(thread_safe_singleton)
class Prediction:
    def __init__(
        self,
        model_dir: Optional[str] = None,
        final_wts_filename: Optional[str] = None,
        base_model_name: str = None,
        custom=False
    ):
        try:
            model_path = os.path.join(model_dir, base_model_name, final_wts_filename)
            self.diqa = Diqa(base_model_name, custom=custom)
            self.diqa._build()
            self.scoring_model = self.diqa.subjective
            self.scoring_model.load_weights(model_path)
        except Exception as e:
            print("Unable to load DIQA model, check model path", str(e))
            sys.exit(1)

    def predict(self, img: Union[np.ndarray, str]) -> float:
        assert img is not None, "Invalid path or image type"
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

