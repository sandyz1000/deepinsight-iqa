import sys
import os
print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(
    __file__, __name__, str(__package__)))
    
import tensorflow as tf
from .model import Diqa
from .utils import image_preprocess
from time import perf_counter
import logging
import numpy as np
import typing
import cv2
logger = logging.getLogger(__name__)
from .utils import thread_safe_memoize


@thread_safe_memoize
def init_model(model_path):
    try:
        diqa = Diqa()
        scoring_model = diqa.subjective_score_model
        # model_path = os.path.join(BASE_DIR, 'weights/diqa/', SUBJECTIVE_MODEL_NAME)
        scoring_model.load_weights(model_path)
        return scoring_model
    except Exception as e:
        print("Unable to load DIQA model, check model path", str(e))
        sys.exit(1)


def show_sample_prediction(scoring_model, reference_image, distorted_image):
    """
    Return the score for the reference and distorted image, which can be later use for 
    comparing with the target mos
    Args:
        reference_image ([type]): [description]
        distorted_image ([type]): [description]

    Returns:
        [type]: [description]
    """
    I_d = image_preprocess(distorted_image)
    I_r = image_preprocess(reference_image)
    dist_prediction = scoring_model.predict(I_d)[0][0]
    ref_prediction = scoring_model.predict(I_r)[0][0]
    return dist_prediction, ref_prediction


def predict(scoring_model, img: typing.Union[np.ndarray, str]) -> float:
    assert img is not None, "Invalid path or image type"
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)

    logger.info("Predicting the final score for IQA")
    img = tf.convert_to_tensor(img, dtype=tf.float32)

    start = perf_counter()
    I_d = image_preprocess(img)
    end = perf_counter()
    logger.debug(f"IQA preprocessing function image_preprocess took {end-start} seconds")

    start = perf_counter()
    I_d = tf.tile(I_d, (1, 1, 1, 3))
    prediction = scoring_model.predict(I_d)[0][0]
    end = perf_counter()

    logger.debug(f"Keras model took {end-start} seconds to predict the iqa score")
    logger.info(f"final IQA score is {prediction}")
    return prediction


def main():
    import argparse
    parser = argparse.ArgumentParser("Script use to predict Image quality using Deep image quality assesement")
    parser.add_argument("--img-path", required=True, help="Pass image location as n args")
    args = vars(parser.parse_args())
    args['scoring_model'] = init_model()
    predict(**args)
