import os
import glob
import sys
from typing import Optional, List, Union
from .utils.utils import calc_mean_score, save_json, image_dir_to_json, image_file_to_json
from .handlers.model_builder import Nima
from deepinsight_iqa.common.utility import thread_safe_singleton, set_gpu_limit
from deepinsight_iqa.data_pipeline.nima_gen.nima_datagen import NimaDataGenerator as TestDataGenerator
import tensorflow as tf
import six
import logging
logger = logging.getLogger(__name__)


@six.add_metaclass(thread_safe_singleton)
class Prediction:
    def __init__(self, weights_file: str, base_model_name: str):
        """ Invoke a predict method of this class to predict image quality using nima model
        """
        try:
            # set_gpu_limit()

            self.nima = Nima(base_model_name, weights=None)
            self.nima.build()
            self.nima.nima_model.load_weights(weights_file)

        except Exception as e:
            print("Unable to load NIMA weights", str(e))
            sys.exit(1)

    def predict(
        self,
        image_source: str,
        predictions_file: Optional[str] = None,
        img_format: str = 'jpg'
    ) -> List:
        # load samples
        if os.path.isfile(image_source):
            image_dir, samples = image_file_to_json(image_source)
        else:
            image_dir = image_source
            samples = image_dir_to_json(image_source, img_type='jpg')

        # initialize data generator
        n_classes = 10
        batch_size = 64
        samples = []
        sample = {"imgage_id": "img_1"}
        samples.append(sample)
        data_generator = TestDataGenerator(
            samples, image_dir, batch_size, n_classes,
            self.nima.preprocessing_function(), img_format=img_format
        )

        # get predictions
        predictions = self.nima.nima_model.predict_generator(
            data_generator, workers=1, use_multiprocessing=False, verbose=1)

        # calc mean scores and add to samples
        for i, sample in enumerate(samples):
            sample['mean_score_prediction'] = calc_mean_score(predictions[i])

        # print(json.dumps(samples, indent=2))

        if predictions_file is not None:
            save_json(samples, predictions_file)

        return samples
