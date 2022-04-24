import sys
import os
from typing import Optional
import tensorflow as tf
from .networks.model import Diqa
from deepinsight_iqa.diqa.utils.tf_imgutils import image_preprocess
from time import perf_counter
import logging
import numpy as np
from typing import Union, Tuple
import cv2
import six
from pathlib import Path
from .networks.utils import loss_fn
from tensorflow.keras import losses as KLosses
import glob
from deepinsight_iqa.data_pipeline.diqa_gen.datagenerator import DiqaCombineDataGen

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
        weight_file: Optional[str] = None,
        model_type: str = None,
        **kwds
    ):
        try:
            self.channel_dim = 3
            bottleneck_layer_name = kwds.pop('bottleneck', None)
            network = kwds.pop('network', 'subjective')
            self.diqa = Diqa(model_type, bottleneck_layer_name)
            
            cond_loss_fn = (
                loss_fn
                if self.network == 'objective'
                else KLosses.MeanSquaredError(name=f'{self.network}_losses')
            )
            
            self.diqa.compile(
                optimizer=tf.optimizers.Nadam(learning_rate=2 * 10 ** -4),
                loss_fn=cond_loss_fn,
                current_ops=self.network
            )
            self.diqa.build()
            model_path = Path(model_dir) / weight_file
            self.diqa.load_weights(model_path, prefix=network)
        
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
        I_d = tf.tile(I_d, (1, 1, 1, self.channel_dim))
        prediction = self.diqa.predict(I_d)[0][0]
        end = perf_counter()

        logger.debug(f"Keras model took {end-start} seconds to predict the iqa score")
        logger.info(f"final IQA score is {prediction}")
        return prediction

    def predict_batch(self, img_dir, batch_size=3,):
        start = perf_counter()
        samples = image_dir_to_json(img_dir)
        images = [row['path'] for row in samples]
        X_data = DiqaCombineDataGen(
            image_dir=img_dir,
            samples=images,
            batch_size=batch_size,
            img_preprocessing=image_preprocess
        )

        predictions = self.diqa.subjective.predict_generator(
            X_data,
            workers=1,
            use_multiprocessing=False,
            verbose=1
        )
        for i, sample in enumerate(samples):
            sample['score'] = predictions[i]
        end = perf_counter()
        logger.debug(f"Predictions on batch of {batch_size} took {end-start} seconds")
        return samples
