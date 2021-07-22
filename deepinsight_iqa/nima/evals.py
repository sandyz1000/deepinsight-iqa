import os
import glob
import sys
from typing import Optional
from .utils.utils import calc_mean_score, save_json, image_dir_to_json, image_file_to_json
from .handlers.model_builder import Nima
from ..data_pipeline.nima_gen.nima_datagen import NimaDataGenerator as TestDataGenerator


class Evaluation:
    """ Given a subset of image dataset generate evaluation report """

    def init_model(self, base_model_name="MobileNet", weights_file=None):
        try:
            self.nima = Nima(base_model_name, weights=None)
            self.nima.build()
            self.nima.nima_model.load_weights(weights_file) if weights_file else None
        except Exception as e:
            print("Unable to load NIMA weights", str(e))
            sys.exit(1)

    def evaluation(
        self,
        image_source: str,
        predictions_file: Optional[str] = None,
        n_classes: int = 10,
        batch_size: int = 64,
        img_format: Optional[str] = 'jpg',
        multiprocessing_data_load: bool = False
    ):
        # TODO: Save output and Calculate metric and save to csv

        if os.path.isfile(image_source):
            image_dir, samples = image_file_to_json(image_source)
        else:
            image_dir = image_source
            samples = image_dir_to_json(image_dir, img_type='jpg')

        # initialize data generator
        data_generator = TestDataGenerator(
            samples, image_dir, batch_size, n_classes,
            self.nima.preprocessing_function(),
            img_format=img_format, do_train=False, shuffle=False
        )

        # get predictions
        predictions = self.nima.nima_model.predict_generator(
            data_generator, workers=8, use_multiprocessing=multiprocessing_data_load, verbose=1)

        # calc mean scores and add to samples
        for i, sample in enumerate(samples):
            sample['mean_score_prediction'] = calc_mean_score(predictions[i])

        if predictions_file is not None:
            save_json(samples, predictions_file)

        return samples
