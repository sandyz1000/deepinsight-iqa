import os
import six
import numpy as np
from deepinsight_iqa.common.utility import thread_safe_singleton
from .utils.utils import image_preprocess
from .handlers.model import Diqa
from scipy.stats import pearsonr, spearmanr
from ..data_pipeline.diqa_gen import diqa_datagen
import pandas as pd


@six.add_metaclass(thread_safe_singleton)
class Evaluation:
    def __init__(
        self,
        image_dir: str,
        model_path: str,
        csv_path: str,
        custom: bool = False,
        batch_size: int = 1,
        out_path: str = "report.csv",
    ):
        """
        Use pearson and spearman correlation matrix for evaluation, we can also use RMSE error to 
        calculate the mean differences
        """

        self.diqa = Diqa(custom=custom)
        self.diqa._build()
        assert csv_path.split(".")[-1:] == 'csv', "Not a valid file extension"

        df = pd.read_csv(csv_path)
        self.test_generator = diqa_datagen.LiveDataRowParser(
            df,
            image_dir,
            batch_size,
            img_preprocessing=image_preprocess,
            shuffle=False,
            check_dir_availability=False
        )

        self.diqa.subjective_score_model.load_weights(model_path)

    def __call__(self):
        # TODO: Use prediction to create evaluation metrics
        nb_samples = len(self.test_generator)
        predictions = self.diqa.subjective_score_model.predict_generator(self.test_generator, steps=nb_samples)
