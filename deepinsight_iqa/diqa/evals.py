import os
import six
import numpy as np
import tensorflow as tf
from typing import Union, Iterator
from deepinsight_iqa.common.utility import thread_safe_singleton
from deepinsight_iqa.diqa.handlers.model import Diqa
from scipy.stats import pearsonr, spearmanr
import pandas as pd


@six.add_metaclass(thread_safe_singleton)
class Evaluation:
    def __init__(
        self,
        data_iter: Union[tf.data.Dataset, Iterator],
        model_path: str,
        csv_path: str,
        custom: bool = False,
        batch_size: int = 1,
        out_path: str = "report.csv",
        **kwargs
    ):
        """
        Use pearson and spearman correlation matrix for evaluation, we can also use RMSE error to 
        calculate the mean differences
        """
        base_model_name = kwargs.pop('base_model_name')
        self.diqa = Diqa(base_model_name, custom=custom)
        self.diqa._build()
        self.datagen = data_iter
        df = pd.read_csv(csv_path)
        
        self.diqa.subjective.load_weights(model_path)
        self.final_model = self.diqa.subjective

    def __call__(self):
        # TODO: Use prediction to create evaluation metrics
        nb_samples = len(self.test_generator)
        predictions = self.final_model.predict_generator(
            self.datagen, steps=nb_samples
        )
