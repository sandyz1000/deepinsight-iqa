import os
import six
import numpy as np
import tensorflow as tf
from typing import Union, Iterator, Tuple, Optional
from deepinsight_iqa.common.utility import thread_safe_singleton
from .networks.model import Diqa
from deepinsight_iqa.diqa.data import get_iqa_datagen
from deepinsight_iqa.data_pipeline.diqa_gen.datagenerator import DiqaCombineDataGen
from deepinsight_iqa.diqa.utils.tf_imgutils import image_preprocess
from deepinsight_iqa.diqa.networks.utils import SpearmanCorrMetric
from pathlib import Path
import pandas as pd


@six.add_metaclass(thread_safe_singleton)
class Evaluation:
    def __init__(
        self,
        # datagen: Union[DiqaCombineDataGen, tf.data.Dataset, Iterator],
        model_dir: Optional[str] = None,
        weight_filename: Optional[str] = None,
        model_type: str = None,
        batch_size: int = 1,
        out_path: str = "report.csv",
        kwargs: dict = {}
    ):
        """
        Use pearson and spearman correlation matrix for evaluation, we can also use RMSE error to 
        calculate the mean differences
        """
        self.kwargs = kwargs
        self.batch_size = kwargs.pop('batch_size', batch_size)
        self.out_path = out_path
        
        bottleneck_layer_name = kwargs.pop('bottleneck', None)
        network = kwargs.pop('network', 'subjective')

        self.diqa = Diqa(model_type, bottleneck_layer_name)
        self.metric = SpearmanCorrMetric()
        self.diqa.load_weights(Path(model_dir, weight_filename), prefix=network)

    def img_pair_score(
        self,
        reference_image: Union[np.ndarray, tf.Tensor],
        distorted_image: Union[np.ndarray, tf.Tensor]
    ) -> Tuple[float, float]:
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
        dist_prediction = self.diqa.subjective_model.predict(I_d)[0][0]
        ref_prediction = self.diqa.subjective_model.predict(I_r)[0][0]
        return dist_prediction, ref_prediction

    def __call__(self, image_dir, csv_path=None, prediction_file=None):
        
        datagen = get_iqa_datagen(
            image_dir,
            csv_path,
            image_preprocess=image_preprocess,
            dataset_type=self.kwargs['dataset_type'],
            input_size=self.kwargs['input_size'],
            do_augment=False,
            do_train=False,
            channel_dim=self.kwargs['channel_dim'],
            batch_size=self.batch_size,
            split_dataset=False
        )
        
        # nb_samples = len(datagen)
        predictions = self.diqa.subjective_model.predict_generator(self.datagen)
        true_values = pd.read_csv(csv_path)['mos'].values
        # TODO: Write to prediction file
        for pred, true_val in zip(predictions, true_values):
            self.metric.update_state(pred, true_val)

        result = self.metric.result()
        self.metric.reset_states()
        return result
