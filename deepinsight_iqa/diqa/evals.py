import six
import numpy as np
import tensorflow as tf
from typing import Union, Tuple, Optional
from deepinsight_iqa.common.utility import thread_safe_singleton
from .networks.model import Diqa
from deepinsight_iqa.diqa.data import get_iqa_datagen
from deepinsight_iqa.diqa.utils.tf_imgutils import image_preprocess
from deepinsight_iqa.diqa.networks.utils import SpearmanCorrMetric
from .networks.utils import loss_fn
from tensorflow.keras import losses as KLosses
from pathlib import Path
import pandas as pd


@six.add_metaclass(thread_safe_singleton)
class Evaluation:
    def __init__(
        self,
        model_dir: Optional[str] = None,
        weight_file: Optional[str] = None,
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
        self.diqa.load_weights(model_path)
        self.metric = SpearmanCorrMetric()

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
        dist_prediction = self.diqa.subjective.predict(I_d)[0][0]
        ref_prediction = self.diqa.subjective.predict(I_r)[0][0]
        return dist_prediction, ref_prediction

    def save_json(self, data, target_file):
        import json
        with Path(target_file).open('w') as f:
            json.dump(data, f, indent=2, sort_keys=True)

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

        nb_samples = len(datagen)
        predictions = self.diqa.predict_generator(datagen)
        df = pd.read_csv(csv_path)
        outputs = []
        for idx in range(nb_samples):
            pred = predictions[idx]
            gt = df['mos']
            outputs.append({
                "gt": gt,
                "pred": pred,
                "reference_image": df["reference_image"],
                "distorted_image": df["distorted_image"]
            })
            self.metric.update_state(pred, gt)

        self.save_json(outputs, prediction_file)
        result = self.metric.result()
        self.metric.reset_states()
        print(f">> Overall Score: {result} >>")

        return result
