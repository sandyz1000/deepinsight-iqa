from typing import Optional, Callable, Iterator, Union
import os
import tqdm
import sys
import enum
import logging
from pathlib import Path
import tensorflow as tf

import tensorflow.keras.models as KM
from tensorflow.keras import losses as KLosses
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

# import keras.models as KM
# from keras import metrics as KMetric
# from keras import losses as KLosses
# from keras.callbacks import ModelCheckpoint, TensorBoard

from . import OBJECTIVE_NW, SUBJECTIVE_NW
from .networks.model import Diqa
from .networks.utils import loss_fn
from deepinsight_iqa.common.utility import get_stream_handler
from deepinsight_iqa.data_pipeline.diqa_gen.datagenerator import DiqaDataGenerator

logger = logging.getLogger(__name__)
logger.addHandler(get_stream_handler())
logger.info(
    '__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(
        __file__, __name__, str(__package__))
)


def return_strategy():
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) == 0:
        return tf.distribute.OneDeviceStrategy(device="/cpu:0")
    elif len(physical_devices) == 1:
        return tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        return tf.distribute.MirroredStrategy()


class Trainer:

    def __init__(
        self,
        train_datagen: Union[DiqaDataGenerator, Iterator, tf.data.Dataset],
        valid_datagen: Union[DiqaDataGenerator, Iterator, tf.data.Dataset] = None,
        model_dir: Optional[str] = None,
        epochs: int = 5,
        batch_size: int = 16,
        use_pretrained: bool = False,
        use_multiprocessing: bool = False,
        extra_epochs: int = 1,
        num_workers: int = 1,
        log_dir: str = 'logs',
        **kwargs
    ):
        """
        Similar to init_train but use Keras generator for training, we have more control over the API
        with image augmentation

        1. Image augmentation such as (crop, shift and rotation) will useful to check the quality
            is added to get more variance in training output
        2. Split of dataset into training and test subsets
        3. Evaluation metric on the test set
        4. Larger Batch size can be used for training

        """
        self.bottleneck_layer = kwargs.pop('bottleneck', None)
        self.model_type = kwargs.pop('model_type', None)
        self.epochs = epochs
        self.model_dir = model_dir
        self.log_dir = Path(log_dir)
        self.log_dir.absolute().mkdir(parents=True, exist_ok=True)

        self.use_multiprocessing = use_multiprocessing
        self.extra_epochs = extra_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_pretrained = use_pretrained
        self.kwargs = kwargs

        self.train_datagen = train_datagen  # type: DiqaDataGenerator
        self.valid_datagen = valid_datagen  # type: DiqaDataGenerator

        if 'steps_per_epoch' in kwargs:
            self.train_datagen.steps_per_epoch = min(kwargs['steps_per_epoch'],
                                                     self.train_datagen.steps_per_epoch)

        if 'validation_steps' in kwargs:
            self.valid_datagen.steps_per_epoch = min(kwargs['validation_steps'],
                                                     self.valid_datagen.steps_per_epoch)

        self.network = kwargs.get('network', 'subjective')

    def train(self, diqa: Diqa):
        raise NotImplementedError
        tbc = TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        model_checkpointer = ModelCheckpoint(
            filepath=self.model_dir,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True
        )

        train_datagen = self.train_datagen
        valid_datagen = self.valid_datagen
        
        if self.network == SUBJECTIVE_NW:
            if isinstance(self.train_datagen, tf.data.Dataset):
                train_datagen = train_datagen.map(
                    lambda im, _, mos: (im, mos),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE
                )
                valid_datagen = valid_datagen.map(
                    lambda im, _, mos: (im, mos),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE
                )
            else:
                train_datagen = ((im, mos) for im, _, mos in train_datagen)
                valid_datagen = ((im, mos) for im, _, mos in valid_datagen)

        diqa.fit_generator(
            train_datagen,
            validation_data=valid_datagen,
            steps_per_epoch=self.train_datagen.steps_per_epoch,
            validation_steps=self.valid_datagen.steps_per_epoch,
            epochs=self.epochs + self.extra_epochs,
            initial_epoch=self.epochs,
            use_multiprocessing=self.use_multiprocessing,
            workers=self.num_workers,
            callbacks=[model_checkpointer, tbc]
        )

    def compile(self, train_bottleneck=False):
        diqa = Diqa(
            self.model_type,
            self.bottleneck_layer,
            train_bottleneck=train_bottleneck,
            kwds={"scaling_factor": self.kwargs['scaling_factor']}
        )
        cond_loss_fn = (
            loss_fn
            if self.network == 'objective'
            else KLosses.MeanSquaredError(name=f'{self.network}_losses')
        )
        
        diqa.compile(
            optimizer=tf.optimizers.Nadam(learning_rate=2 * 10 ** -4),
            loss_fn=cond_loss_fn,
            current_ops=self.network
        )
        # diqa.build()
        return diqa
    
    def load_weights(self, diqa: KM.Model, model_path: str):
        if not os.path.exists(model_path):
            model_path = Path(self.model_dir) / model_path

        if model_path.exists():
            diqa.load_weights(model_path)
        else:
            print(f"Model path {model_path} not found, training from start!!")
        
    def slow_trainer(self, diqa: Diqa):

        tbc = TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        tbc.set_model(diqa)

        # ## ## ## ## ## ## ## ## ## ##
        # BEGIN EPOCH
        # ## ## ## ## ## ## ## ## ## ##

        for epoch in tqdm.tqdm(range(self.epochs), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}'):

            batch_idx = 0
            while batch_idx < self.train_datagen.steps_per_epoch:

                data = self.train_datagen[batch_idx]
                metrics = diqa.train_step(data)
                batch_idx += 1

                logs = {
                    'loss': metrics['loss'],
                    'accuracy': metrics['accuracy'],
                }
                tbc.on_batch_end(batch_idx, logs=logs)

            batch_idx = 0
            while self.valid_datagen and batch_idx < self.valid_datagen.steps_per_epoch:

                val_data = self.valid_datagen[batch_idx]
                val_metrics = diqa.test_step(val_data)
                batch_idx += 1

                logs = {
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy']
                }
                tbc.on_batch_end(batch_idx, logs=logs)

            logs = {
                'loss': metrics['loss'],
                'accuracy': metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy']
            }
            tbc.on_epoch_end(epoch=epoch, logs=logs)

            if self.valid_datagen:
                template = f"Epoch {epoch + 1}, Loss: {metrics['loss']}, Accuracy: {metrics['accuracy'] * 100}, "
                f"Test Loss: {val_metrics['loss']}, Test Accuracy: {val_metrics['accuracy'] * 100}"
            else:
                template = f"Epoch {epoch + 1}, Loss: {metrics['logs']}, Accuracy: {metrics['accuracy']}"
            print(template)

    def reset_state(self, diqa: Diqa):
        # Reset metrics every epoch
        diqa.acc1_metric.reset_states()
        if self.network:
            diqa.o_loss_metric.reset_states()
        else:
            diqa.s_loss_metric.reset_states()

    def save_weights(self, diqa: Diqa):
        diqa.save_pretrained(self.model_dir, prefix=self.network)