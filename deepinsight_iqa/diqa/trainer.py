from typing import Optional, Callable, Iterator, Union
import os
import tqdm
import sys
import enum
import logging
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import metrics as KMetric
from tensorflow.keras import losses as KLosses
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow.keras.backend as K
from .networks.model import Diqa, BaseModel
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


# class TrainerStep:
#     def __init__(self, model, name, is_training: bool = False, **kwds) -> None:
#         self.model = model  # type: BaseModel
#         self.loss = KMetric.Mean(name=f'{name}_loss', dtype=tf.float32)
#         self.accuracy = KMetric.MeanSquaredError(name=f'{name}_accuracy')
#         # self.accuracy = SpearmanCorrMetric(name=f'{name}_accuracy')
#         self.is_training = is_training
#         self.scaling_factor = kwds['scaling_factor']

#     def __call__(self, distorted, reference):
#         # reference = tf.slice(reference, begin=[0, 0, 0, 0], size=(reference.shape[:-1] + [1]))
#         # distorted_gray = tf.slice(distorted, begin=[0, 0, 0, 0], size=(distorted.shape[:-1] + [1]))
#         # e_gt, r = calculate_error_map(distorted_gray, reference, scaling_factor=self.scaling_factor)
#         # if self.is_training:
#         #     loss_value, gradients = gradient(self.model, distorted, e_gt, r)
#         #     self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
#         # else:
#         #     loss_value = loss_fn(self.model, distorted, e_gt, r)

#         # loss = self.loss(loss_value)
#         # err_pred = self.model(distorted, is_objective=True)
#         # _shape = tf.TensorShape([e_gt.shape[0], tf.reduce_prod(e_gt.shape[1:]).numpy()])

#         # err_pred = tf.reshape(err_pred, shape=_shape)
#         # e_gt = tf.reshape(e_gt, shape=_shape)
#         # acc = self.accuracy(e_gt, err_pred)

#         # return loss, acc

#     def reset_states(self):
#         self.loss.reset_states()
#         self.accuracy.reset_states()


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
        weight_file: str = None,
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
        self.diqa = Diqa(
            self.model_type,
            self.bottleneck_layer,
            optimizer=tf.optimizers.Nadam(learning_rate=2 * 10 ** -4),
            train_bottleneck=kwargs.pop('train_bottleneck', False)
        )

        self.train_datagen = train_datagen  # type: DiqaDataGenerator
        self.valid_datagen = valid_datagen  # type: DiqaDataGenerator

        if 'steps_per_epoch' in kwargs:
            self.train_datagen.steps_per_epoch = min(kwargs['steps_per_epoch'],
                                                     self.train_datagen.steps_per_epoch)

        if 'validation_steps' in kwargs:
            self.valid_datagen.steps_per_epoch = min(kwargs['validation_steps'],
                                                     self.valid_datagen.steps_per_epoch)

        network = kwargs.pop('network', 'subjective')
        if self.use_pretrained:
            self.diqa.load_weights(self.model_dir, weight_file, prefix=network)

        self.__state_name = None

    def train(self):
        pass

    def slow_trainer(self):
        pass

    def train_objective(self):

        train_step = TrainerStep(
            self.diqa,
            "objective",
            is_training=True,
            scaling_factor=self.kwargs['scaling_factor']
        )
        valid_step = TrainerStep(
            self.diqa,
            "objective",
            is_training=False,
            scaling_factor=self.kwargs['scaling_factor']
        )

        tbc = TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        tbc.set_model(self.diqa.objective)

        # ## ## ## ## ## ## ## ## ## ##
        # BEGIN EPOCH
        # ## ## ## ## ## ## ## ## ## ##

        for epoch in tqdm.tqdm(range(self.epochs), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}'):
            min_step_count = min(self.train_datagen.steps_per_epoch,
                                 self.valid_datagen.steps_per_epoch)
            batch_idx = 0
            while batch_idx < min_step_count:

                I_d, I_r, mos = self.train_datagen[batch_idx]
                if I_d.shape != I_r.shape:
                    continue
                
                loss, accuracy = train_step(I_d, I_r)

                if self.valid_datagen:
                    I_d_val, I_r_val, _ = self.valid_datagen[batch_idx]
                    val_loss, val_accuracy = valid_step(I_d_val, I_r_val)

                tbc.on_batch_end(batch_idx, logs={
                    'loss': loss,
                    'accuracy': accuracy,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                })
                batch_idx += 1

            tbc.on_epoch_end(epoch=epoch, logs={
                'loss': train_step.loss.result(),
                'accuracy': train_step.accuracy.result(),
                'val_loss': valid_step.loss.result(),
                'val_accuracy': valid_step.accuracy.result()
            })

            if self.valid_datagen:
                template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
                print(template.format(
                    epoch + 1,
                    train_step.loss.result(),
                    train_step.accuracy.result() * 100,
                    valid_step.loss.result(),
                    valid_step.accuracy.result() * 100)
                )
            else:
                template = 'Epoch {}, Loss: {}, Accuracy: {}'
                print(template.format(
                    epoch + 1,
                    train_step.loss.result(),
                    train_step.accuracy.result() * 100,)
                )

            # Reset metrics every epoch
            train_step.reset_states()
            valid_step.reset_states()

        self.__state_name = 'objective'

    def save_weights(self):
        self.diqa.save_pretrained(self.model_dir, prefix=self.__state_name)

    def train_final(self):
        name = 'subjective'

        tbc = TensorBoard(log_dir=self.log_dir.as_posix(), histogram_freq=1)
        model_checkpointer = ModelCheckpoint(
            filepath=self.model_dir,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True
        )

        train_datagen = self.train_datagen
        valid_datagen = self.valid_datagen
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

        self.diqa.compile(
            optimizer=tf.optimizers.Nadam(learning_rate=2 * 10 ** -4),
            loss=KLosses.MeanSquaredError(name=f'{name}_losses'),
            metrics=[KMetric.MeanSquaredError(name=f'{name}_accuracy')]
        )

        self.diqa.subjective.fit(
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

        self.__state_name = 'subjective'
