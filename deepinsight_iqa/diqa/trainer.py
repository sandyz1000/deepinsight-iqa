from typing import Optional, Callable, Iterator, Union
import os
import tqdm
import sys
import enum
import logging
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from .networks.model import Diqa, BaseModel
from .networks.utils import gradient, calculate_error_map, loss
from deepinsight_iqa.common.utility import set_gpu_limit, get_stream_handler
# set_gpu_limit(10)

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


class ModelType(enum.Enum):
    objective = "objective"
    subjective = "subjective"


def generate_random_name(batch_size, epochs):
    import random_name
    model_filename = f"diqa-{random_name.generate_name()}-{batch_size}-{epochs}.h5"
    return model_filename


class TrainerStep:
    def __init__(self, model, name, is_training: bool = False, optimizer=None, **kwds) -> None:
        self.model = model  # type: BaseModel
        self.loss = tf.keras.metrics.Mean(f'{name}_loss', dtype=tf.float32)
        self.accuracy = tf.keras.metrics.MeanSquaredError(f'{name}_accuracy')
        self.is_training = is_training
        self.optimizer = optimizer
        self.scaling_factor = kwds['scaling_factor']

    def __call__(self, I_d, I_r):
        I_d, e_gt, r = calculate_error_map(I_d, I_r, scaling_factor=self.scaling_factor)
        if self.is_training:
            loss_value, gradients = gradient(self.model, I_d, e_gt, r, objective_output=self.is_training)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        else:
            loss_value = loss(self.model, I_d, e_gt, r)
        self.loss(loss_value)
        self.accuracy(e_gt, self.model(I_d))


class Trainer:

    def __init__(
        self,
        train_datagen: Union[Iterator, tf.data.Dataset],
        valid_datagen: Union[Iterator, tf.data.Dataset] = None,
        model_dir: Optional[str] = None,
        subjective_weightfname: Optional[str] = None,
        objective_weightfname: Optional[str] = None,
        epochs: int = 5, 
        batch_size: int = 16,
        multiprocessing_data_load: bool = False,
        extra_epochs: int = 1, 
        num_workers_data_load: int = 1,
        use_pretrained: bool = False,
        log_dir: str = 'logs',
        custom: bool = False,
        verbose: bool = False,
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
        self.base_model_name = kwargs.pop('base_model_name')
        self.epochs = epochs
        self.subjective_weightfname = subjective_weightfname if subjective_weightfname \
            else generate_random_name(batch_size, epochs)
        self.objective_weightfname = objective_weightfname if objective_weightfname \
            else generate_random_name(batch_size, epochs)
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.multiprocessing_data_load = multiprocessing_data_load
        self.extra_epochs = extra_epochs
        self.batch_size = batch_size
        self.num_workers_data_load = num_workers_data_load
        self.use_pretrained = use_pretrained
        self.steps_per_epoch = kwargs['steps_per_epoch']
        self.validation_steps = kwargs['validation_steps']
        self.kwargs = kwargs
        self.diqa = Diqa(self.base_model_name, custom=custom)
        self.diqa.build(batch_size)

        self.train_datagen = train_datagen
        self.valid_datagen = valid_datagen

        network = kwargs.pop('network')
        if kwargs.get('pretrained', False):
            self.diqa.load_weights(self.model_dir, network)

    def train_objective(self):
        from .utils.callbacks import TensorBoardBatch
        
        # TODO: Convert eager execution with graph
        train_step = TrainerStep(
            self.diqa, "train", True,
            optimizer=tf.optimizers.Nadam(learning_rate=2 * 10 ** -4),
            scaling_factor=self.kwargs['scaling_factor']
        )
        valid_step = TrainerStep(
            self.diqa, "valid", False,
            scaling_factor=self.kwargs['scaling_factor']
        )

        # tensorboard = TensorBoardBatch(
        #     log_dir=self.log_dir,
        #     metrics=[
        #         {'loss': train_step.loss}, {'accuracy': train_step.accuracy},
        #         {'val_loss': valid_step.loss}, {'val_accuracy': valid_step.accuracy}
        #     ]
        # )

        # ## ------------------------------
        # BEGIN EPOCH
        # ## ------------------------------
        train_summary_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, 'train'))
        test_summary_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, 'valid'))

        for epoch in tqdm.tqdm(range(self.epochs), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}'):
            train_step_cnt = 0
            for I_d, I_r, mos in self.train_datagen:
                train_step(I_d, I_r)
                train_step_cnt += 1
                if train_step_cnt >= self.steps_per_epoch:
                    break
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_step.loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_step.accuracy.result(), step=epoch)

            if self.valid_datagen:
                valid_step_cnt = 0
                for I_d, I_r, mos in self.valid_datagen:
                    valid_step(I_d, I_r)
                    valid_step_cnt += 1
                    if valid_step_cnt >= self.validation_steps:
                        break
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', valid_step.loss.result(), step=epoch)
                    tf.summary.scalar('accuracy', valid_step.accuracy.result(), step=epoch)

            # tensorboard.on_epoch_end(epoch)
            if self.valid_datagen:
                template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
                logging.info(template.format(
                    epoch + 1,
                    train_step.loss.result(),
                    train_step.accuracy.result() * 100,
                    valid_step.loss.result(),
                    valid_step.accuracy.result() * 100)
                )
            else:
                template = 'Epoch {}, Loss: {}, Accuracy: {}'
                logging.info(template.format(
                    epoch + 1,
                    train_step.loss.result(),
                    train_step.accuracy.result() * 100,)
                )

        self.diqa.save_pretrained(self.model_dir, prefix='objective')

    def train_final(self):
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)
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

        self.diqa.subjective_model.fit(
            train_datagen,
            validation_data=valid_datagen,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            epochs=self.epochs + self.extra_epochs,
            initial_epoch=self.epochs,
            use_multiprocessing=self.multiprocessing_data_load,
            workers=self.num_workers_data_load,
            callbacks=[model_checkpointer, tensorboard_callback]
        )
        
        self.diqa.save_pretrained(self.model_dir, prefix='subjective')
        
