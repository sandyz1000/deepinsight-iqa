from typing import Optional, Callable, Iterator, Union
from .handlers.utils import gradient, calculate_error_map, loss
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from .handlers.model import Diqa
from .utils.callbacks import TensorBoardBatch
import logging
from abc import abstractmethod, ABCMeta
from six import add_metaclass
import tqdm
import sys
import enum

logger = logging.getLogger(__name__)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(logging.Formatter(
    "%(asctime)s %(name)-12s %(levelname)-8s %(message)s", "%Y-%m-%dT%H:%M:%S"
))
logger.addHandler(stdout_handler)


logger.info(
    '__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(
        __file__, __name__, str(__package__))
)


class ModelType(enum.Enum):
    objective = "objective"
    subjective = "subjective"


def generate_random_name(batch_size, epochs):
    import random_name
    model_filename = f"diqa-{random_name.generate_name()}-{batch_size}-{epochs}.h5"
    return model_filename


class TrainerStep:
    def __init__(self, model, name, is_training: bool = False, optimizer=None, **kwds) -> None:
        self.model = model
        self.loss = tf.keras.metrics.Mean(f'{name}_loss', dtype=tf.float32)
        self.accuracy = tf.keras.metrics.MeanSquaredError(f'{name}_accuracy')
        self.is_training = is_training
        self.optimizer = optimizer
        self.scaling_factor = kwds['scaling_factor']

    def __call__(self, I_d, I_r):
        I_d, e_gt, r = calculate_error_map(I_d, I_r, scaling_factor=self.scaling_factor)
        if self.is_training:
            loss_value, gradients = gradient(self.model, I_d, e_gt, r)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        else:
            loss_value = loss(self.model, I_d, e_gt, r)
        self.loss(loss_value)
        self.accuracy(e_gt, self.model(I_d))


@add_metaclass(ABCMeta)
class Trainer:
    def __init__(
        self,
        model_dir: Optional[str] = None,
        objective_wts_filename: Optional[str] = None,
        final_wts_filename: Optional[str] = None,
        log_dir: str = './logs',
        epochs: int = 5, batch_size: int = 16,
        multiprocessing_data_load: bool = False,
        extra_epochs: int = 1, num_workers_data_load: int = 1,
        use_pretrained: bool = False,
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
        if not final_wts_filename:
            final_wts_filename = generate_random_name(batch_size, epochs)
        self.final_wts_filename = final_wts_filename
        self.objective_wts_filename = objective_wts_filename
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
        self.diqa._build()

    def loadweights(self, load_model: str):
        filename = self.objective_wts_filename if load_model == ModelType.objective else self.final_wts_filename
        model_path = os.path.join(self.model_dir, self.base_model_name, filename)
        assert os.path.exists(model_path), FileNotFoundError("Objective Model file not found")
        if load_model == ModelType.objective:
            self.diqa.objective.load_weights(model_path)
        else:
            self.diqa.subjective.load_weights(model_path)

    @abstractmethod
    def train_subjective(self, model: tf.keras.Model):
        """
        Train Subjective Model
        ------------------------
        """
        raise NotImplemented

    @abstractmethod
    def train_objective(self, model: tf.keras.Model):
        """
        Train objective Model
        ------------------------
        For the custom training loop, it is necessary to:

        1. Define a metric to measure the performance of the model.
        2. Calculate the loss and the gradients.
        3. Use the optimizer to update the weights.
        4. Print the accuracy.

        Args:
            model ([type]): [description]

        Raises:
            NotImplemented: [description]
        """
        raise NotImplemented


class Train(Trainer):

    def __init__(
        self,
        train_iter: Union[Iterator, tf.data.Dataset],
        valid_iter: Union[Iterator, tf.data.Dataset] = None,
        model_dir: Optional[str] = None,
        final_wts_filename: Optional[str] = None,
        objective_wts_filename: Optional[str] = None,
        epochs: int = 5, batch_size: int = 16,
        multiprocessing_data_load: bool = False,
        extra_epochs: int = 1, num_workers_data_load: int = 1,
        use_pretrained: bool = False,
        log_dir: str = 'logs',
        custom: bool = False,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(
            model_dir=model_dir,
            final_wts_filename=final_wts_filename,
            objective_wts_filename=objective_wts_filename,
            epochs=epochs,
            batch_size=batch_size,
            extra_epochs=extra_epochs,
            multiprocessing_data_load=multiprocessing_data_load,
            num_workers_data_load=num_workers_data_load,
            use_pretrained=use_pretrained,
            log_dir=log_dir, custom=custom, verbose=verbose,
            **kwargs
        )
        self.trainiter = train_iter
        self.validiter = valid_iter

    def train_objective(self, model: tf.keras.Model):

        train_step = TrainerStep(
            model, "train", True,
            optimizer=tf.optimizers.Nadam(learning_rate=2 * 10 ** -4),
            scaling_factor=self.kwargs['scaling_factor']
        )
        valid_step = TrainerStep(
            model, "valid", False,
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

        for epoch in tqdm.tqdm(range(self.epochs)):
            train_step_cnt = 0
            for I_d, I_r, mos in self.trainiter:
                train_step(I_d, I_r)
                train_step_cnt += 1
                if train_step_cnt >= self.steps_per_epoch:
                    break
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_step.loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_step.accuracy.result(), step=epoch)

            if self.validiter:
                valid_step_cnt = 0
                for I_d, I_r, mos in self.validiter:
                    valid_step(I_d, I_r)
                    valid_step_cnt += 1
                    if valid_step_cnt >= self.validation_steps:
                        break
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', valid_step.loss.result(), step=epoch)
                    tf.summary.scalar('accuracy', valid_step.accuracy.result(), step=epoch)

            # tensorboard.on_epoch_end(epoch)
            if self.validiter:
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

        # Save the objective model
        model_path = os.path.join(self.model_dir, self.base_model_name, self.objective_wts_filename)
        os.makedirs(os.path.basename(model_path), exist_ok=True)
        model.save(model_path)

    def train_subjective(self, model: tf.keras.Model):
        model_checkpointer = ModelCheckpoint(
            filepath=self.model_dir,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True
        )

        subjective_datagen = (
            lambda datagen: ((im, mos) for im, _, mos in datagen)
        )

        model.fit(
            subjective_datagen(self.trainiter),
            steps_per_epoch=self.steps_per_epoch,
            validation_data=subjective_datagen(self.validiter),
            validation_steps=self.validation_steps,
            epochs=self.epochs + self.extra_epochs,
            initial_epoch=self.epochs,
            verbose=1,
            use_multiprocessing=self.multiprocessing_data_load,
            workers=self.num_workers_data_load,
            callbacks=[model_checkpointer]
        )
        # Save the subjective model
        model_path = os.path.join(self.model_dir, self.base_model_name, self.final_wts_filename)
        os.makedirs(os.path.basename(model_path), exist_ok=True)
        model.save(model_path)
        K.clear_session()
