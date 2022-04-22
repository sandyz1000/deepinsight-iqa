import os
import numpy as np
from typing import List, Union, Optional
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from ..data_pipeline.nima_gen.nima_datagen import NimaDataGenerator
from .handlers.model_builder import Nima
from deepinsight_iqa.common.utility import set_gpu_limit
from .utils.keras_utils import TensorBoardBatch
import logging
# set_gpu_limit(10)
logger = logging.getLogger(__name__)

logger.info('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(
    __file__, __name__, str(__package__)))


class Train:
    def __init__(
        self,
        base_model_name: str,
        n_classes: int,
        samples: Union[List, np.ndarray],
        image_dir: str,
        learning_rate_dense: float,
        learning_rate_all: float,
        dropout_rate: float,
        img_format: str = 'jpg',
        decay_dense: int = 0,
        existing_weights: Optional[str] = None,
        batch_size: int = 32,
        decay_all: int = 0,
    ) -> None:
        self.base_model_name = base_model_name
        # build NIMA model and load existing weights if they were provided in config
        self.nima = Nima(base_model_name, n_classes, learning_rate_dense, dropout_rate, decay=decay_dense)
        self.nima.build()

        if existing_weights is not None:
            self.nima.nima_model.load_weights(existing_weights)

        # split samples in train and validation set, and initialize data generators
        samples_train, samples_test = train_test_split(samples, test_size=0.05, shuffle=True, random_state=10207)

        self.training_generator = NimaDataGenerator(
            samples_train,
            image_dir,
            batch_size,
            n_classes,
            self.nima.preprocessing_function(),
            img_format=img_format,
            shuffle=True
        )

        self.validation_generator = NimaDataGenerator(
            samples_test,
            image_dir,
            batch_size,
            n_classes,
            self.nima.preprocessing_function(),
            img_format=img_format,
            shuffle=False,
        )

        self.nima.learning_rate = learning_rate_all
        self.nima.decay = decay_all
        self.nima.compile()
        self.nima.nima_model.summary()

    def _set_layer_prior_bottleneck(self, trainable=False):
        """Set upto bottleneck layers to False """
        # start training only dense layers
        for layer in self.nima.base_model.layers:
            layer.trainable = trainable

    def train(
        self,
        epochs_train_dense: int,
        epochs_train_all: int,
        job_dir: str,
        multiprocessing_data_load: bool = False,
        num_workers_data_load: int = 2,
        **kwargs
    ):

        # initialize callbacks TensorBoardBatch and ModelCheckpoint
        tensorboard = TensorBoardBatch(log_dir=os.path.join(job_dir, 'logs'))

        model_save_name = 'weights_' + self.base_model_name.lower() + '_{epoch:02d}_{val_loss:.3f}.hdf5'
        model_file_path = os.path.join(job_dir, 'weights', model_save_name)
        model_checkpointer = ModelCheckpoint(filepath=model_file_path,
                                             monitor='val_loss',
                                             verbose=1,
                                             save_best_only=True,
                                             save_weights_only=True)
        self._set_layer_prior_bottleneck()
        self.nima.nima_model.fit_generator(generator=self.training_generator,
                                           validation_data=self.validation_generator,
                                           epochs=epochs_train_dense,
                                           verbose=1,
                                           use_multiprocessing=multiprocessing_data_load,
                                           workers=num_workers_data_load,
                                           max_q_size=30,
                                           callbacks=[tensorboard, model_checkpointer])

        # start training all layers
        self._set_layer_prior_bottleneck(trainable=True)

        self.nima.nima_model.fit_generator(generator=self.training_generator,
                                           validation_data=self.validation_generator,
                                           epochs=epochs_train_dense + epochs_train_all,
                                           initial_epoch=epochs_train_dense,
                                           verbose=1,
                                           use_multiprocessing=multiprocessing_data_load,
                                           workers=num_workers_data_load,
                                           max_q_size=30,
                                           callbacks=[tensorboard, model_checkpointer])

        K.clear_session()
