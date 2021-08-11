from typing import Optional, Callable
from .handlers.utils import gradient, optimizer, calculate_error_map
import os
import tensorflow as tf
from ..data_pipeline.diqa_gen import diqa_datagen
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
import pandas as pd
from .handlers.model import Diqa
from .utils.callbacks import TensorBoardBatch
import logging
logger = logging.getLogger()

logger.info(
    '__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(
        __file__, __name__, str(__package__))
)


def generate_random_name(batch_size, epochs):
    import random_name
    model_filename = f"diqa-{random_name.generate_name()}-{batch_size}-{epochs}.h5"
    return model_filename


class TrainWithTFDS:
    def __init__(
        self,
        tfdataset: tf.data.TFRecordDataset,
        image_preprocess: Optional[Callable] = None, 
        epochs: int = 5,
        extra_epochs: int = 1,
        batch_size: int = 16,
        log_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        model_filename: Optional[str] = None,
        custom: bool = False,
        **kwargs
    ):
        base_model_name = kwargs.pop('base_model_name')
        self.image_preprocess = image_preprocess
        self.epochs = epochs
        self.extra_epochs = extra_epochs
        self.log_dir = log_dir
        if not model_filename:
            model_filename = generate_random_name(batch_size, epochs)
        self.model_path = os.path.join(model_dir, model_filename)
        self.batch_size = batch_size
        self.diqa = Diqa(base_model_name, custom=custom)
        self.diqa._build()
        self.tfdataset = tfdataset
        self.final_model = self.diqa.subjective_score_model

    def _training_objective_map(self, model, epochs=1, prefix='objective-model'):
        epoch_accuracy = tf.keras.metrics.MeanSquaredError()
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        tensorboard = TensorBoardBatch(self.log_dir, metrics=[{'loss': train_loss}, {'accuracy': epoch_accuracy}])
        tensorboard.set_model(model)
        opt = optimizer()
        train = self.tfdataset.map(calculate_error_map)

        for epoch in range(epochs):
            step = 0
            for I_d, e_gt, r in train:
                loss_value, gradients = gradient(model, I_d, e_gt, r)
                opt.apply_gradients(zip(gradients, model.trainable_weights))
                train_loss(loss_value)
                epoch_accuracy(e_gt, model(I_d))

                if step % 100 == 0:
                    print('step %s: mean loss = %s' % (step, epoch_accuracy.result()))
                step += 1
            # Invoke tensorflow callback here
            tensorboard.on_epoch_end(epoch)

        # Save model to destination path
        model_path = f"{prefix}-{self.model_path}"
        model.save(model_path)
        return model

    def _train_subjective_map(self, model, epochs=1, prefix='subjective-model'):
        # TODO: Fix error here
        train = self.tfdataset.map(lambda img: (self.image_preprocess(img)))
        tensorboard = TensorBoardBatch(self.log_dir)
        tensorboard.set_model(model)

        history = model.fit(train, epochs=epochs, callbacks=[tensorboard])
        model_path = f"{prefix}-{self.model_path}"
        model.save(model_path)
        return history

    def load_weights(self) -> None:
        self.final_model.load_weights(self.model_path)

    def train(self, use_pretrained: bool = False):
        """
        Train objective Model
        ------------------------
        For the custom training loop, it is necessary to:

        1. Define a metric to measure the performance of the model.
        2. Calculate the loss and the gradients.
        3. Use the optimizer to update the weights.
        4. Print the accuracy.

        Train Subjective Model
        ------------------------
        """
        self.load_weights() if use_pretrained else None
        # -------- OBJECTIVE TRAINING SESSION --------- #
        # Load pre-trained model for objectives error map
        self._training_objective_map(self.diqa.objective_score_model, epochs=self.epochs,)
        
        # -------- SUBJECTIVE TRAINING SESSION --------- #
        # Load pre-trained model for subjetive error map
        self._train_subjective_map(self.final_model, epochs=self.extra_epochs,)


class Train:
    _DATAGEN_MAPPING = {
        "tid2013": diqa_datagen.TID2013DataRowParser,
        "csiq": diqa_datagen.CSIQDataRowParser,
        "live": diqa_datagen.LiveDataRowParser
    }

    def __init__(
        self,
        image_dir: str,
        csv_path: str,
        dataset_type: str, 
        image_preprocess: Optional[Callable] = None, 
        model_dir: Optional[str] = None,
        model_filename: Optional[str] = None,
        epochs: int = 5, batch_size: int = 16,
        multiprocessing_data_load: bool = False,
        extra_epochs: int = 1, num_workers_data_load: int = 1,
        use_pretrained: bool = False,
        log_dir: str = './logs', 
        custom: bool = False,
        **kwargs
    ):
        assert dataset_type in self._DATAGEN_MAPPING.keys(), "Invalid dataset_type, unable to use generator"
        base_model_name = kwargs.pop('base_model_name')
        do_augment = kwargs['do_augment']
        self.csv_path = os.path.join(image_dir, csv_path)
        assert os.path.splitext(self.csv_path)[-1] == '.csv' and os.path.exists(self.csv_path), \
            "Not a valid file extension"
        self.epochs = epochs
        if not model_filename:
            model_filename = generate_random_name(batch_size, epochs)
        self.model_path = os.path.join(model_dir, model_filename)
        self.log_dir = log_dir
        self.multiprocessing_data_load = multiprocessing_data_load
        self.extra_epochs = extra_epochs
        self.img_format = 'jpg'
        self.image_dir = image_dir
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.num_workers_data_load = num_workers_data_load
        self.use_pretrained = use_pretrained
        self.data_gen_cls = self._DATAGEN_MAPPING[self.dataset_type]
        self.diqa = Diqa(base_model_name, custom=custom)
        self.diqa._build()
        self.final_model = self.diqa.subjective_score_model
        # diqa.objective_score_model.summary()

        df = pd.read_csv(self.csv_path).sample(frac=1)
        samples_train, samples_test = df.iloc[:int(len(df) * 0.7), ], df.iloc[int(len(df) * 0.7):, ]

        self.train_generator = self.data_gen_cls(
            samples_train,
            self.image_dir,
            self.batch_size,
            img_preprocessing=image_preprocess,
            do_augment=do_augment,
            shuffle=True
        )

        self.valid_generator = self.data_gen_cls(
            samples_test,
            self.image_dir,
            self.batch_size,
            img_preprocessing=image_preprocess,
            do_augment=do_augment,
            shuffle=False
        )

    def train(self):
        """
        Similar to init_train but use Keras generator for training, we have more control over the API
        with image augmentation

        Train objective Model
        ------------------------
        1. Image augmentation such as (crop, shift and rotation) will useful to check the quality
            is added to get more variance in training output
        2. Split of dataset into training and test subsets
        3. Evaluation metric on the test set
        4. Larger Batch size can be used for training

        Train Subjective Model
        ------------------------

        """

        tensorboard = TensorBoardBatch(log_dir=self.log_dir)
        model_checkpointer = ModelCheckpoint(filepath=self.model_path,
                                             monitor='val_loss',
                                             verbose=1,
                                             save_best_only=True,
                                             save_weights_only=True)

        # Define Metrics
        epoch_accuracy = tf.keras.metrics.MeanSquaredError()
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        _optimizer = optimizer()

        if self.use_pretrained:
            self.final_model.load_weights(self.model_path)

        # ## ------------------------------
        # BEGIN EPOCH
        # ## ------------------------------

        def subjective_datagen(datagen):
            for I_d, I_r, mos in iter(datagen):
                yield I_d, mos

        for epoch in range(self.epochs):
            step = 0
            for I_d, I_r, mos in iter(self.train_generator):
                I_d, e_gt, r = calculate_error_map(I_d, I_r)
                loss_value, gradients = gradient(self.diqa.objective_score_model, I_d, e_gt, r)
                _optimizer.apply_gradients(zip(gradients, self.diqa.objective_score_model.trainable_weights))
                train_loss(loss_value)
                epoch_accuracy(e_gt, self.diqa.objective_score_model(I_d))

                if step % 100 == 0:
                    print('step %s: mean loss = %s' % (step, epoch_accuracy.result()))
            step += 1
            # Invoke tensorflow callback here
            tensorboard.on_epoch_end(epoch)

        # Save the objective model
        self.diqa.objective_score_model.save(self.model_path)

        self.final_model.fit_generator(
            generator=subjective_datagen(self.train_generator),
            steps_per_epoch=100,
            validation_data=subjective_datagen(self.valid_generator),
            validation_steps=100,
            epochs=self.epochs + self.extra_epochs,
            initial_epoch=self.epochs,
            verbose=1,
            use_multiprocessing=self.multiprocessing_data_load,
            workers=self.num_workers_data_load,
            max_q_size=30,
            callbacks=[tensorboard, model_checkpointer]
        )
        # Save the subjective model
        self.final_model.save(self.model_path)
        K.clear_session()
