from .model import Diqa
from .utils import (gradient, optimizer, calculate_subjective_score,
                    rescale, average_reliability_map, error_map)
from .callbacks import TensorBoardBatch
import tensorflow as tf
from ..data_pipeline.diqa_gen import diqa_datagen
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
import pandas as pd
from .model import Diqa
from .utils import image_preprocess, gradient, optimizer, rescale, error_map, average_reliability_map
from .callbacks import TensorBoardBatch
import logging
logger = logging.getLogger()

logger.info('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(
    __file__, __name__, str(__package__)))


def calculate_error_map(I_d, I_r, SCALING_FACTOR=1 / 32):
    r = rescale(average_reliability_map(I_d, 0.2), SCALING_FACTOR)
    e_gt = rescale(error_map(I_r, I_d, 0.2), SCALING_FACTOR)
    return (I_d, e_gt, r)


class TrainDeepIMAWithTFDS:
    def __init__(self, epochs=5, extra_epochs=1, batch_size=16, log_dir=None, model_path=None, ):
        self.epochs = epochs
        self.extra_epochs = extra_epochs
        self.log_dir = log_dir
        self.model_path = model_path
        self.batch_size = batch_size

    def _training_objective_map(self, model, train, prefix='objective-model'):
        epoch_accuracy = tf.keras.metrics.MeanSquaredError()
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        tensorboard = TensorBoardBatch(self.log_dir, metrics=[{'loss': train_loss}, {'accuracy': epoch_accuracy}])
        tensorboard.set_model(model)
        opt = optimizer()

        for epoch in range(self.epochs):
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

    def _train_subjective_map(self, model, train, epochs=1, log_dir=None, model_path=None, prefix='subjective-model'):
        tensorboard = TensorBoardBatch(log_dir)
        tensorboard.set_model(model)

        history = model.fit(train, epochs=epochs, callbacks=[tensorboard])
        model_path = f"{prefix}-{model_path}"
        model.save(model_path)
        return history

    def init_train(self, tfdataset, custom=False, use_pretrained=False):
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
        # from functools import partial
        diqa = Diqa(custom=custom)

        # -------- OBJECTIVE TRAINING SESSION --------- #
        # Load pre-trained model for objectives error map
        train = tfdataset.map(calculate_error_map)
        if not use_pretrained:
            self._training_objective_map(diqa.objective_score_model,
                                         train, epochs=self.epochs,
                                         log_dir=self.log_dir, model_path=self.model_path)
        else:
            diqa.objective_score_model.load_weights(self.model_path)

        # -------- SUBJECTIVE TRAINING SESSION --------- #
        # Load pre-trained model for subjetive error map
        train = tfdataset.map(calculate_subjective_score)
        if not use_pretrained:
            self._train_subjective_map(diqa.subjective_score_model,
                                       train, epochs=self.extra_epochs,
                                       log_dir=self.log_dir, model_path=self.model_path)
        else:
            diqa.subjective_score_model.load_weights(self.model_path)

        return diqa.subjective_score_model




class TrainDeepIMAWithGenerator:
    _DATAGEN_MAPPING = {
        "tid2013": diqa_datagen.TID2013DataRowParser,
        "csiq": diqa_datagen.CSIQDataRowParser,
        "liva": diqa_datagen.LiveDataRowParser
    }

    def __init__(self, image_dir, csv_path,
                 model_path=None,
                 epochs=5, batch_size=16,
                 multiprocessing_data_load=False,
                 extra_epochs=1, num_workers_data_load=1,
                 dataset_type="tid2013", use_pretrained=False, log_dir='./logs'):
        self.epochs = epochs
        self.model_path = model_path
        self.log_dir = log_dir
        self.multiprocessing_data_load = multiprocessing_data_load
        self.extra_epochs = extra_epochs
        self.img_format = 'jpg'
        self.image_dir = image_dir
        self.dataset_type = dataset_type
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers_data_load = num_workers_data_load
        self.use_pretrained = use_pretrained

    def keras_init_train(self, custom=False):
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
        diqa = Diqa(custom=custom)
        assert self.csv_path.split(".")[-1:] == 'csv', "Not a valid file extension"

        df = pd.read_csv(self.csv_path)
        samples_train, samples_test = df.iloc[:len(df) * 0.7, ], df.iloc[len(df) * 0.7:, ]
        data_gen_cls = self._DATAGEN_MAPPING[self.dataset_type]
        train_generator = data_gen_cls(samples_train,
                                       self.image_dir,
                                       self.batch_size,
                                       img_preprocessing=image_preprocess,
                                       shuffle=True)

        valid_generator = data_gen_cls(samples_test,
                                       self.image_dir,
                                       self.batch_size,
                                       img_preprocessing=image_preprocess,
                                       shuffle=False)

        # initialize callbacks TensorBoardBatch and ModelCheckpoint
        tensorboard = TensorBoardBatch(log_dir=self.log_dir)
        model_checkpointer = ModelCheckpoint(filepath=self.model_path,
                                             monitor='val_loss',
                                             verbose=1,
                                             save_best_only=True,
                                             save_weights_only=True)

        # diqa.objective_score_model.summary()

        # Define Metrics
        epoch_accuracy = tf.keras.metrics.MeanSquaredError()
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        _optimizer = optimizer()

        final_model = diqa.subjective_score_model()
        if self.use_pretrained:
            final_model.load_weights(self.model_path)

        # ## ------------------------------
        # BEGIN EPOCH
        # ## ------------------------------
        def calculate_error_map(I_d, I_r, SCALING_FACTOR=1 / 32):
            r = rescale(average_reliability_map(I_d, 0.2), SCALING_FACTOR)
            e_gt = rescale(error_map(I_r, I_d, 0.2), SCALING_FACTOR)
            return (I_d, e_gt, r)

        objective_datagen = (lambda train_generator: [(yield calculate_error_map(I_d, I_r))
                                                      for I_d, I_r, mos in train_generator.flow()])
        subjective_datagen = (lambda train_generator: [(yield I_d, mos)
                                                       for I_d, I_r, mos in train_generator.flow()])

        for epoch in range(self.epochs):
            step = 0
            for I_d, e_gt, r in objective_datagen(train_generator):
                loss_value, gradients = gradient(diqa.objective_score_model, I_d, e_gt, r)
                _optimizer.apply_gradients(zip(gradients, diqa.objective_score_model.trainable_weights))
                train_loss(loss_value)
                epoch_accuracy(e_gt, diqa.objective_score_model(I_d))

                if step % 100 == 0:
                    print('step %s: mean loss = %s' % (step, epoch_accuracy.result()))
            step += 1
            # Invoke tensorflow callback here
            tensorboard.on_epoch_end(epoch)

        # Save the objective model
        diqa.objective_score_model.save(self.model_path)

        final_model.fit_generator(generator=subjective_datagen,
                                  steps_per_epoch=100,
                                  validation_data=valid_generator,
                                  validation_steps=100,
                                  epochs=self.epochs + self.extra_epochs,
                                  initial_epoch=self.epochs,
                                  verbose=1,
                                  use_multiprocessing=self.multiprocessing_data_load,
                                  workers=self.num_workers_data_load,
                                  max_q_size=30,
                                  callbacks=[tensorboard, model_checkpointer])
        # Save the subjective model
        final_model.save(self.model_path)
        K.clear_session()
