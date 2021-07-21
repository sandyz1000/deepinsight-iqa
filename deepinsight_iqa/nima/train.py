import os
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(
    __file__, __name__, str(__package__)))

from ..data_pipeline.nima_gen.nima_datagen import NimaDataGenerator
from .handlers.model_builder import Nima
from .keras_utils import TensorBoardBatch


def train(base_model_name,
          n_classes,
          samples,
          image_dir,
          batch_size,
          epochs_train_dense,
          epochs_train_all,
          learning_rate_dense,
          learning_rate_all,
          dropout_rate,
          job_dir,
          img_format='jpg',
          existing_weights=None,
          multiprocessing_data_load=False,
          num_workers_data_load=2,
          decay_dense=0,
          decay_all=0,
          **kwargs):

    # build NIMA model and load existing weights if they were provided in config
    nima = Nima(base_model_name, n_classes, learning_rate_dense, dropout_rate, decay=decay_dense)
    nima.build()

    if existing_weights is not None:
        nima.nima_model.load_weights(existing_weights)

    # split samples in train and validation set, and initialize data generators
    samples_train, samples_test = train_test_split(samples, test_size=0.05, shuffle=True, random_state=10207)

    training_generator = NimaDataGenerator(samples_train,
                                           image_dir,
                                           batch_size,
                                           n_classes,
                                           nima.preprocessing_function(),
                                           img_format=img_format,
                                           shuffle=True)

    validation_generator = NimaDataGenerator(samples_test,
                                             image_dir,
                                             batch_size,
                                             n_classes,
                                             nima.preprocessing_function(),
                                             img_format=img_format,
                                             shuffle=False,)

    # initialize callbacks TensorBoardBatch and ModelCheckpoint
    tensorboard = TensorBoardBatch(log_dir=os.path.join(job_dir, 'logs'))

    model_save_name = 'weights_' + base_model_name.lower() + '_{epoch:02d}_{val_loss:.3f}.hdf5'
    model_file_path = os.path.join(job_dir, 'weights', model_save_name)
    model_checkpointer = ModelCheckpoint(filepath=model_file_path,
                                         monitor='val_loss',
                                         verbose=1,
                                         save_best_only=True,
                                         save_weights_only=True)

    # start training only dense layers
    for layer in nima.base_model.layers:
        layer.trainable = False

    nima.compile()
    nima.nima_model.summary()

    nima.nima_model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  epochs=epochs_train_dense,
                                  verbose=1,
                                  use_multiprocessing=multiprocessing_data_load,
                                  workers=num_workers_data_load,
                                  max_q_size=30,
                                  callbacks=[tensorboard, model_checkpointer])

    # start training all layers
    for layer in nima.base_model.layers:
        layer.trainable = True

    nima.learning_rate = learning_rate_all
    nima.decay = decay_all
    nima.compile()
    nima.nima_model.summary()

    nima.nima_model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  epochs=epochs_train_dense + epochs_train_all,
                                  initial_epoch=epochs_train_dense,
                                  verbose=1,
                                  use_multiprocessing=multiprocessing_data_load,
                                  workers=num_workers_data_load,
                                  max_q_size=30,
                                  callbacks=[tensorboard, model_checkpointer])

    K.clear_session()
