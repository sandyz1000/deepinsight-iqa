print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(
    __file__, __name__, str(__package__)))
from .model import Diqa
from .utils import (gradient, optimizer, calculate_subjective_score,
                    rescale, average_reliability_map, error_map)
from .callbacks import TensorBoardBatch
import tensorflow as tf


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
