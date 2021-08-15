import tensorflow as tf
import keras.backend as K
from keras.callbacks import TensorBoard


class TensorBoardBatch(TensorBoard):
    def __init__(self, *args, **kwargs):
        super(TensorBoardBatch, self).__init__(*args, **kwargs)
        self.train_loss = args['metrics'].get('loss', None)
        self.train_accuracy = args['metrics'].get('accuracy', None)
        self.train_loss = args['metrics'].get('val_loss', None)
        self.train_accuracy = args['metrics'].get('val_accuracy', None)
        self.batch_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        with self._train_writer.as_default():
            tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', self.train_accuracy.result(), step=epoch)

        with self._val_writer.as_default():
            tf.summary.scalar('loss', self.valid_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', self.valid_accuracy.result(), step=epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            with self._train_writer.as_default():
                tf.summary.scalar(name, value, step=epoch)

        self._train_writer.flush()
