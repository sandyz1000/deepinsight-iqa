
import keras.backend as K
import tensorflow as tf
from keras.callbacks import TensorBoard


class TensorBoardBatch(TensorBoard):
    def __init__(self, *args, **kwargs):
        super(TensorBoardBatch, self).__init__(*args, **kwargs)
        self.batch_counter = 0

    def on_batch_end(self, batch, logs=None):
        self.batch_counter += 1
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.batch_counter)

        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)

        self.writer.flush()
