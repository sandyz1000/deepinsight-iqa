import os
import time
import tensorflow as tf
from pathlib import Path
import tensorflow.keras.layers as KL
import tensorflow.keras.applications as KA
import tensorflow.keras.models as KM
from tensorflow.keras import losses as KLosses
from tensorflow.keras import metrics as KMetric

# import keras.layers as KL
# import keras.applications as KA
# import keras.models as KM
# from keras import metrics as KMetric

from .. import (
    CUSTOM_MODEL_TYPE,
    IMAGENET_MODEL_TYPE,
    DTF_DATETIMET
)
from .utils import gradient, calculate_error_map, SpearmanCorrMetric


def generate_random_name(batch_size, epochs):
    import random_name
    model_filename = f"diqa-{random_name.generate_name()}-{batch_size}-{epochs}.h5"
    return model_filename


def get_bottleneck(
    model_type: str, *,
    bn_layer: str = None,
    train_bottleneck: bool = True
) -> KM.Model:
    """Get the bottleneck layer given the name

    :param _type_ model_name: _description_
    :raises AttributeError: _description_
    :return _type_: _description_
    """
    mapping = \
        {
            "InceptionV3": KA.InceptionV3,
            "MobileNet": KA.MobileNet,
            "InceptionResNetV2": KA.InceptionResNetV2
        }

    if model_type == IMAGENET_MODEL_TYPE:

        model_params = dict(input_shape=(None, None, 3), include_top=False, weights="imagenet")
        if bn_layer not in mapping.keys():
            raise ValueError("Invalid model_name, enter from given options ")

        model = mapping[bn_layer](**model_params)  # type: KM.Model

    elif model_type == CUSTOM_MODEL_TYPE:
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(None, None, 1), batch_size=1, name='original_image'),
                
                KL.Conv2D(48, (3, 3), name='Conv1', activation='relu', padding='same'),
                KL.Conv2D(48, (3, 3), name='Conv2', activation='relu', padding='same', strides=(2, 2)),
                KL.Conv2D(64, (3, 3), name='Conv3', activation='relu', padding='same'),
                KL.Conv2D(64, (3, 3), name='Conv4', activation='relu', padding='same', strides=(2, 2)),
                KL.Conv2D(64, (3, 3), name='Conv5', activation='relu', padding='same'),
                KL.Conv2D(64, (3, 3), name='Conv6', activation='relu', padding='same', strides=(2, 2)),
                KL.Conv2D(128, (3, 3), name='Conv7', activation='relu', padding='same'),
                KL.Conv2D(128, (3, 3), name='bottleneck', activation='relu', padding='same', strides=(2, 2))
            ],
            name=model_type,
        )  # type: KM.Model

    else:
        raise AttributeError("Invalid model options ", {model_type})

    model.trainable = train_bottleneck
    return model


class DiqaMixin:
    
    def compile(self, optimizer=None, loss_fn=None, **kwds):
        """State output is set to subjective by default, re-compile if you want
        the model to return objective output

        :param _type_ optimizer: _description_
        :param _type_ loss_fn: _description_
        :param _type_ metrics: _description_, defaults to []
        :param _type_ out_state: _description_, defaults to None
        """
        super().compile(**kwds)
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def build(self):
        super().build(input_shape=self._input_shape)

    def _get_model_fname(self, model_dir, prefix, model_type):
        now = time.strftime(DTF_DATETIMET)
        pathfmt = "{}-{}-{}".format(prefix, model_type, now)
        model_path = Path(model_dir, pathfmt)
        return model_path

    def load_pretrained(self, model_dir, model_path: Path = None):
        """Helper function to verify and load weight in safe manner

        :param _type_ diqa: _description_
        :param _type_ model_path: _description_, defaults to None
        :raises FileNotFoundError: _description_
        """
        assert model_path and isinstance(model_path, Path), TypeError("Should be of type pathlib.Path")
        
        if not model_path.exists():
            model_path = Path(model_dir) / model_path

        if model_path.exists():
            self.load_weights(model_path.as_posix() + '/')
        else:
            raise FileNotFoundError(f"Model path {model_path} not found")

    def save_pretrained(self, model_dir, prefix='subjective', model_path: Path = None):
        """Save config and weights to file"""

        os.makedirs(model_dir, exist_ok=True)
        model_path = model_path or self._get_model_fname(model_dir, prefix, self.model_type)
        self.save_weights(model_path.as_posix() + '/', save_format='tf')


@tf.keras.utils.register_keras_serializable()
class ObjectiveModel(DiqaMixin, KM.Model):
    """
    ## Objective Error Model AKA "objective_error_map"
    For the training phase, it is convenient to utilize the *tf.data* input pipelines to produce a 
    much cleaner and readable code. The only requirement is to create the function to apply to the input.
    """

    def __init__(self, bottleneck: KM.Model, scaling_factor, custom: bool = False, kwds={}) -> None:
        super().__init__()
        self.custom = custom
        self.model_type = kwds
        self.bottleneck = bottleneck
        self.loss_fn = None
        self.optimizer = None
        self.scaling_factor = scaling_factor

        self._input_shape = self.bottleneck.input_shape

        self.loss_metric = KMetric.Mean(name=f'loss', dtype=tf.float32)
        self.ms_metric = KMetric.MeanSquaredError(name=f'accuracy-mean', dtype=tf.float32)
        self.corr_metric = SpearmanCorrMetric(name=f'accuracy-corr', dtype=tf.float32)

        if self.custom:
            self.final = KL.Conv2D(1, (1, 1), name='final', padding='same', activation='linear')
        else:
            self.conv1 = KL.Conv2D(512, (1, 1), use_bias=False, activation='relu')
            self.conv2 = KL.Conv2D(256, (1, 1), use_bias=False, activation='relu')
            self.conv3 = KL.Conv2D(128, (1, 1), use_bias=False, activation='relu', name='bottleneck')
            self.final = KL.Conv2D(1, (1, 1), name='final', use_bias=False, activation='relu')

        print(">>> Created objective model >>>")

    def call(self, input_tensor: tf.Tensor):
        out = self.bottleneck(input_tensor)

        if self.custom:
            out = self.final(out)
        else:
            out = self.conv1(out)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.final(out)

        return out

    @property
    def metrics(self):
        return [self.loss_metric, self.ms_metric, self.corr_metric]

    @tf.function
    def test_step(self, data):
        return self.o_step(data)
            
    @tf.function
    def train_step(self, data):
        return self.o_step(data, train_step=True)

    @tf.function
    def o_step(self, data, train_step=False):
        (dist, dist_gray, ref_gray), mos = data

        e_gt, r = calculate_error_map(dist_gray, ref_gray, scaling_factor=self.scaling_factor)
        if train_step:
            loss_value, gradients = gradient(self, dist, e_gt, r)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        else:
            loss_value = self.loss_fn(self, dist, e_gt, r)

        err_pred = self(dist)
        
        self.loss_metric.update_state(loss_value)
        self.ms_metric.update_state(e_gt, err_pred)

        return {
            "accuracy": self.ms_metric.result(),
            "loss": self.loss_metric.result(),
        }

    def reset_state(self):
        self.ms_metric.reset_states()
        self.loss_metric.reset_states()
        # self.corr_metric.reset_states()


@tf.keras.utils.register_keras_serializable()
class SubjectiveModel(DiqaMixin, KM.Model):
    """
    ## Subjective Model

    *Note: It would be a good idea to use the Spearman's rank-order correlation coefficient (SRCC) or 
    Pearson's linear correlation coefficient (PLCC) as accuracy metrics.*

    # Subjective Score Model

    To create the subjective score model, let's use the output of f(·) to train a regressor.

    Training a model with the fit method of *tf.keras.Model* expects a dataset that returns two arguments. 
    The first one is the input, and the second one is the target.

    Arguments:
            ds {[type]} -- [description]
            feature_map {[type]} -- [description]

    Keyword Arguments:
            use_pretrained {bool} -- [description] (default: {False})
    """

    def __init__(self, bottleneck: KM.Model, kwds={}) -> None:
        super().__init__()
        self.model_type = kwds
        self.bottleneck = bottleneck
        self.loss_fn = None
        self.optimizer = None
        self._input_shape = self.bottleneck.input_shape

        self.loss_metric = KMetric.Mean(name=f'loss', dtype=tf.float32)
        self.ms_metric = KMetric.MeanSquaredError(name=f'accuracy-mean', dtype=tf.float32)

        self.gap = KL.GlobalAveragePooling2D(data_format='channels_last')
        self.dense1 = KL.Dense(128, activation='relu')
        self.dense2 = KL.Dense(128, activation='relu')
        self.final = KL.Dense(1)

    @property
    def metrics(self):
        return [self.loss_metric, self.ms_metric]

    def call(self, input_tensor: tf.Tensor):
        out = self.bottleneck(input_tensor)

        out = self.gap(out)
        out = self.dense1(out)
        out = self.dense2(out)
        return self.final(out)

    @tf.function
    def test_step(self, data):
        return self.s_step(data)
    
    @tf.function
    def train_step(self, data):
        return self.s_step(data, train_step=True)
        
    @tf.function
    def s_step(self, data, train_step=False):
        (distorted, dist_gray, reference), mos = data
        if train_step:
            # Train here subjective loss
            with tf.GradientTape() as tape:
                preds = self(distorted)
                g_loss = self.loss_fn(preds, mos)
            grads = tape.gradient(g_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        else:
            preds = self(distorted)
            g_loss = self.loss_fn(preds, mos)

        self.loss_metric.update_state(g_loss)
        self.ms_metric.update_state(mos, preds)

        return {
            "accuracy": self.ms_metric.result(),
            "loss": self.loss_metric.result(),
        }

    def reset_state(self):
        self.ms_metric.reset_states()
        self.loss_metric.reset_states()