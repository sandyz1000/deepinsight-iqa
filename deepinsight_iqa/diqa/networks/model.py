import typing as tp
import os
import time
from pathlib import Path
import tensorflow as tf
# import tensorflow.keras.layers as KL
# import tensorflow.keras.applications as KA
# import tensorflow.keras.models as KM
# from tensorflow.keras import metrics as KMetric
import keras.layers as KL
import keras.applications as KA
import keras.models as KM
from keras import metrics as KMetric

from abc import abstractmethod
from .. import (
    CUSTOM_MODEL_TYPE,
    IMAGENET_MODEL_TYPE,
    MODEL_FILE_NAME,
    CONFIG_FILE_NAME,
    SUBJECTIVE_NW,
    OBJECTIVE_NW,
    DTF_DATETIMET
)
from .utils import gradient, calculate_error_map, loss_fn, SpearmanCorrMetric


def generate_random_name(batch_size, epochs):
    import random_name
    model_filename = f"diqa-{random_name.generate_name()}-{batch_size}-{epochs}.h5"
    return model_filename


class BaseModel(KM.Model):
    @abstractmethod
    def save_pretrained(self, saved_path: str, prefix):
        """_summary_

        :param _type_ saved_path: _description_
        :param _type_ prefix: _description_
        """

    @abstractmethod
    def load_weights(self, model_path: str, prefix):
        """_summary_

        :param _type_ model_path: _description_
        :param _type_ prefix: _description_
        """


class CustomModel(tf.Module):
    def __init__(self, **kwds):
        super(CustomModel, self).__init__()
        self.kwds = kwds
        self.conv1 = KL.Conv2D(48, (3, 3), name='Conv1', activation='relu', padding='same')
        self.conv2 = KL.Conv2D(48, (3, 3), name='Conv2', activation='relu', padding='same', strides=(2, 2))
        self.conv3 = KL.Conv2D(64, (3, 3), name='Conv3', activation='relu', padding='same')
        self.conv4 = KL.Conv2D(64, (3, 3), name='Conv4', activation='relu', padding='same', strides=(2, 2))
        self.conv5 = KL.Conv2D(64, (3, 3), name='Conv5', activation='relu', padding='same')
        self.conv6 = KL.Conv2D(64, (3, 3), name='Conv6', activation='relu', padding='same', strides=(2, 2))
        self.conv7 = KL.Conv2D(128, (3, 3), name='Conv7', activation='relu', padding='same')
        self.conv8 = KL.Conv2D(128, (3, 3), name='bottleneck', activation='relu', padding='same', strides=(2, 2))

    @tf.function
    def __call__(self, inputs: tf.Tensor):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)

        return out


def get_bottleneck(
    model_type: str, *,
    bottleneck: str = None,
    train_bottleneck: bool = True,
    kwds={}
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
        if bottleneck not in mapping.keys():
            raise ValueError("Invalid model_name, enter from given options ")

        model = mapping[bottleneck](**model_params)  # type: KM.Model

    elif model_type == CUSTOM_MODEL_TYPE:
        model = CustomModel(**kwds)  # type: KM.Model

    else:
        raise AttributeError("Invalid model options ", {model_type})

    model.trainable = train_bottleneck
    return model


class Diqa(BaseModel):
    def __init__(
        self,
        model_type: str,
        bn_layer: str,
        train_bottleneck=False,
        objective_fn: KM.Model = None,
        subjective_fn: KM.Model = None,
        is_training: bool = False,
        kwds={}
    ) -> None:
        super(Diqa, self).__init__()
        bottleneck = get_bottleneck(
            model_type=model_type,
            bottleneck=bn_layer,
            train_bottleneck=train_bottleneck,
            kwds=kwds
        )

        self.model_type = model_type
        self.custom = True if model_type == CUSTOM_MODEL_TYPE else False
        # Initialize objective and subjective model for training/inference

        if objective_fn is None:
            self.objective = ObjectiveModel(bottleneck, custom=self.custom)
        else:
            self.objective = objective_fn
        if subjective_fn is None:
            self.subjective = SubjectiveModel(bottleneck)
        else:
            self.subjective = subjective_fn
        
        self.loss_metric = KMetric.Mean(name=f'loss', dtype=tf.float32)
        self.acc1_metric = KMetric.MeanSquaredError(name=f'accuracy', dtype=tf.float32)
        self.acc2_metric = SpearmanCorrMetric(name=f'accuracy', dtype=tf.float32)
        self._metrics = [self.loss_metric, self.acc1_metric, self.acc2_metric]
    
    def call(self, inputs: tf.Tensor):
        """Call the model

        :param tf.Tensor input_tensor: _description_
        :param bool is_objective: _description_, defaults to False
        """
        if self.__out_state == OBJECTIVE_NW:
            return self.objective(inputs)

        return self.subjective(inputs)

    @property
    def metrics(self):
        return self._metrics

    def compile(self, optimizer, loss_fn, metrics=[], out_state=None):
        """State output is set to subjective by default, re-compile if you want
        the model to return objective output

        :param _type_ optimizer: _description_
        :param _type_ loss_fn: _description_
        :param _type_ metrics: _description_, defaults to []
        :param _type_ out_state: _description_, defaults to None
        """
        super(Diqa, self).compile()
        self.metrics = metrics
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.__out_state = out_state

    def train_step(self, data):
        distorted, reference, mos = data
        
        if self.__out_state == OBJECTIVE_NW:
            reference = tf.slice(reference, begin=[0, 0, 0, 0], size=(reference.shape[:-1] + [1]))
            distorted_gray = tf.slice(distorted, begin=[0, 0, 0, 0], size=(distorted.shape[:-1] + [1]))
            e_gt, r = calculate_error_map(distorted_gray, reference, scaling_factor=self.scaling_factor)
            if self.is_training:
                loss_value, gradients = gradient(self.objective, distorted, e_gt, r)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
            else:
                loss_value = self.loss_fn(self.model, distorted, e_gt, r)

            err_pred = self.objective(distorted)
            _shape = tf.TensorShape([e_gt.shape[0], tf.reduce_prod(e_gt.shape[1:]).numpy()])

            err_pred = tf.reshape(err_pred, shape=_shape)
            e_gt = tf.reshape(e_gt, shape=_shape)
            
            self.loss_metric.update_state(loss_value)
            self.acc1_metric.update_state(e_gt, err_pred)

            # return loss, acc
            return {
                "accuracy": self.acc1_metric.result(),
                "loss": self.loss_metric.result(),
            }
        else:
            if self.is_training:
                # Train here subjective loss
                with tf.GradientTape() as tape:
                    preds = self.subjective(distorted)
                    g_loss = self.loss_fn(preds, mos)
                grads = tape.gradient(g_loss, self.subjective.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
            else:
                g_loss = self.loss_fn(preds, mos)

            self.loss_metric.update_state(g_loss)
            self.acc1_metric.update_state(mos, preds)

            # return loss, acc
            return {
                "accuracy": self.acc1_metric.result(),
                "loss": self.loss_metric.result(),
            }

    @classmethod
    def __get_model_fname(cls, saved_path, prefix, model_type):
        now = time.strftime(DTF_DATETIMET)
        filename, ext = os.path.splitext(MODEL_FILE_NAME)
        model_path = os.path.join(saved_path, f"{prefix}-{filename}-{model_type}-{now}{ext}")
        return model_path

    @classmethod
    def save_pretrained(cls, saved_path, prefix=None, model_type=None, model_path=None):
        """Save config and weights to file"""

        os.makedirs(saved_path, exist_ok=True)
        model_path = model_path or cls.__get_model_fname(saved_path, prefix)
        model = cls.build()
        if prefix == OBJECTIVE_NW:
            model.objective_net.save_weights(model_path)
        else:
            model.subjective_net.save_weights(model_path)

    def load_weights(self, model_dir: Path, model_path: Path, prefix):

        if not os.path.exists(model_path):
            model_path = Path(model_dir) / model_path

            assert model_path.exists(), \
                FileNotFoundError(f"Model path {model_path} not found")

        if prefix == OBJECTIVE_NW:
            self.objective.load_weights(model_path)
        else:
            self.subjective.load_weights(model_path)


class ObjectiveModel(KM.Model):
    """
    ## Objective Error Model AKA "objective_error_map"
    For the training phase, it is convenient to utilize the *tf.data* input pipelines to produce a 
    much cleaner and readable code. The only requirement is to create the function to apply to the input.
    """

    def __init__(self, bottleneck: KM.Model, custom: bool = False) -> None:
        super(ObjectiveModel, self).__init__()
        self.custom = custom
        self.bottleneck = bottleneck
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


class SubjectiveModel(KM.Model):
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

    def __init__(self, bottleneck: KM.Model) -> None:
        super(SubjectiveModel, self).__init__()
        # self.name = 'subjective_error_map'
        self.bottleneck = bottleneck
        self.gap = KL.GlobalAveragePooling2D(data_format='channels_last')
        self.dense1 = KL.Dense(128, activation='relu')
        self.dense2 = KL.Dense(128, activation='relu')
        self.final = KL.Dense(1)

    def call(self, input_tensor: tf.Tensor):
        out = self.bottleneck(input_tensor)

        out = self.gap(out)
        out = self.dense1(out)
        out = self.dense2(out)
        return self.final(out)
