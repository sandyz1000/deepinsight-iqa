import typing as tp
import os
import time
from pathlib import Path
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.applications as KA
import tensorflow.keras.models as KM
from abc import abstractmethod
from .. import (
    CUSTOM_MODEL_TYPE,
    IMAGENET_MODEL_TYPE,
    MODEL_FILE_NAME,
    CONFIG_FILE_NAME,
    SUBJECTIVE_NW,
    OBJECTIVE_NW
)


class BaseModel(tf.keras.Model):
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


class CustomModel(KM.Model):
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

    def call(self, inputs: tf.Tensor):
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
        optimizer=None,
        train_bottleneck=False,
        kwds={}
    ) -> None:
        super(Diqa, self).__init__()
        self.optimizer = optimizer
        bottleneck = get_bottleneck(
            model_type=model_type,
            bottleneck=bn_layer,
            train_bottleneck=train_bottleneck,
            kwds=kwds
        )
        self.custom = True if model_type == CUSTOM_MODEL_TYPE else False
        # Initialize objective and subjective model for training/inference

        self.__objective_net = ObjectiveModel(bottleneck, custom=self.custom)
        self.__subjective_net = SubjectiveModel(bottleneck)

    @property
    def objective_model(self):
        return self.__objective_net

    @property
    def subjective_model(self):
        return self.__subjective_net

    def call(
        self,
        input_tensor: tf.Tensor,
        objective_output: bool = False
    ):
        """Call the model 

        :param tf.Tensor input_tensor: _description_
        :param bool objective_output: _description_, defaults to False
        """
        return self.__objective_net(input_tensor) if objective_output \
            else self.__subjective_net(input_tensor)

    def __get_model_fname(self, saved_path, prefix):
        now = time.time()
        filename, ext = os.path.splitext(MODEL_FILE_NAME)
        model_path = os.path.join(saved_path, f"{prefix}-{filename}-{now}{ext}")
        return model_path

    def save_pretrained(self, saved_path, prefix=None):
        """Save config and weights to file"""

        os.makedirs(saved_path, exist_ok=True)
        model_path = self.__get_model_fname(saved_path, prefix)

        if prefix == OBJECTIVE_NW:
            self.__objective_net.save_weights(model_path)
        else:
            self.__subjective_net.save_weights(model_path)

    def load_weights(self, model_dir: Path, model_path: Path, prefix):

        if not os.path.exists(model_path):
            model_path = Path(model_dir) / model_path

            assert model_path.exists(), \
                FileNotFoundError(f"Model path {model_path} not found")

        if prefix == OBJECTIVE_NW:
            self.__objective_net.load_weights(model_path)
        else:
            self.__subjective_net.load_weights(model_path)


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
