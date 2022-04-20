import typing as tp
import os
import time
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.applications as KA
import tensorflow.keras.models as KM
from abc import abstractmethod
MODEL_FILE_NAME = "model.h5"
CONFIG_FILE_NAME = "config.yml"

# TODO:
# - Verify if model need to be build
# - Print the keras layer in terminal and plot


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
    def __init__(
        self, *args,
        model_name: str = "diqa_custom",
        shape: tp.Union[tf.TensorShape, tp.Tuple] = (None, None, 1),
        batch_size: int = 1,
        in_layer_name: str = 'original_image', **kwds
    ) -> None:
        super(CustomModel, self).__init__(*args, **kwds)
        self.layers = [
            KL.InputLayer(shape=shape, batch_size=batch_size, name=in_layer_name),
            KL.Conv2D(48, (3, 3), name='Conv1', activation='relu', padding='same'),
            KL.Conv2D(48, (3, 3), name='Conv2', activation='relu', padding='same', strides=(2, 2)),
            KL.Conv2D(64, (3, 3), name='Conv3', activation='relu', padding='same'),
            KL.Conv2D(64, (3, 3), name='Conv4', activation='relu', padding='same', strides=(2, 2)),
            KL.Conv2D(64, (3, 3), name='Conv5', activation='relu', padding='same'),
            KL.Conv2D(64, (3, 3), name='Conv6', activation='relu', padding='same', strides=(2, 2)),
            KL.Conv2D(128, (3, 3), name='Conv7', activation='relu', padding='same'),
            KL.Conv2D(128, (3, 3), name='bottleneck', activation='relu', padding='same', strides=(2, 2))
        ]
        self.model_name = model_name

    def call(self, inputs: tf.Tensor):
        output = self.layers[0](inputs)
        for layer in self.layers[1:]:
            output = layer(output)

        return output


def get_bottleneck(model_name: str, **kwds) -> KM.Model:
    """Get the bottleneck layer given the name

    :param _type_ model_name: _description_
    :raises AttributeError: _description_
    :return _type_: _description_
    """
    try:
        if model_name == "imagenet":
            mapping = \
                {
                    "InceptionV3": KA.InceptionV3,
                    "MobileNet": KA.MobileNet,
                    "InceptionResNetV2": KA.InceptionResNetV2
                }
            model_params = dict(input_shape=(None, None, 3), include_top=False, weights="imagenet")
            if kwds.get('bottleneck') not in mapping.keys():
                raise ValueError("Invalid model_name, enter from given options ")

            model = mapping[kwds.get('bottleneck')](**model_params)

        elif model_name == "diqa_custom":
            model = CustomModel(model_name=model_name, **kwds)

        return model

    except Exception as e:
        raise e


class Diqa(BaseModel):
    def __init__(
        self,
        base_model_name: str,
        shape=(None, None, 1),
        batch_size=1,
        in_layer_name='original_image',
        **kwds
    ) -> None:
        bottleneck = get_bottleneck(
            model_name=base_model_name,
            shape=shape,
            batch_size=batch_size,
            in_layer_name=in_layer_name
        )
        self.custom = True if base_model_name == "diqa_custom" else False
        # Initialize objective and subjective model for training/inference

        self.__objective = ObjectiveModel(bottleneck, custom=self.custom)
        self.__subjective = SubjectiveModel(bottleneck)

    @property
    def objective_model(self):
        return self.__objective

    @property
    def subjective_model(self):
        return self.__subjective

    def call(
        self,
        input_tensor: tf.Tensor,
        objective_output: bool = False
    ):
        """Call the model 

        :param tf.Tensor input_tensor: _description_
        :param bool objective_output: _description_, defaults to False
        """
        return self.__objective(input_tensor) if objective_output \
            else self.__subjective(input_tensor)

    def __get_model_fname(self, saved_path, prefix):
        now = time.time()
        filename, ext = os.path.splitext(MODEL_FILE_NAME)
        model_path = os.path.join(saved_path, f"{prefix}-{filename}-{now}{ext}")
        return model_path

    def save_pretrained(self, saved_path, prefix):
        """Save config and weights to file"""

        os.makedirs(saved_path, exist_ok=True)
        model_path = self.__get_model_fname(saved_path, prefix)

        if prefix == 'objective':
            self.__objective.save_weights(model_path)

        self.__subjective.save_weights(model_path)

    def load_weights(self, saved_path: str, prefix):
        model_path = self.__get_model_fname(saved_path, prefix)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} not found")

        if prefix == 'objective':
            self.__objective.load_weights(model_path)

        self.__subjective.load_weights(model_path)


class ObjectiveModel(KM.Model):
    """
    ## Objective Error Model
    For the training phase, it is convenient to utilize the *tf.data* input pipelines to produce a 
    much cleaner and readable code. The only requirement is to create the function to apply to the input.
    """

    def __init__(self, bottleneck: KM.Model, custom: bool = False) -> None:
        name = 'objective_error_map'
        self.custom = custom
        self.loss = (lambda name: tf.keras.metrics.Mean(name, dtype=tf.float32))
        self.accuracy = (lambda name: tf.keras.metrics.MeanSquaredError(name))
        self.optimizer = tf.optimizers.Nadam(learning_rate=2 * 10 ** -4)

        self.layers = [bottleneck]
        if self.custom:
            self.layers.append(KL.Conv2D(1, (1, 1), name='final', padding='same', activation='linear'))
        else:
            self.layers += [
                KL.Conv2D(512, (1, 1), use_bias=False, activation='relu'),
                KL.Conv2D(256, (1, 1), use_bias=False, activation='relu'),
                KL.Conv2D(128, (1, 1), use_bias=False, activation='relu', name='bottleneck'),
                KL.Conv2D(1, (1, 1), name='final', use_bias=False, activation='relu')
            ]

    def call(self, input_tensor: tf.Tensor):
        output_tensor = self.layers[0](input_tensor)

        for layer in self.layers[1:]:
            output_tensor = layer(output_tensor)

        return output_tensor


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
        self.name = 'subjective_error_map'
        self.optimizer = tf.optimizers.Nadam(learning_rate=2 * 10 ** -4)
        self.bottleneck = bottleneck
        self.layers = [
            bottleneck,
            KL.GlobalAveragePooling2D(data_format='channels_last'),
            KL.Dense(128, activation='relu'),
            KL.Dense(128, activation='relu'),
            KL.Dense(1)
        ]

    def compile(self):
        self.model.compile(
            optimizer=self.optimizer,
            loss=tf.losses.MeanSquaredError(),
            metrics=[tf.metrics.MeanSquaredError()]
        )
        return self

    def call(self, input_tensor: tf.Tensor):
        output_tensor = self.layers[0](input_tensor)

        for layer in self.layers[1:]:
            output_tensor = layer(output_tensor)

        return output_tensor
