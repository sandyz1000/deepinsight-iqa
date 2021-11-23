import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.applications as KA


class Diqa(object):
    def __init__(self, base_model_name, custom=False) -> None:
        # Initialize objective model for training
        self.__objective_network = ObjectiveNetwork(base_model_name, custom=custom)
        self.__subjective_network = SubjectiveNetwork()
        self.__subjective_model = None
        self.__objective_model = None

    @property
    def subjective(self):
        return self.__subjective_model

    @property
    def objective(self):
        return self.__objective_model

    def _build(self):
        self.__objective_model = self.__objective_network().model
        self.__subjective_model = self.__subjective_network(
            self.objective.input, 
            self.objective.get_layer('bottleneck').output
        ).model


class ObjectiveNetwork:
    def __init__(self, base_model_name, custom=False) -> None:
        self.model = None
        self.base_model_name = base_model_name
        self.custom = custom
        self.loss = (lambda name: tf.keras.metrics.Mean(name, dtype=tf.float32))
        self.accuracy = (lambda name: tf.keras.metrics.MeanSquaredError(name))
        self.optimizer = tf.optimizers.Nadam(learning_rate=2 * 10 ** -4)

    def _get_base_module(self):
        if self.base_model_name == 'InceptionV3':
            model = KA.InceptionV3(input_shape=(None, None, 3),
                                   include_top=False, weights="imagenet")
        elif self.base_model_name == 'MobileNet':
            model = KA.MobileNetV2(input_shape=(None, None, 3),
                                   include_top=False, weights="imagenet")

        elif self.base_model_name == "InceptionResNetV2":
            model = KA.InceptionResNetV2(input_shape=(None, None, 3),
                                         include_top=False, weights="imagenet")
        elif self.base_model_name == "custom":
            input_ = tf.keras.Input(shape=(None, None, 1), batch_size=1, name='original_image')
            f = KL.Conv2D(48, (3, 3), name='Conv1', activation='relu', padding='same')(input_)
            f = KL.Conv2D(48, (3, 3), name='Conv2', activation='relu', padding='same', strides=(2, 2))(f)
            f = KL.Conv2D(64, (3, 3), name='Conv3', activation='relu', padding='same')(f)
            f = KL.Conv2D(64, (3, 3), name='Conv4', activation='relu', padding='same', strides=(2, 2))(f)
            f = KL.Conv2D(64, (3, 3), name='Conv5', activation='relu', padding='same')(f)
            f = KL.Conv2D(64, (3, 3), name='Conv6', activation='relu', padding='same', strides=(2, 2))(f)
            f = KL.Conv2D(128, (3, 3), name='Conv7', activation='relu', padding='same')(f)
            f = KL.Conv2D(128, (3, 3), name='bottleneck', activation='relu', padding='same', strides=(2, 2))(f)
            model = tf.keras.Model(input_, f, name="diqa_custom")
        else:
            raise AttributeError("Invalid base_model name, should be a valid model from keras")

        return model

    def __call__(self):
        """
        #### Objective Error Model
        For the training phase, it is convenient to utilize the *tf.data* input pipelines to produce a 
        much cleaner and readable code. The only requirement is to create the function to apply to the input.
        """
        model = self._get_base_module()
        input_ = model.input
        model.trainable = False
        
        if self.custom:
            g = KL.Conv2D(1, (1, 1), name='final', padding='same', activation='linear')(model.output)
        else:
            nn = KL.Conv2D(512, (1, 1), use_bias=False, activation='relu')(model.output)
            nn = KL.Conv2D(256, (1, 1), use_bias=False, activation='relu')(nn)
            f = KL.Conv2D(128, (1, 1), use_bias=False, activation='relu', name='bottleneck')(nn)
            g = KL.Conv2D(1, (1, 1), name='final', use_bias=False, activation='relu')(f)

        self.model = tf.keras.Model(input_, g, name='objective_error_map')
        return self


class SubjectiveNetwork:
    def __init__(self) -> None:
        self.model = None

    def __call__(self, _input, bottleneck):
        """
        #### Subjective Model
        *Note: It would be a good idea to use the Spearman’s rank-order correlation coefficient (SRCC) or 
        Pearson’s linear correlation coefficient (PLCC) as accuracy metrics.*

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

        optimizer = tf.optimizers.Nadam(learning_rate=2 * 10 ** -4)
        v = KL.GlobalAveragePooling2D(data_format='channels_last')(bottleneck)
        h = KL.Dense(128, activation='relu')(v)
        h = KL.Dense(128, activation='relu')(v)
        h = KL.Dense(1)(h)
        self.model = tf.keras.Model(_input, h, name='subjective_error_map')

        self.model.compile(
            optimizer=optimizer,
            loss=tf.losses.MeanSquaredError(),
            metrics=[tf.metrics.MeanSquaredError()]
        )
        return self
