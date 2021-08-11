import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.applications as KA


class Diqa(object):
    def __init__(self, base_model_name, custom=False) -> None:
        self.custom = custom
        self.base_model_name = base_model_name
        # Initialize objective model for training
        self.subjective_score_model = None
        self.objective_score_model = None

    def _build(self):
        self.__build_obj_model()
        self.__build_sub_model()

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
        else:
            raise AttributeError("Invalid base_model name, should be a valid model from keras")

        return model

    def __build_obj_model(self):
        """
        #### Objective Error Model
        For the training phase, it is convenient to utilize the *tf.data* input pipelines to produce a 
        much cleaner and readable code. The only requirement is to create the function to apply to the input.
        """
        if self.custom:
            input_ = tf.keras.Input(shape=(None, None, 1), batch_size=1, name='original_image')
            f = KL.Conv2D(48, (3, 3), name='Conv1', activation='relu', padding='same')(input_)
            f = KL.Conv2D(48, (3, 3), name='Conv2', activation='relu', padding='same', strides=(2, 2))(f)
            f = KL.Conv2D(64, (3, 3), name='Conv3', activation='relu', padding='same')(f)
            f = KL.Conv2D(64, (3, 3), name='Conv4', activation='relu', padding='same', strides=(2, 2))(f)
            f = KL.Conv2D(64, (3, 3), name='Conv5', activation='relu', padding='same')(f)
            f = KL.Conv2D(64, (3, 3), name='Conv6', activation='relu', padding='same', strides=(2, 2))(f)
            f = KL.Conv2D(128, (3, 3), name='Conv7', activation='relu', padding='same')(f)
            f = KL.Conv2D(128, (3, 3), name='Conv8', activation='relu', padding='same', strides=(2, 2))(f)
            g = KL.Conv2D(1, (1, 1), name='Conv9', padding='same', activation='linear')(f)
        else:
            model = self._get_base_module()
            input_ = model.input
            model.trainable = False
            nn = KL.Conv2D(512, (1, 1), use_bias=False, activation='relu',
                           name='bottleneck-1')(model.output)
            nn = KL.Conv2D(256, (1, 1), use_bias=False, activation='relu', name='bottleneck-2')(nn)
            f = KL.Conv2D(128, (1, 1), use_bias=False, activation='relu', name='bottleneck-3')(nn)
            g = KL.Conv2D(1, (1, 1), use_bias=False, activation='relu', name='bottleneck-4')(f)

        self.objective_score_model = tf.keras.Model(input_, g, name='objective_error_map')

    def __build_sub_model(self):
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
        f = self.objective_score_model.get_layer('bottleneck-3').output
        v = KL.GlobalAveragePooling2D(data_format='channels_last')(f)
        h = KL.Dense(128, activation='relu')(v)
        h = KL.Dense(128, activation='relu')(v)
        h = KL.Dense(1)(h)
        self.subjective_score_model = tf.keras.Model(self.objective_score_model.input, h, name='subjective_error')

        self.subjective_score_model.compile(
            optimizer=optimizer,
            loss=tf.losses.MeanSquaredError(),
            metrics=[tf.metrics.MeanSquaredError()]
        )
