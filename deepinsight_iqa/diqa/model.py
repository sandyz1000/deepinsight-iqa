import tensorflow as tf


class Diqa(object):
    def __init__(self, custom=False) -> None:
        self.custom = custom
        # Initialize objective model for training
        self._objective_score_build()
        self._subjective_score_build()

    @property
    def objective_score_model(self):
        return self.__objective_error_map

    @property
    def subjective_score_model(self):
        return self.__subjective_error_map

    @objective_score_model.setter
    def objective_score_model(self, model: tf.keras.Model):
        self.__objective_error_map = model.fit_generator

    @subjective_score_model.setter
    def subjective_score_model(self, model: tf.keras.Model):
        self.__subjective_error_map = model

    def _objective_score_build(self):
        """
        #### Objective Error Model
        For the training phase, it is convenient to utilize the *tf.data* input pipelines to produce a 
        much cleaner and readable code. The only requirement is to create the function to apply to the input.
        """
        if self.custom:
            input_ = tf.keras.Input(shape=(None, None, 1), batch_size=1, name='original_image')
            f = tf.keras.layers.Conv2D(48, (3, 3), name='Conv1', activation='relu', padding='same')(input_)
            f = tf.keras.layers.Conv2D(48, (3, 3), name='Conv2', activation='relu', padding='same', strides=(2, 2))(f)
            f = tf.keras.layers.Conv2D(64, (3, 3), name='Conv3', activation='relu', padding='same')(f)
            f = tf.keras.layers.Conv2D(64, (3, 3), name='Conv4', activation='relu', padding='same', strides=(2, 2))(f)
            f = tf.keras.layers.Conv2D(64, (3, 3), name='Conv5', activation='relu', padding='same')(f)
            f = tf.keras.layers.Conv2D(64, (3, 3), name='Conv6', activation='relu', padding='same', strides=(2, 2))(f)
            f = tf.keras.layers.Conv2D(128, (3, 3), name='Conv7', activation='relu', padding='same')(f)
            f = tf.keras.layers.Conv2D(128, (3, 3), name='Conv8', activation='relu', padding='same', strides=(2, 2))(f)
            g = tf.keras.layers.Conv2D(1, (1, 1), name='Conv9', padding='same', activation='linear')(f)
        else:
            model = tf.keras.applications.MobileNetV2(input_shape=(None, None, 3),
                                                      include_top=False,
                                                      weights="imagenet")

            input_ = model.input
            model.trainable = False
            nn = tf.keras.layers.Conv2D(512, (1, 1), use_bias=False, activation='relu',
                                        name='bottleneck-1')(model.output)
            nn = tf.keras.layers.Conv2D(256, (1, 1), use_bias=False, activation='relu', name='bottleneck-2')(nn)
            f = tf.keras.layers.Conv2D(128, (1, 1), use_bias=False, activation='relu', name='bottleneck-3')(nn)
            g = tf.keras.layers.Conv2D(1, (1, 1), use_bias=False, activation='relu', name='bottleneck-4')(f)

        self.__objective_error_map = tf.keras.Model(input_, g, name='objective_error_map')

    def _subjective_score_build(self):
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
        f = self.__objective_error_map.get_layer('bottleneck-3').output
        v = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')(f)
        h = tf.keras.layers.Dense(128, activation='relu')(v)
        h = tf.keras.layers.Dense(128, activation='relu')(v)
        h = tf.keras.layers.Dense(1)(h)
        self.__subjective_error_map = tf.keras.Model(self.__objective_error_map.input, h, name='subjective_error')

        self.__subjective_error_map.compile(
            optimizer=optimizer,
            loss=tf.losses.MeanSquaredError(),
            metrics=[tf.metrics.MeanSquaredError()]
        )
