
# %%
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, GlobalAveragePooling2D
# from keras.layers import Conv2D, Dense, GlobalAveragePooling2D
from deepinsight_iqa.diqa.utils.tf_imgutils import gaussian_filter, image_normalization, rescale, image_shape
import tensorflow.keras.backend as K
import imquality.datasets


# %%
print(f'tensorflow version {tf.__version__}')


# %%
builder = imquality.datasets.LiveIQA()
builder.download_and_prepare()


# %% [markdown]
# After downloading and preparing the data, turn the builder into a dataset, and shuffle it. Note that the batch is equal to 1. The reason is that each image has a different shape. Increasing the batch TensorFlow will raise an error.

# %%
# Test MNIST notebook for checking tensorflow_dataset
import tensorflow_datasets as tfds

# The following is the equivalent of the `load` call above.

# You can fetch the DatasetBuilder class by string
mnist_builder = tfds.builder('mnist')

# Download the dataset
mnist_builder.download_and_prepare()

# Construct a tf.data.Dataset
ds = mnist_builder.as_dataset(split='train')

# Get the `DatasetInfo` object, which contains useful information about the
# dataset and its features
info = mnist_builder.info
print(info)


# %%
ds = builder.as_dataset(shuffle_files=True)['train']
ds = ds.shuffle(1024).batch(1)

# %% [markdown]
# The output is a generator; therefore, to access it using the bracket operator [ ] causes an error. 
# There are two ways to access the images in the generator. The first way is to turn the generator into an 
# iterator and extract a single sample at a time using the *next* function.

# %%
next(iter(ds)).keys()

# %% [markdown]
# As you can see, the output is a dictionary that contains the tensor representation for the distorted image, 
# the reference image, and the subjective score (dmos).
#
# Another way is to extract samples from the generator by taking samples with a for loop:

# %%
for features in ds.take(2):
    distorted_image = features['distorted_image']
    reference_image = features['reference_image']
    dmos = tf.round(features['dmos'][0], 2)
    distortion = features['distortion'][0]
    print(f'The distortion of the image is {dmos} with'
          f' a distortion {distortion} and shape {distorted_image.shape}')
    # show_images([reference_image, distorted_image])

# %% [markdown]
# # Methodology
#
# ## Image Normalization
#
# The first step for DIQA is to pre-process the images. The image is converted into grayscale, and then a low-pass filter is applied. The low-pass filter is defined as:
#
# \begin{align*}
# \hat{I} = I_{gray} - I^{low}
# \end{align*}
#
# where the low-frequency image is the result of the following algorithm:
#
# 1. Blur the grayscale image.
# 2. Downscale it by a factor of 1 / 4.
# 3. Upscale it back to the original size.
#
# The main reasons for this normalization are (1) the Human Visual System (HVS) is not sensitive to changes in the low-frequency band, and (2) image distortions barely affect the low-frequency component of images.

# %%


def image_preprocess(image: tf.Tensor) -> tf.Tensor:
    image = tf.cast(image, tf.float32)
    image = tf.image.rgb_to_grayscale(image)
    image_low = gaussian_filter(image, 16, 7 / 6)
    image_low = rescale(image_low, 1 / 4, method=tf.image.ResizeMethod.BICUBIC)
    image_low = tf.image.resize(image_low, size=image_shape(image), method=tf.image.ResizeMethod.BICUBIC)
    return image - tf.cast(image_low, image.dtype)


# %%
for features in ds.take(2):
    distorted_image = features['distorted_image']
    reference_image = features['reference_image']
    I_d = image_preprocess(distorted_image)
    I_d = tf.image.grayscale_to_rgb(I_d)
    I_d = image_normalization(I_d, 0, 1)
    # Show reference and distorted image
    # show_images([distorted_image, reference_image])

    # show_images([reference_image, I_d])

# %% [markdown]
# **Fig 1.** On the left, the original image. On the right, the image after applying the low-pass filter.
# %% [markdown]
# ## Objective Error Map
#
# For the first model, objective errors are used as a proxy to take advantage of the effect of increasing data. The loss function is defined by the mean squared error between the predicted and ground-truth error maps.
#
# \begin{align*}
# \mathbf{e}_{gt} = err(\hat{I}_r, \hat{I}_d)
# \end{align*}
#
# and *err(·)* is an error function. For this implementation, the authors recommend using
#
# \begin{align*}
# \mathbf{e}_{gt} = | \hat{I}_r -  \hat{I}_d | ^ p
# \end{align*}
#
# with *p=0.2*. The latter is to prevent that the values in the error map are small or close to zero.

# %%


def error_map(reference: tf.Tensor, distorted: tf.Tensor, p: float = 0.2) -> tf.Tensor:
    assert reference.dtype == tf.float32 and distorted.dtype == tf.float32, 'dtype must be tf.float32'
    return tf.pow(tf.abs(reference - distorted), p)


# %%
for features in ds.take(3):
    reference_image = features['reference_image']
    I_r = image_preprocess(reference_image)
    I_d = image_preprocess(features['distorted_image'])
    e_gt = error_map(I_r, I_d, 0.2)
    I_d = image_normalization(tf.image.grayscale_to_rgb(I_d), 0, 1)
    e_gt = image_normalization(tf.image.grayscale_to_rgb(e_gt), 0, 1)
    show_images([reference_image, I_d, e_gt])

# %% [markdown]
# **Fig 2.** On the left, the original image. In the middle, the pre-processed image, and finally, the image representation of the error map.
# %% [markdown]
# ## Reliability Map
#
# According to the authors, the model is likely to fail to predict images with homogeneous regions. To prevent it, they propose a reliability function. The assumption is that blurry areas have lower reliability than textured ones. The reliability function is defined as
#
# \begin{align*}
# \mathbf{r} = \frac{2}{1 + exp(-\alpha|\hat{I}_d|)} - 1
# \end{align*}
#
# where α controls the saturation property of the reliability map. The positive part of a sigmoid is used to assign sufficiently large values to pixels with low intensity.

# %%


def reliability_map(distorted: tf.Tensor, alpha: float) -> tf.Tensor:
    assert distorted.dtype == tf.float32, 'The Tensor must by of dtype tf.float32'
    return 2 / (1 + tf.exp(- alpha * tf.abs(distorted))) - 1

# %% [markdown]
# The previous definition might directly affect the predicted score. Therefore, the average reliability map is used instead.
#
# \begin{align*}
# \mathbf{\hat{r}} = \frac{1}{\frac{1}{H_rW_r}\sum_{(i,j)}\mathbf{r}(i,j)}\mathbf{r}
# \end{align*}
#
# For the Tensorflow function, we just calculate the reliability map and divide it by its mean.

# %%


def average_reliability_map(distorted: tf.Tensor, alpha: float) -> tf.Tensor:
    r = reliability_map(distorted, alpha)
    return r / tf.reduce_mean(r)


# %%
for features in ds.take(2):
    reference_image = features['reference_image']
    I_d = image_preprocess(features['distorted_image'])
    r = average_reliability_map(I_d, 1)
    r = image_normalization(tf.image.grayscale_to_rgb(r), 0, 1)
    # show_images([reference_image, r], cmap='gray')

# %% [markdown]
# **Fig 3.** On the left, the original image, and on the right, its average reliability map.
# %% [markdown]
# ## Loss function
#
# The loss function is defined as the mean square error of the product between the reliability map and the objective error map. The error is the difference between the predicted error map and the ground-truth error map.
#
# \begin{align*}
# \mathcal{L}_1(\hat{I}_d; \theta_f, \theta_g) = ||g(f(\hat{I}_d, \theta_f), \theta_g) - \mathbf{e}_{gt}) \odot \mathbf{\hat{r}}||^2_2
# \end{align*}
#
# The loss function requires to multiply the error by the reliability map; therefore, we cannot use the default loss implementation *tf.loss.MeanSquareError*.
#

# %%


def loss(model, x, y_true, r):
    y_pred = model(x)
    return tf.reduce_mean(tf.square((y_true - y_pred) * r))

# %% [markdown]
# After creating the custom loss, we need to tell TensorFlow how to differentiate it. The good thing is that we can take advantage of [automatic differentiation](https://www.tensorflow.org/tutorials/customization/autodiff) using *tf.GradientTape*.

# %%


def gradient(model, x, y_true, r):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, y_true, r)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

# %% [markdown]
# ## Optimizer
# The authors suggested using a Nadam optimizer with a learning rate of *2e-4*.


# %%
optimizer = tf.optimizers.Nadam(learning_rate=2 * 10 ** -4)
# optimizer = keras.optimizers.Nadam(learning_rate=2 * 10 ** -4)

# %% [markdown]
# # Training
#
# ## Objective Error Model
# For the training phase, it is convenient to utilize the *tf.data* input pipelines to produce a much cleaner and readable code. The only requirement is to create the function to apply to the input.

# %%


def calculate_error_map(features):
    I_d = image_preprocess(features['distorted_image'])
    I_r = image_preprocess(features['reference_image'])
    r = rescale(average_reliability_map(I_d, 0.2), 1 / 4)
    e_gt = rescale(error_map(I_r, I_d, 0.2), 1 / 4)
    return (I_d, e_gt, r)

# %% [markdown]
# Then, map the *tf.data.Dataset* to the *calculate_error_map* function.


# %%
train = ds.map(calculate_error_map)

# %% [markdown]
# Applying the transformation is executed in almost no time. The reason is that the processor is not performing any operation to the data yet, it happens on demand. This concept is commonly called [lazy-evaluation](https://wiki.python.org/moin/Generators).
#
# So far, the following components are implemented:
# 1. The generator that pre-processes the input and calculates the target.
# 2. The loss and gradient functions required for the custom training loop.
# 3. The optimizer function.
#
# The only missing bits are the models' definition.
# %% [markdown]
#
# ![alt text](https://d3i71xaburhd42.cloudfront.net/4b1f961ae1fac044c23c51274d92d0b26722f877/4-Figure2-1.png "CNN architecture")
#
#
# %% [markdown]
# **Fig 4.** Architecture of the objective error model and subjective score model.
# %% [markdown]
# In the previous image, it is depicted how:
# - The pre-processed image gets into the convolutional neural network (CNN).
# - It is transformed by 8 convolutions with the Relu activation function and "same" padding. This is defined as f(·).
# - The output of f(·) is processed by the last convolution with a linear activation function. This is defined as g(·).

# %%
input = tf.keras.Input(shape=(None, None, 1), batch_size=1, name='original_image')
# input = keras.Input(shape=(None, None, 1), batch_size=1, name='original_image')
f = Conv2D(48, (3, 3), name='Conv1', activation='relu', padding='same')(input)
f = Conv2D(48, (3, 3), name='Conv2', activation='relu', padding='same', strides=(2, 2))(f)
f = Conv2D(64, (3, 3), name='Conv3', activation='relu', padding='same')(f)
f = Conv2D(64, (3, 3), name='Conv4', activation='relu', padding='same', strides=(2, 2))(f)
f = Conv2D(64, (3, 3), name='Conv5', activation='relu', padding='same')(f)
f = Conv2D(64, (3, 3), name='Conv6', activation='relu', padding='same')(f)
f = Conv2D(128, (3, 3), name='Conv7', activation='relu', padding='same')(f)
f = Conv2D(128, (3, 3), name='Conv8', activation='relu', padding='same')(f)
g = Conv2D(1, (1, 1), name='Conv9', padding='same', activation='linear')(f)

objective_error_map = tf.keras.Model(input, g, name='objective_error_map')
# objective_error_map = keras.Model(input, g, name='objective_error_map')
objective_error_map.summary()

# %%
# Load pre-trained model for objectives error map
objective_error_map.load_weights("models/objective-model.v1.h5")

# %% [markdown]
# For the custom training loop, it is necessary to:
#
# 1. Define a metric to measure the performance of the model.
# 2. Calculate the loss and the gradients.
# 3. Use the optimizer to update the weights.
# 4. Print the accuracy.

# %%
for epoch in range(1):
    epoch_accuracy = tf.keras.metrics.MeanSquaredError()

    step = 0
    for I_d, e_gt, r in train:
        loss_value, gradients = gradient(objective_error_map, I_d, e_gt, r)
        optimizer.apply_gradients(zip(gradients, objective_error_map.trainable_weights))

        epoch_accuracy(e_gt, objective_error_map(I_d))

        if step % 100 == 0:
            print('step %s: mean loss = %s' % (step, epoch_accuracy.result()))

        step += 1

# %% [markdown]
# *Note: It would be a good idea to use the Spearman’s rank-order correlation coefficient (SRCC) or Pearson’s linear correlation coefficient (PLCC) as accuracy metrics.*
#
# # Subjective Score Model
#
# To create the subjective score model, let's use the output of f(·) to train a regressor.

# %%
v = GlobalAveragePooling2D(data_format='channels_last')(f)
h = Dense(128, activation='relu')(v)
h = Dense(1)(h)
subjective_error = tf.keras.Model(input, h, name='subjective_error')

subjective_error.compile(
    optimizer=optimizer,
    loss=tf.losses.MeanSquaredError(),
    metrics=[tf.metrics.MeanSquaredError()])

subjective_error.summary()

# Load pre-train weights
subjective_error.load_weights('models/subjective-model.v1.h5')
# subjective_error.load_weights('models/subjective-model.v4.h5')
# %% [markdown]
# Training a model with the fit method of *tf.keras.Model* expects a dataset that returns two arguments. The first one is the input, and the second one is the target.

# %%


def calculate_subjective_score(features):
    I_d = image_preprocess(features['distorted_image'])
    mos = features['dmos']
    return (I_d, mos)


train = ds.map(calculate_subjective_score)

# %% [markdown]
# Then, *fit* the subjective score model.

# %%
history = subjective_error.fit(train, epochs=1)

# %% [markdown]
# # Prediction
#
# Predicting with the already trained model is simple. Just use the *predict* method in the model.

# %%
sample = next(iter(ds))
I_d = image_preprocess(sample['distorted_image'])
I_r = image_preprocess(sample['reference_image'])
target = sample['dmos'][0]

# %%
prediction = subjective_error.predict(I_d)[0][0]
print(f'the predicted value is: {prediction:.4f} and target is: {target:.4f}')
prediction = subjective_error.predict(I_r)[0][0]
print(f'the predicted value with reference images is: {prediction:.4f}')


# %%
# Display images
distorted_image = sample['distorted_image']
reference_image = sample['reference_image']
# I_d = image_preprocess(distorted_image)
# I_d = tf.image.grayscale_to_rgb(I_d)
# I_d = image_normalization(I_d, 0, 1)
print("Show reference and distorted image")
show_images([distorted_image, reference_image])
print("Show reference and processed image")
# show_images([reference_image, I_d])


# %%

import cv2
x = cv2.imread("/Users/sandipdey/Documents/Bad_quality/4019-26043006399-PAN.jpg")
dataset = tf.data.Dataset.from_tensor_slices(x)
