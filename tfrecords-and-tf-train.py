# %%
import tensorflow as tf
print(tf.__version__)

# %% [markdown]
# # Protocol Buffers
#
# Protocol buffers provide a mechanism to read and store structured data in an efficient way.
# The description of the data structure is written as a protocol buffer message into a `.proto`
# file which tells the protocol buffer compiler how the data will be stored. Then the compiler
# creates a class that implements efficient encoding and parsing of the data. This class comes
# with methods to get and set the data. Think of it as an optimized version of XML.
#
# Here is an example of a protocol buffer message from the [guide](https://developers.google.com/protocol-buffers/docs/overview):
# ```
# message Person {
#   required string name = 1;
#   required int32 id = 2;
#   optional string email = 3;
#
#   enum PhoneType {
#     MOBILE = 0;
#     HOME = 1;
#     WORK = 2;
#   }
#
#   message PhoneNumber {
#     required string number = 1;
#     optional PhoneType type = 2 [default = HOME];
#   }
#
#   repeated PhoneNumber phone = 4;
# }
# ```
#
# Each message type has one or more uniquely numbered fields, and each field has a name and a value type, where value types can be numbers (integer or floating-point), booleans, strings, raw bytes, or even (as in the example above) other protocol buffer message types, allowing you to structure your data hierarchically.
#
# For our purposes, you don't even need to write a protocol buffer message yourself, so no need to know the details of protocol buffers in general. It is enough to get familiar with the following protocol buffer messages which are part of Tensorflow:
#
#  1 - `tf.train.BytesList`,   `tf.train.FloatList`, `tf.train.Int64List`: These protocol buffer messages has a single field named 'value' that can hold data of the following types:
#
# For `tf.train.BytesList`
#   - `string`
#   - `byte`
#
# For `tf.train.FloatList`
#
#   - `float` (`float32`)
#   - `double` (`float64`)
#
# For `tf.train.Int64List`
#
#   - `bool`
#   - `enum`
#   - `int32`
#   - `uint32`
#   - `int64`
#   - `uint64`
#
# Think of these three messages as the data types.
#
#  2 - `tf.train.Feature`: This message has a single field that can accept one of the following above types. Think of this as a single feature of a data point.
#
#  3 - `tf.train.Features`: This protocol buffer message is a `{"string": tf.train.Feature}` mapping. Think of this as the separate features of a given data point.
#
#  4 - `tf.train.Example`: This is a protocol buffer message with a single field of type `tf.train.Features` called `features`. Think of this as an abstraction of a single data point. It is simply a wrapper around the `tf.train.Features` message.
#
#  Take a look at the [source](https://medium.com/r/?url=https%3A%2F%2Fgithub.com%2Ftensorflow%2Ftensorflow%2Fblob%2Fr2.0%2Ftensorflow%2Fcore%2Fexample%2Ffeature.proto) to get a better idea on what these objects are.
#
# In order to convert a data point, a single row in your data, into a `tf.train.Example` you need to create a  `tf.train.Feature` for each feature in the data. Tensorflow documentation provides the following shortcut functions for this purpose:
#
#
#
#
#

# %%


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# %% [markdown]
# Each function takes a scalar input value and returns a `tf.train.Feature` containing one of the three list types above. Let's look at a few examples of how these work.


# %%
# strings needs to be converted into bytes.
print(_bytes_feature(b'some string'))

print(_float_feature(0.5))

print(_int64_feature(True))
print(_int64_feature(1))

# %% [markdown]
# Non-scalar features need to be converted into binary-strings using `tf.io.serialize_tensor` function. To convert the binary-string back to tensor use `tf.io.parse_tensor` function.

# %%
import numpy as np

a = np.random.randn(2, 2)
_bytes_feature(tf.io.serialize_tensor(a))

# %% [markdown]
# ## Steps to create `tf.train.Example` message from data:
#
# For each data point do the following,
#
#  1 - For each feature, depending on the type run one of the three shortcut functions above to create a `tf.train.Feature` message.
#
#  2 - Create a dictionary where the keys are the feature names and the values are the `tf.train.Feature` messages created in the first step. The dictionary may look like this:
#
#  ```
#  feature = {
#       'feature0': _int64_feature(feature0),
#       'feature1': _int64_feature(feature1),
#       'feature2': _bytes_feature(feature2),
#       'feature3': _float_feature(feature3),
#   }
#
#  ```
#
#
#  3 - Create a `tf.train.Example` message from the dictionary you created in step 2. Recall that a `tf.train.Example` message has a single filed of type `tf.train.Features`. So first create a `tf.train.Features` message from the dictionary and then create an `tf.train.Example` message using this. Here is what this looks like:
#
#  ```
#  example_proto = tf.train.Example(features = tf.train.Features(feature=feature))
#  ```
# %% [markdown]
# In order to give an example let's create an artificial dataset:

# %%
import numpy as np

# The number of observations in the dataset.
n_observations = 1000

# Boolean feature, encoded as False or True.
feature0 = np.random.choice([False, True], n_observations)

# Float feature
feature1 = np.random.randn(n_observations)

# String feature
strings = np.array([b'cat', b'dog'])
feature2 = np.random.choice(strings, n_observations)

# Non-scalar Float feature, 2x2 matrices sampled from a standard normal distribution
feature3 = np.random.randn(n_observations, 2, 2)

dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))

# %% [markdown]
# Let's write a function that creates a tf.train.Example following the above steps.

# %%


def create_example(feature0, feature1, feature2, feature3):
    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _float_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _bytes_feature(feature3),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto


# %%
for feature0, feature1, feature2, feature3 in dataset.take(1):
    example_proto = create_example(feature0,
                                   feature1,
                                   feature2,
                                   tf.io.serialize_tensor(feature3))
    print(example_proto)

# %% [markdown]
# Notice that we needed to convert feature3 into a byte-string using tf.io.serialize_tensor since feature3 is non-scalar.
#
# Now we know how to create tf.train.Example messages from the data. Next we will go over how to write these into a TFRefords file and how to read them.
# %% [markdown]
# # TFRecords
#
# To read data efficiently it can be helpful to serialize your data and store it in a set of files (100-200MB each) that can each be read linearly. This is especially true if the data is being streamed over a network. This can also be useful for caching any data-preprocessing. The TFRecord format is a simple format for storing a sequence of binary records.
#
# A TFRecord file consists of a sequence of records where each record is a byte-string. It is not required to use `tf.train.Example` messages in TFRecords files but I will use them. We already saw above how to create `tf.train.Example` messages from the data. In order to write these messages as TFRecords, we need to convert them into byte-strings. For this purpose, we can use `SerializeToString()` method. So we can update our function to convert the Example message into a byte-string as follows:
#
#

# %%


def serialize_example(feature0, feature1, feature2, feature3):
    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _float_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _bytes_feature(feature3),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# %%
for feature0, feature1, feature2, feature3 in dataset.take(1):
    serialized_example = serialize_example(feature0,
                                           feature1,
                                           feature2,
                                           tf.io.serialize_tensor(feature3))
    print(serialized_example)

# %% [markdown]
# ## Writing into TFRecords
#
# In order to write the data into a TFRecords file we need to convert each data point into a byte-string follwoing the above process and write it into file using a `tf.io.TFRecordsWriter`.

# %%
file_path = 'data.tfrecords'
with tf.io.TFRecordWriter(file_path) as writer:
    for feature0, feature1, feature2, feature3 in dataset:

        serialized_example = serialize_example(feature0,
                                               feature1,
                                               feature2,
                                               tf.io.serialize_tensor(feature3))
        writer.write(serialized_example)

# %% [markdown]
# This will create `data.tfrecords` file in the specified path, which is the working directory in the above case.
#
# A great place to apply some preprocessing to your data such as data augmentation is before you serialize the example and write into file. We will see an example of this below with image data.
#
# %% [markdown]
# ## Reading TFRecords
#
# Let's see how to read the records we created. For this first we create a `tf.data.TFRecordDataset` from the list of TFRecord file names.

# %%
file_paths = [file_path]
tfrecord_dataset = tf.data.TFRecordDataset(file_paths)

# %% [markdown]
# Now each data point in this dataset are simply the raw byte-strings as returned by `serialize_example` function.

# %%
for record in tfrecord_dataset.take(5):
    print(record)

# %% [markdown]
# The following function reads a serialized_example and parse it using the feature description.

# %%


def read_tfrecord(serialized_example):
    feature_description = {
        'feature0': tf.io.FixedLenFeature((), tf.int64),
        'feature1': tf.io.FixedLenFeature((), tf.float32),
        'feature2': tf.io.FixedLenFeature((), tf.string),
        'feature3': tf.io.FixedLenFeature((), tf.string),
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    feature0 = example['feature0']
    feature1 = example['feature1']
    feature2 = example['feature2']
    feature3 = tf.io.parse_tensor(example['feature3'], out_type=tf.float64)

    return feature0, feature1, feature2, feature3


# %%
parsed_dataset = tfrecord_dataset.map(read_tfrecord)

for data in parsed_dataset.take(2):
    print(data)

# %% [markdown]
# # Example with image data
#
# Now let's go over an example involving image data. The data set I will be using is [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) and I have only took the first 100 images from each category for simplicity and put them under a folder named data in the working directory.
# %% [markdown]
# First we will create a list of image paths.

# %%
import glob

data_dir = 'data/'
image_paths = glob.glob(data_dir + '*.jpg')

# %% [markdown]
# The first 100 images are cat images and the second 100 images are dog images. We will label cats as 0 and dogs as 1.

# %%
labels = np.append(np.zeros(100, dtype=int), np.ones(100, dtype=int))

# %% [markdown]
# Let's look at the first 9 images.

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for i, path in enumerate(image_paths[:9]):
    img = tf.keras.preprocessing.image.load_img(path)
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
plt.show()

# %% [markdown]
# Now we are ready to create TFRecords. Other than the feature descriptions the serialize_example function is the same as above.

# %%


def serialize_example(image, label, image_shape):
    feature = {
        'image': _bytes_feature(image),
        'label': _int64_feature(label),
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
    }

    #  Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# %%
tfrecord_dir = 'tfrecords/data.tfrecords'
with tf.io.TFRecordWriter(tfrecord_dir) as writer:
    for image_path, label in zip(image_paths, labels):

        img = tf.keras.preprocessing.image.load_img(image_path)
        img_array = tf.keras.preprocessing.image.img_to_array(img)

        img_array = tf.keras.preprocessing.image.random_zoom(img_array, (0.5, 0.5),
                                                             row_axis=0,
                                                             col_axis=1,
                                                             channel_axis=2)

        img_bytes = tf.io.serialize_tensor(img_array)
        image_shape = img_array.shape

        example = serialize_example(img_bytes, label, image_shape)
        writer.write(example)

# %% [markdown]
# In the above loop, we first convert the image into a numpy array and then apply some data augmentation, in this case random_zoom. Then we convert the resulting array into byte-strings and write it to file together with the labels and image dimensions.
# %% [markdown]
# In order to read from the tfrecords, we proceed as above.

# %%


def read_tfrecord(serialized_example):
    feature_description = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.int64),
        'height': tf.io.FixedLenFeature((), tf.int64),
        'width': tf.io.FixedLenFeature((), tf.int64),
        'depth': tf.io.FixedLenFeature((), tf.int64)
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    image = tf.io.parse_tensor(example['image'], out_type=float)
    image_shape = [example['height'], example['width'], example['depth']]
    image = tf.reshape(image, image_shape)

    return image, example['label']

# %% [markdown]
# Let's apply this function to the tfrecord dataset and then look at the first 9 images again.


# %%
tfrecord_dataset = tf.data.TFRecordDataset(tfrecord_dir)
parsed_dataset = tfrecord_dataset.map(read_tfrecord)

plt.figure(figsize=(10, 10))

for i, data in enumerate(parsed_dataset.take(9)):
    img = tf.keras.preprocessing.image.array_to_img(data[0])
    # img.save('{}.jpg'.format(i))
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
plt.show()


# %%
