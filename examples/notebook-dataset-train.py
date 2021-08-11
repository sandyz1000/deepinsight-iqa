# %% [markdown]
# ## Datagenerator, TFRecords for both nima and diqa
# IQA Tasks:
# ----------
# -> Datagenerator for IQA -- Done
# -> Classifier n/w for good and bad -- Done
# -> Modify network with new data -- Doing
# -> Metrics for IQA -- Doing
# TODO: Verify Datagenerator and training script


# %%
import os
import sys
import json
import tensorflow as tf
import cv2
from scipy.ndimage import variance, mean, maximum
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
sys.path.append(os.path.realpath(os.pardir))
from deepinsight_iqa.data_pipeline.dataset import (
    Tid2013RecordDataset, CSIQRecordDataset, AVARecordDataset,
    LiveRecordDataset
)
DATA_DIR = "/Users/sdey/Documents/dataset/image-quality-assesement"
tfrecord_outdir = os.path.expanduser("~/tensorflow_dataset")
# %% [markdown]
# ### TID 2013 Dataset
# %%
tfrecord = Tid2013RecordDataset()
# %%
tfrecord_path = tfrecord.write_tfrecord_dataset(os.path.join(DATA_DIR, "tid2013"), "mos.csv")

# %%
tid2013_ds = tfrecord.load_tfrecord_dataset(os.path.join(tfrecord_outdir, "tid2013/data.tf.records"))
# %%
x = next(iter(tid2013_ds.take(1)))

# %% [markdown]
# ### CSIQ Dataset
# %%
tfrecord = CSIQRecordDataset()
# %%
tfrecord_path = tfrecord.write_tfrecord_dataset(os.path.join(DATA_DIR, "CSIQ"), "csiq.csv")
# %%
csiq_ds = tfrecord.load_tfrecord_dataset(os.path.join(tfrecord_outdir, "csiq/data.tf.records"))
dist_img, ref_img, distortion, dmos, dmos_std = next(iter(csiq_ds.take(1)))

# %% [markdown]
# ### Live Dataset
# %%
tfrecord = LiveRecordDataset()
# %%
tfrecord_path = tfrecord.write_tfrecord_dataset(os.path.join(DATA_DIR, "live"), "dmos.csv")

# %%
live_ds = tfrecord.load_tfrecord_dataset(os.path.join(tfrecord_outdir, "live/data.tf.records"))
distorted_image, reference_image, distortion, dmos, dmos_realigned, dmos_realigned_std = next(iter(live_ds.take(1)))

# %% [markdown]
# ### AVA Dataset
from deepinsight_iqa.data_pipeline.dataset import AVARecordDataset
tfrecord = AVARecordDataset()
# %%
tfrecord_path = tfrecord.write_tfrecord_dataset(
    os.path.join(DATA_DIR, "ava"), "AVA.txt")
# %%
ava_ds = tfrecord.load_tfrecord_dataset(os.path.join(tfrecord_outdir, "ava/data.tf.records"))
image, mos, score_dist, linked_tags, challenge = next(iter(ava_ds.take(1)))

# %%
from deepinsight_iqa.diqa.train import TrainWithTFDS, Train
from deepinsight_iqa.diqa.utils.img_utils import image_preprocess
# %%

def parse_config(job_dir, config_file):
    os.makedirs(os.path.join(job_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(job_dir, 'logs'), exist_ok=True)
    config = json.load(open(config_file, 'r'))
    return config


# %%
job_dir = os.path.realpath(os.path.pardir)
config_file = os.path.realpath(os.path.join(job_dir, "confs/diqa_conf.json"))
cfg = parse_config(job_dir, config_file)
dataset_type = cfg.pop('dataset_type')
# %%
# ## TRAIN TID2013
image_dir, input_file = "/Users/sdey/Documents/dataset/image-quality-assesement/tid2013", "mos.csv"
dataset_type = "tid2013"
trainer = Train(image_dir, input_file, dataset_type, do_augment=True, **cfg)
# trainer.final_model.summary()
# %%
features, labels = next(iter(trainer.train_generator))
# %%
trainer.train()
# %%
dataset_type = "live"
image_dir, input_file = "/Users/sdey/Documents/dataset/image-quality-assesement/live", "dmos.csv"
trainer = Train(image_dir, input_file, dataset_type, do_augment=True, **cfg)
# trainer.final_model.summary()
# %%
features, labels = next(iter(trainer.train_generator))
# %%
trainer.train()
# %%
image_dir, input_file = "/Users/sdey/Documents/dataset/image-quality-assesement/CSIQ", "csiq.csv"
dataset_type = "csiq"
trainer = Train(image_dir, input_file, dataset_type, do_augment=False, **cfg)
# %%
features, labels = next(iter(trainer.train_generator))
# %%
trainer.train()
# %%
from deepinsight_iqa.data_pipeline.diqa_gen.diqa_datagen import AVADataRowParser
image_dir, input_file = "/Users/sdey/Documents/dataset/image-quality-assesement/ava", "AVA.txt"
gen = AVADataRowParser(input_file, image_dir, shuffle=True)
# %%
features, mos_scores, distributions = next(iter(gen))


# %% [markdown]
# ## Objective Model and Subjective Model features
import os
import cv2
import tensorflow as tf
import dill as pickle
# import imutils
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


OBJECTIVE_MODEL_NAME = "weights/diqa/objective-model-mobilenetv2.tid2013.h5"
SUBJECTIVE_MODEL_NAME = "weights/diqa/subjective-model-mobilenetv2.tid2013.h5"


model = tf.keras.applications.MobileNetV2(input_shape=(None, None, 3), include_top=False, weights="imagenet")
input_ = model.input
model.trainable = False
nn = tf.keras.layers.Conv2D(512, (1, 1), use_bias=False, activation='relu', name='bottleneck-1')(model.output)
nn = tf.keras.layers.Conv2D(256, (1, 1), use_bias=False, activation='relu', name='bottleneck-2')(nn)
f = tf.keras.layers.Conv2D(128, (1, 1), use_bias=False, activation='relu', name='bottleneck-3')(nn)
g = tf.keras.layers.Conv2D(1, (1, 1), use_bias=False, activation='relu', name='bottleneck-4')(f)
objective_error_map = tf.keras.Model(input_, g, name='objective_error_map')

# %%
optimizer = tf.optimizers.Nadam(learning_rate=2 * 10 ** -4)
f = objective_error_map.get_layer('bottleneck-3').output
v = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')(f)
h = tf.keras.layers.Dense(128, activation='relu')(v)
h = tf.keras.layers.Dense(128, activation='relu')(v)
output = tf.keras.layers.Dense(1)(h)
subjective_error_map = tf.keras.Model(objective_error_map.input, output, name='subjective_error')

subjective_error_map.compile(optimizer=optimizer, loss=tf.losses.MeanSquaredError(),
                             metrics=[tf.metrics.MeanSquaredError()])

subjective_error_map.load_weights(SUBJECTIVE_MODEL_NAME)

# %%


def rgbtogray(image: tf.Tensor):
    image = tf.squeeze(tf.image.rgb_to_grayscale(image))
    image = tf.image.convert_image_dtype(image, tf.dtypes.float32)
    return image


# %%
plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)

img = cv2.imread(img_path, 0)
plt.subplot(151), plt.imshow(img, "gray"), plt.title("Original Image")

original = np.fft.fft2(img)
plt.subplot(152), plt.imshow(np.log(1 + np.abs(original)), "gray"), plt.title("Spectrum")

center = np.fft.fftshift(original)
plt.subplot(153), plt.imshow(np.log(1 + np.abs(center)), "gray"), plt.title("Centered Spectrum")

inv_center = np.fft.ifftshift(center)
plt.subplot(154), plt.imshow(np.log(1 + np.abs(inv_center)), "gray"), plt.title("Decentralized")

processed_img = np.fft.ifft2(inv_center)
plt.subplot(155), plt.imshow(np.abs(processed_img), "gray"), plt.title("Processed Image")

plt.show()

# %%
plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)
img = tf.image.decode_image(tf.io.read_file(img_path))
gray = rgbtogray(img)
(h, w) = gray.shape
(cX, cY) = (int(w / 2.0), int(h / 2.0))
size = 60
plt.subplot(151), plt.imshow(gray, "gray"), plt.title("Original Image")

original = tf.signal.fft2d(tf.cast(gray, tf.complex64), name=None)
plt.subplot(152), plt.imshow(tf.math.log(1 + tf.abs(original)), "gray"), plt.title("Spectrum")

center = tf.signal.fftshift(original)
plt.subplot(153), plt.imshow(tf.math.log(1 + np.abs(center)), "gray"), plt.title("Centered Spectrum (Before)")
_center = center.numpy()
_center[cY - size:cY + size, cX - size:cX + size] = 0
center = tf.convert_to_tensor(_center)
plt.subplot(153), plt.imshow(tf.math.log(1 + np.abs(center)), "gray"), plt.title("Centered Spectrum (After)")

inv_center = tf.signal.ifftshift(center)
plt.subplot(154), plt.imshow(tf.math.log(1 + np.abs(inv_center)), "gray"), plt.title("Decentralized")

recon = tf.signal.ifft2d(inv_center)
plt.subplot(155), plt.imshow(tf.math.abs(recon), "gray"), plt.title("Processed Image")

plt.show()

# %%
img = tf.image.decode_image(tf.io.read_file(img_path))
gray = rgbtogray(img)
res = bluriness_score_fft(gray)
# %%

# %%
f1 = diqa_emb_model.predict(np.expand_dims(orig, axis=0))
f1 = (f1 - np.mean(f1)) / (np.max(f1) - np.min(f1))
# %%
f2 = nima_emb_model.predict(
    np.expand_dims(cv2.resize(orig, nima_emb_model.input_shape[1:-1],
                              interpolation=cv2.INTER_CUBIC), axis=0)
)
f2 = (f2 - np.mean(f2)) / np.std(f2)
# %%

# orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
blur_orig = cv2.GaussianBlur(orig, (5, 5), 0)
blur_gray = cv2.cvtColor(blur_orig, cv2.COLOR_BGR2GRAY)
# %%

blur_orig_exp = np.expand_dims(blur_orig, 0)


# %%
lap = cv2.Laplacian(blur_gray, cv2.CV_64F)
fm, maximum = lap.var(), np.amax(lap)
# %%

from scipy.ndimage import variance
from skimage.filters import laplace
# %%
# Grayscale image
# blur_gray = rgb2gray(blur_orig)

# Edge detection
edge_laplace = laplace(blur_gray, ksize=3)

# Print output
print(f"Variance: {variance(edge_laplace)}")
print(f"Maximum : {np.amax(edge_laplace)}")

f4 = np.array([variance(edge_laplace), np.amax(edge_laplace)])

# %%

plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)

img_c2 = np.fft.fft2(blur_gray)
img_c3 = np.fft.fftshift(img_c2)
img_c4 = np.fft.ifftshift(img_c3)
img_c5 = np.fft.ifft2(img_c4)

plt.subplot(151), plt.imshow(blur_gray, "gray"), plt.title("Original Image")
plt.subplot(152), plt.imshow(np.log(1 + np.abs(img_c2)), "gray"), plt.title("Spectrum")
plt.subplot(153), plt.imshow(np.log(1 + np.abs(img_c3)), "gray"), plt.title("Centered Spectrum")
plt.subplot(154), plt.imshow(np.log(1 + np.abs(img_c4)), "gray"), plt.title("Decentralized")
plt.subplot(155), plt.imshow(np.abs(img_c5), "gray"), plt.title("Processed Image")


# %%
