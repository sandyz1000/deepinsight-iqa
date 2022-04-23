# %%
%load_ext autoreload
%autoreload 2
%reload_ext autoreload
# %%
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
# sys.path.append(os.path.realpath(os.path.pardir))
from deepinsight_iqa.diqa.data import get_iqa_datagen
from deepinsight_iqa.diqa.utils.tf_imgutils import image_normalization, image_preprocess
from deepinsight_iqa.diqa.predict import Prediction
from deepinsight_iqa.diqa.trainer import Trainer
from deepinsight_iqa.cli import parse_config
from deepinsight_iqa.common.utility import set_gpu_limit
set_gpu_limit(10)

job_dir = os.path.realpath(os.path.curdir)

# ## Set image directory and path
# %%
image_dir = "image_quality_data/data"
csv_path = "combine.csv"
# cfg_path = "configs/diqa/mobilenet.json"
# cfg_path = "configs/diqa/inceptionv3.json"
# cfg_path = "configs/diqa/resnetv2.json"
cfg_path = "configs/diqa/default.json"
# resolve_config_path = (lambda cfg_path: Path(os.path.dirname(__file__)) / cfg_path)
# cfg = parse_config(resolve_config_path(cfg_path))
cfg = parse_config(cfg_path)
# %%
train, valid = get_iqa_datagen(
    image_dir,
    os.path.join(image_dir, csv_path),
    do_augment=cfg['use_augmentation'],
    image_preprocess=image_preprocess,
    input_size=cfg['input_size'],
    batch_size=cfg['batch_size'],
    channel_dim=1,
    do_train=True
)
# %%
it = iter(train)
X_dist, dist_gray, X_ref, Y = next(it)
plt.imshow(X_ref[0], cmap='gray')
# %%
network = 'objective'
cfg.pop('network')
model_dir = cfg.pop('model_dir', 'weights/diqa')

# %%

trainer = Trainer(
    train, valid,
    network=network,
    model_dir=model_dir,
    **cfg
)
diqa = trainer.compile(train_bottleneck=True)
# %%
# weight_file = cfg.pop('weight_file', 'objective-model-custom-1650708810.9949222.h5')
# model_path = Path(model_dir) / weight_file
# diqa.build(input_shape=(None, None, 1))
# trainer.load_weights(diqa, model_path)

# %%
# trainer.train(diqa)
# %%
trainer.slow_trainer(diqa)
# %%
# trainer.save_weights(diqa)
diqa.save("temp-1", save_format='tf')
# %%
image_dir, csv_path = "/Volumes/SDM/Dataset/iqa", "technical/combine.csv"
config_file = os.path.realpath(os.path.join(job_dir, "confs/diqa_inceptionv3.json"))
cfg = parse_config(job_dir, config_file)
model_dir = os.path.join(os.path.expanduser('~/Documents/utilities-github/deepinsight-iqa'), cfg['model_dir'])
prediction = Prediction(
    model_dir=model_dir, subjective_weightfname=cfg['subjective_weightfname'],
    model_type=cfg['base_model_name']
)

# %%
import pandas as pd

df = pd.read_csv(os.path.join(image_dir, "PhotoQualitySampleSheet2.csv"))

# %%
from pandarallel import pandarallel

# %%
pandarallel.initialize(nb_workers=4)


# %%
# df['diqa_mobilenet'] = df['url'].apply(download_npredict)
# df['diqa_inceptionv3'] = df['url'].apply(download_npredict)
os.makedirs('images', exist_ok=True)
# df['url'].parallel_apply(download_files)
download_files(df['url'][0])
# df['diqa'] = parallelize_on_rows(df, download_npredict)

# %%

# %% [markdown]
# ## Verify Data generator for AVA, CSIQ, TID2013, LIVE
# %%


# %%
def visualize(image, fftShift):
    magnitude = 20 * np.log(np.abs(fftShift))
    (fig, ax) = plt.subplots(1, 2, )
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Input")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    # display the magnitude image
    ax[1].imshow(magnitude, cmap="gray")
    ax[1].set_title("Magnitude Spectrum")
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.show()


def tf_detect_blur_fft(image, size=60, thresh=10, vis=False):
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    fft = tf.signal.fft2d(image, name=None)
    fftShift = tf.signal.fftshift(fft)
    if vis:
        visualize(image, fftShift)
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = tf.signal.ifftshift(fftShift)
    recon = tf.signal.ifft2d(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    return (mean, mean <= thresh)


def detect_blur_fft(image, size=60, thresh=10, vis=False):
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    if vis:
        visualize(image, fftShift)

    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return (mean, mean <= thresh)


# %%
dir_path = "/Volumes/SDM/IQA_500/GOOD"
img_path = os.path.join(dir_path, "4016_3000CD0118970_APPROVED_DISB_PAN.jpg")
thresh, vis = 20, True
orig = cv2.imread(img_path)
# orig = imutils.resize(orig, width=500)
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
# %%
from PIL import Image
# apply our blur detector using the FFT
gray = cv2.GaussianBlur(gray, (9, 9), 0)
(mean, blurry) = detect_blur_fft(gray, size=30, thresh=thresh, vis=vis)
# plt.imshow(gray, cmap='gray')

# %%
# draw on the image, indicating whether or not it is blurry
image = np.dstack([gray] * 3)
color = (0, 0, 255) if blurry else (0, 255, 0)
text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
text = text.format(mean)
cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
print("[INFO] {}".format(text))
cv2.imshow("Output", image)
cv2.waitKey(0)
# %%
plt.imshow(image)

# %%
gray = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
cvt_images = []
for radius in range(1, 30, 2):
    _gray = cv2.GaussianBlur(gray, (radius, radius), 0)
    # apply our blur detector using the FFT
    (mean, blurry) = detect_blur_fft(_gray, size=60, thresh=thresh, vis=False)
    _gray = np.dstack([_gray] * 3)
    color = (0, 0, 255) if blurry else (0, 255, 0)
    text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
    text = text.format(mean)
    cv2.putText(_gray, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    print("[INFO] Kernel: {}, Result: {}".format(radius, text))
    cvt_images.append(_gray)

# %%

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)

img_c1 = cv2.imread(img_path, 0)
img_c2 = np.fft.fft2(img_c1)
img_c3 = np.fft.fftshift(img_c2)
img_c4 = np.fft.ifftshift(img_c3)
img_c5 = np.fft.ifft2(img_c4)

plt.subplot(151), plt.imshow(img_c1, "gray"), plt.title("Original Image")
plt.subplot(152), plt.imshow(np.log(1 + np.abs(img_c2)), "gray"), plt.title("Spectrum")
plt.subplot(153), plt.imshow(np.log(1 + np.abs(img_c3)), "gray"), plt.title("Centered Spectrum")
plt.subplot(154), plt.imshow(np.log(1 + np.abs(img_c4)), "gray"), plt.title("Decentralized")
plt.subplot(155), plt.imshow(np.abs(img_c5), "gray"), plt.title("Processed Image")

# %%
