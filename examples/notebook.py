# %%
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import tensorflow as tf
sys.path.append(os.path.realpath(os.pardir))
from deepinsight_iqa.diqa.data import get_combine_datagen
from deepinsight_iqa.diqa.utils.tf_imgutils import image_normalization, image_preprocess


def parse_config(job_dir, config_file):
    os.makedirs(os.path.join(job_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(job_dir, 'logs'), exist_ok=True)
    config = json.load(open(config_file, 'r'))
    return config


job_dir = os.path.realpath(os.path.pardir)
# %% [markdown]
# ## Set image directory and path
# %%
image_dir, csv_path = "/Users/sdey/Documents/dataset/image-quality-assesement", "combine.csv"
config_file = os.path.realpath(os.path.join(job_dir, "confs/diqa_mobilenet.json"))
cfg = parse_config(job_dir, config_file)
train, valid = get_combine_datagen(
    image_dir, csv_path, do_augment=cfg['use_augmentation'],
    image_preprocess=image_preprocess, input_size=cfg['input_size']
)
# %%
it = iter(train)
X_dist, X_ref, Y = next(it)

# %%
from deepinsight_iqa.diqa.train import Trainer
trainer = Trainer(train, valid, **cfg)
# %%
from deepinsight_iqa.diqa.predict import Prediction
from urllib.request import urlopen
from urllib.error import HTTPError


# %%
model_dir = os.path.join(os.path.expanduser('~/Documents/utilities-github/deepinsight-iqa'), cfg['model_dir'])
prediction = Prediction(
    model_dir=model_dir, final_wts_filename=cfg['final_wts_filename'],
    base_model_name=cfg['base_model_name']
)

# %%
import pandas as pd

df = pd.read_csv(os.path.join(image_dir, "PhotoQualitySampleSheet2.csv"))

# %%
count = 0
def download_npredict(url, readFlag=cv2.IMREAD_COLOR):
    """
    Download the image, convert it to a NumPy array, and then read it into OpenCV format
    """
    try:
        resp = urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, readFlag)
        score = prediction.predict(image)
        count += 1
        return score
    except Exception:
        return None

df['diqa'] = df['url'].apply(download_npredict)
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
