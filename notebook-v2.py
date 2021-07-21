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
from src.diqa.utils import image_preprocess
# %%
import os
import tensorflow as tf
import cv2
from scipy.ndimage import variance, mean, maximum
from skimage.color import rgb2gray

# %%
from src.data_pipeline.dataset import Tid2013RecordDataset
_tfrecord = Tid2013RecordDataset()
tfrecord_path = _tfrecord.write_tfrecord_dataset("/Volumes/SDM/Dataset/image_quality/tid2013", "mos.csv")
# %%
tfrecord_path = os.path.expanduser("~/tensorflow_dataset/tid2013/data.tf.records")
# %%
ds = _tfrecord.load_tfrecord_dataset(tfrecord_path)
# %%
x = next(iter(ds.take(1)))

# %%
from src.data_pipeline.dataset import CSIQRecordDataset
_tfrecord = CSIQRecordDataset()
# %%
tfrecord_path = _tfrecord.write_tfrecord_dataset("/Volumes/SDM/Dataset/image_quality/CSIQ", "csiq.csv")
# %%
ds = _tfrecord.load_tfrecord_dataset(tfrecord_path)
dist_img, ref_img, distortion, dmos, dmos_std = next(iter(ds.take(1)))
# %%
from src.data_pipeline.dataset import LiveRecordDataset
_tfrecord = LiveRecordDataset()
tfrecord_path = _tfrecord.write_tfrecord_dataset("/Volumes/SDM/Dataset/image_quality/live", "dmos.csv")
# %%
ds = _tfrecord.load_tfrecord_dataset(tfrecord_path)
distorted_image, reference_image, distortion, dmos, dmos_realigned, dmos_realigned_std = next(iter(ds.take(1)))
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


# %% [markdown]
# ## NIMA features
from keras import backend as K
import importlib
import tensorflow


def earth_movers_distance(y_true, y_pred):
    cdf_true = K.cumsum(y_true, axis=-1)
    cdf_pred = K.cumsum(y_pred, axis=-1)
    emd = K.sqrt(K.mean(K.square(cdf_true - cdf_pred), axis=-1))
    return K.mean(emd)


class Nima(object):
    def __init__(self, base_model_name, n_classes=10, learning_rate=0.001, dropout_rate=0,
                 loss=earth_movers_distance, decay=0, weights='imagenet'):
        self.n_classes = n_classes
        self.base_model_name = base_model_name
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.loss = loss
        self.decay = decay
        self.nima_model = None
        self.weights = weights
        self._get_base_module()

    def _get_base_module(self):
        # import Keras base model module
        if self.base_model_name == 'InceptionV3':
            self.base_module = importlib.import_module('tensorflow.keras.applications.inception_v3')
        elif self.base_model_name == 'InceptionResNetV2':
            self.base_module = importlib.import_module('tensorflow.keras.applications.inception_resnet_v2')
        else:
            self.base_module = importlib.import_module('tensorflow.keras.applications.' + self.base_model_name.lower())

    def build(self):
        # get base model class
        BaseCnn = getattr(self.base_module, self.base_model_name)

        # load pre-trained model
        self.base_model = BaseCnn(input_shape=(224, 224, 3), weights=self.weights, include_top=False, pooling='avg')

        # add dropout and dense layer
        x = tf.keras.layers.Dropout(self.dropout_rate)(self.base_model.output)
        x = tf.keras.layers.Dense(units=self.n_classes, activation='softmax')(x)

        self.nima_model = tf.keras.models.Model(self.base_model.inputs, x)

    def compile(self):
        self.nima_model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate, decay=self.decay),
                                loss=self.loss)

    def preprocessing_function(self):
        return self.base_module.preprocess_input


weights_file = "weights/nima/MobileNet/weights_mobilenet_technical_0.11.hdf5"
base_model_name = "MobileNet"
nima = Nima(base_model_name, weights=None)
nima.build()
nima.nima_model.load_weights(weights_file)

# %%
diqa_emb_model = tf.keras.Model(objective_error_map.input, v, name='emb_output')
nima_emb_model = nima.base_model

# %%
dir_path = "/Volumes/SDM//-IQA/IQA_500/GOOD"
img_path = os.path.join(dir_path, "4016_3000CD0118970_APPROVED_DISB_PAN.jpg")
# %%
from src.classifier import compute_feature

feat = compute_feature(diqa_emb_model, nima)
_, feature = feat(img_path)
# %%
def bluriness_score_fft(image, size=60, thresh=10, vis=False):
    def visualize(image, magnitude):
        import matplotlib.pyplot as plt
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
    assert isinstance(image, tf.Tensor) or isinstance(image, np.ndarray), \
        "Not a valid image format, should be tf.tensor or numpy.ndarray"
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    fft = tf.signal.fft2d(tf.cast(image, tf.complex64), name=None)
    fftShift = tf.signal.fftshift(fft)
    if vis:
        magnitude = tf.math.log(tf.abs(fftShift))
        visualize(image, fftShift)
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = tf.signal.ifftshift(fftShift)
    recon = tf.signal.ifft2d(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = tf.math.log(tf.math.abs(recon))
    mean = tf.math.reduce_mean(magnitude)
    variance = tf.math.reduce_variance(magnitude)
    return (mean, variance, 20 * mean <= thresh)


def rgbtogray(image: tf.Tensor):
    image = tf.squeeze(tf.image.rgb_to_grayscale(image))
    image = tf.image.convert_image_dtype(image, tf.dtypes.float32)
    return image


# %%
get_tf_statistics = (lambda magnitude: (tf.math.reduce_mean(magnitude, name='magnitude_mean'),
                                        tf.math.reduce_variance(magnitude, name='magnitude_var'),
                                        tf.math.reduce_max(magnitude, name='magnitude_max')))

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
# from scipy.ndimage import variance, mean, maximum
get_statistics = (lambda im: (mean(im), variance(im)))

func_normalize_img = (lambda img: cv2.normalize(img, None, alpha=0, beta=1,
                                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

# %%
def detect_blur_laplace(image):
    if len(image.shape) == 3:
        image = rgb2gray(image)
    edge_laplace = laplace(image, ksize=3)
    # if len(image.shape) == 3:
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge_laplace = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
    return edge_laplace

# %%
def detect_blur_fft(image, size=60):
    # if len(image.shape) == 3:
    #     image = rgb2gray(image)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = np.log(np.abs(recon))
    # magnitude = 20 * np.log(np.abs(recon))
    return magnitude


# %%
dir_path = "/Volumes/SDM//-IQA/IQA_500/GOOD"
img_path = os.path.join(dir_path, "4016_3000CD0118970_APPROVED_DISB_PAN.jpg")
# %%
im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
# %%
edge = detect_blur_laplace(im)
get_statistics(edge)
# %%
edge_fft = detect_blur_fft(im)
get_statistics(edge_fft)
# %%
from src.classifier import compute_feature
feat = compute_feature(diqa_emb_model, nima_emb_model)
# %%
features = feat(img_path)
# %%
from skimage import io
orig = io.imread(img_path)
feat = detect_blur_fft(orig)
# %% [markdown]
# ## LBP bottleneck feature

# compute the Local Binary Pattern representation
# of the image, and then use the LBP representation
# to build the histogram of patterns
NUM_POINTS, RADIUS = 24, 8
LBP_MODEL_NAME = "weights/lbp/lbp-score-model.pkl"
model = pickle.load(open(LBP_MODEL_NAME, 'rb'))

# %%
orig = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
# orig = imutils.resize(orig, width=500)
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

plt.imshow(gray, cmap='gray')


# %%
from src.classifier import prepare_dataset, DEEP_EMD_MODELS
file_pattern = "/Volumes/SDM//-IQA/IQA_500/*/*.jpg"
dem = DEEP_EMD_MODELS(diqa_emb_model, nima)
prepare_dataset(file_pattern, dem)

# %%
from src.classifier import build_classfier
features_path, images_name, clf_path = "/Volumes/SDM//-IQA/IQA_500/iqa_feat.csv", \
    "/Volumes/SDM//-IQA/IQA_500/img_name.txt", "clf_path.pkl"
clf = build_classfier(features_path, images_name, clf_path)

# %%
from sklearn.linear_model import LogisticRegression


def build_classfier(feat_path: str, img_names: str, output_fp: str):
    # NOTE: All features should be in the same scale 0, 1 else prediction will go haywire
    # best_param = {'C': 10, 'gamma': 1, 'kernel': 'rbf'}  # Best parameter are evaluated using gridsearch
    # model = SVC(**best_param)
    features = np.loadtxt(feat_path, delimiter=",")
    images_path = tf.io.gfile.GFile(img_names, 'r').readlines()
    labels = list(map(lambda x: 1 if os.path.dirname(x).split("/")[-1] == "BAD" else 0,
                      images_path))
    # scaler = StandardScaler()
    # features = scaler.fit_transform(features)
    model = LogisticRegression()
    model.fit(features, labels)
    # model.fit(features, labels)
    pickle.dump(model, open(output_fp, 'wb'))
    return model


clf = build_classfier(features_path, images_name, clf_path)
# %%
import albumentations as A
from PIL import Image
orig = np.array(Image.open("/Users/sandipdey/Downloads/4016_3000CD0118970_APPROVED_DISB_PAN.jpg"))
transform = A.augmentations.transforms.MotionBlur(blur_limit=(7, 9), p=1)
augment = transform(image=orig)['image']
plt.imshow(augment)


# %%
from src.classifier import compute_feature
feat = compute_feature(diqa_emb_model, nima)
test_dir = "/Volumes/SDM//-IQA/IQA_100"
# %%
_, feature = feat(os.path.join(test_dir, "GOOD/4016_3000CD0118873_CANCELLED_CNCLD_PAN.jpg"))
# %%
_, feature = feat(os.path.join(test_dir, "BAD/4016_3003CD0064141_CANCELLED_CNCLD_VOTER.jpg"))
# %%
_, feature = feat(os.path.join(test_dir, "BAD/4016_3003CD0064270_CANCELLED_CNCLD_PAN.jpg"))
# %%
_, feature = feat(os.path.join(test_dir, "BAD/4019-26026005890-PAN.jpg"))
# %%
_, feature = feat(os.path.join(test_dir, "GOOD/4016_3000CD0118941_APPROVED_DISB_VOTER.jpg"))

# %%
test_pos_dir = "/Volumes/SDM//-IQA/IQA/GOOD"
test_neg_dir = "/Volumes/SDM//-IQA/IQA/BAD"

# %%
import glob2
features = [feat(ip)[1] for ip in glob2.glob(test_neg_dir + "/*.jpg")]
# %%
clf.predict(np.array(features))
# unique, counts = np.unique(x, return_counts=True)
# %%
features = [feat(ip)[1] for ip in glob2.glob(test_pos_dir + "/*.jpg")]
clf.predict(np.array(features))
# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# %%
# feature = scaler.fit_transform(feature)
clf.predict(feature[np.newaxis, ...])
# %%
from skimage import feature
from scipy.ndimage import variance
from skimage.filters import laplace


def _compute_lbp_features(image, eps=1e-7):
    lbp = feature.local_binary_pattern(image, NUM_POINTS, RADIUS, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, NUM_POINTS + 3), range=(0, NUM_POINTS + 2))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist

# feat = compute_lbp_features(gray)
# feat = feat.reshape(1, -1)
# model.predict(feat)


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
from sklearn.svm import SVC

best_param = {'C': 10, 'gamma': 1, 'kernel': 'rbf'}  # Best parameter are evaluated using gridsearch
model = SVC(**best_param)

# features = np.concatinate((f1, f2, f3, f4), axis=1)
# labels = 1  # 0 defines GOOD and 1 defines BAD
# model.fit(features, labels)
# predictions = model.predict(features)

# %%
clf_model_path = "weights/im_quality_clf.pkl"
pickle.dump(model, open(clf_model_path, 'wb'))

# %% [markdown]
# #### Scatter plot for blur projection


# %%

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
