from concurrent.futures import ThreadPoolExecutor
import glob2
import concurrent.futures
from skimage.color import rgb2gray
import numpy as np
import os
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from skimage import feature
from skimage.filters import laplace
from scipy.ndimage import variance, mean
import dill as pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import namedtuple
from .diqa.utils import image_preprocess
import tqdm

NUM_POINTS, RADIUS = 24, 8

DEEP_EMD_MODELS = namedtuple('DEEP_EMD_MODELS', ('diqa_emb_model', 'nima_emb_model', ))


func_normalize_img = (lambda img: cv2.normalize(img, None, alpha=0, beta=1,
                                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

get_statistics = (lambda im: (mean(im), variance(im)))


def _compute_hist(features, eps=1e-7, bins=np.arange(0, 11)):
    (hist, _) = np.histogram(features.ravel(), bins=bins)
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist


def lbp_features(image: np.ndarray, eps=1e-7):
    if len(image.shape) == 3:
        image = rgb2gray(image)
    lbp = feature.local_binary_pattern(image, NUM_POINTS, RADIUS, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, NUM_POINTS + 3), range=(0, NUM_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist


def blur_detection_laplace(image, method='laplace', size=60):
    if len(image.shape) == 3:
        image = rgb2gray(image)
    
    edges = laplace(image, ksize=3)
    return edges


def blur_detection_fft(image, size=60):
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
    return magnitude


class compute_feature:
    """
    f1: 10d hist of 128 dimension features
    f2: 10d hist of 1024 dimension bottleneck features
    f3: hist of 26 dimension features computed from LBP
    f4: mean, variance (using FFT)
    f5: mean, variance (using Laplace)
    """

    def __init__(self, diqa_emb_model, nima):
        nima_preprocessing_func = nima.preprocessing_function()
        nima_emb_model = nima.base_model

        def _compute_diqa_feature(orig):
            img = tf.convert_to_tensor(orig, dtype=tf.float32)
            I_d = image_preprocess(img)
            I_d = tf.tile(I_d, (1, 1, 1, 3))
            f1 = diqa_emb_model.predict(np.expand_dims(orig, axis=0))
            hist = _compute_hist(f1)
            return hist

        def _compute_nima_feature(orig, eps=1e-7):
            # input_shape = nima_emb_model.input_shape[1:-1]
            orig = nima_preprocessing_func(orig)
            f2 = nima_emb_model.predict(np.expand_dims(orig, axis=0))
            hist = _compute_hist(f2)
            return hist

        self._compute_diqa_feature = _compute_diqa_feature
        self._compute_nima_feature = _compute_nima_feature

    def __call__(self, img_path: str):
        orig = np.array(Image.open(img_path))
        feature = [func(orig) for func in [
            self._compute_diqa_feature, self._compute_nima_feature, lbp_features]
        ]
        feature.append(get_statistics(func_normalize_img(blur_detection_laplace(orig))))
        feature.append(get_statistics(func_normalize_img(blur_detection_fft(orig))))
        feature = np.concatenate(feature, axis=0)
        return img_path, feature


def prepare_dataset(file_pattern: str, dem: DEEP_EMD_MODELS, output_csv: str = 'iqa_feat.csv', fp="img_name.txt"):
    dirname = os.path.dirname(file_pattern).replace("*", "")
    images_path = glob2.glob(file_pattern)
    func = compute_feature(dem.diqa_emb_model, dem.nima_emb_model)
    results = []
    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     for future in tqdm.tqdm(concurrent.futures.as_completed([
    #         executor.submit(func, path) for path in images_path
    #     ]), total=len(images_path)):
    #         results.append(future.result())
    results = [func(path) for path in images_path]

    img_paths, features = list(zip(*results))
    np.savetxt(os.path.join(dirname, output_csv), features, delimiter=",")
    with open(os.path.join(dirname, fp), 'w') as f:
        f.write("\n".join(img_paths))


def build_classfier(feat_path: str, img_names: str, output_fp: str):
    """Build SVM classifier with all the features calculated above

    :param feature_file: [description]
    :type feature_file: str
    """
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
    pickle.dump(model, open(output_fp, 'wb'))
    return model
