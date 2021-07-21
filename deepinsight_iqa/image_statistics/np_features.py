from collections import defaultdict
from scipy.stats import itemfreq
from skimage import feature
import operator
import cv2
import os
import numpy as np
from PIL import Image
from scipy.ndimage import variance, mean, maximum
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import laplace
from skimage.transform import resize
from .utility import visualize

DULLNESS_THRESHOLD = 0.15
BRIGHTSPOT_THRESHOLD = 252

get_statistics = (lambda im: (mean(im), variance(im), maximum(im)))


def perform_color_analysis(path):
    im = Image.open(path)  # .convert("RGB")

    # cut the images into two halves as complete average may give bias results
    size = im.size
    halves = (size[0] / 2, size[1] / 2)
    im1 = im.crop((0, 0, size[0], halves[1]))
    im2 = im.crop((0, halves[1], size[0], size[1]))

    try:
        light_percent1, dark_percent1 = _color_analysis(im1)
        light_percent2, dark_percent2 = _color_analysis(im2)
    except Exception as e:
        return None

    light_percent = (light_percent1 + light_percent2) / 2
    dark_percent = (dark_percent1 + dark_percent2) / 2

    return dark_percent, light_percent


def _color_analysis(img):
    # obtain the color palatte of the image
    palatte = defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1

    # sort the colors present in the image
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse=True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]):  # dull : too much darkness
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]):  # bright : too much whiteness
            light_shade += x[1]
        shade_count += x[1]

    light_percent = round((float(light_shade) / shade_count) * 100, 2)
    dark_percent = round((float(dark_shade) / shade_count) * 100, 2)
    return light_percent, dark_percent


def blur_fft(image, size=60, thresh=10, vis=False):
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


def bluriness_score(img_path, use_cv=True):
    """
    #### Image Blurrness

    To measure the image blurrness, I refered to the following paper: 
    "Diatom Autofocusing in Brightfield Microscopy: A Comparative Study".

    In this paper the author Pech-Pacheco et al. has provided variance of the Laplacian Filter 
    which can be used to measure if the image blurryness score.

    In this technique, the single channel of an image is convolved  with the the laplacian filter. 
    If the specified value is less than a threshold value, then image is blurry otherwise not.

    """
    
    if use_cv:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(image, cv2.CV_64F)
        fm, maximum = lap.var(), np.amax(lap)
        return fm, maximum
    else:
        img = resize(io.imread(img_path), (400, 600))
        img = rgb2gray(img)

        # Edge detection
        edge_laplace = laplace(img, ksize=3)
        return variance(edge_laplace), maximum(edge_laplace)


def bluriness_score_fft(image, size=60):
    image = rgb2gray(image) if len(image.shape) == 3 else image
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    # magnitude = 20 * np.log(np.abs(recon))
    magnitude = np.log(1 + np.abs(recon))

    return variance(magnitude), maximum(magnitude)


def average_pixel_width(path):
    """
    #### 3. Uniform Images (with no pixel variations)

    #### Feature 3 - Average Pixel Width (using edge detection)

    Some images may contain no pixel variation and are entirely uniform. 
    Average Pixel Width is a measure which indicates the amount of edges present in the image. 
    If this number comes out to be very low, then the image is most likely a uniform image and may 
    not represent right content.

    To compute this measure, we need to use skimage's Canny Detection

    """
    im = Image.open(path)
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0] * im.size[1]))
    return apw * 100


def get_dominant_color(path):
    """
    #### Dominant Color

    ## 4. What are the key colors used in the image ?

    Colors used in the images play a significant role in garnering the attraction from users. 
    Additional features related to colors such as Dominant and Average colors can be created.

    """
    img = cv2.imread(path)
    arr = np.float32(img)
    pixels = arr.reshape((-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    quantized = palette[labels.flatten()]
    quantized = quantized.reshape(img.shape)

    dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]
    return dominant_color


def get_average_color(path):
    img = cv2.imread(path)
    average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
    return average_color


# 5. Dimensions of the Image
# Too Big Images or Too Small Images might not be very good for generating good attraction.
# Users may skip viewing a very large or very small sized image. Hence for advertisers it is
# important to set precise dimensions and size of the image. Hence we can create additional features.

# - Image width
# - Image height
# - Image size

def get_size(filename, images_path=None):
    filename = os.path.join(images_path, filename)
    st = os.stat(filename)
    return st.st_size


def get_dimensions(filename, images_path=None):
    filename = os.path.join(images_path, filename)
    img_size = Image.open(filename).size
    return img_size


def bright_spot(img, threshold=BRIGHTSPOT_THRESHOLD):
    """
    Detect excess Illumination
    TODO: Check other thresholding technique
    """
    assert img is not None, "Image path not available"

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)

    is_excess = thresh.any()
    return is_excess


def dullness(img, threshold=float(50.0 / 255.0)):
    """
    Dull score base on color thresholding
    """
    assert img is not None, "Image path not available"

    _, red_thresh = cv2.threshold(img[..., 0], threshold, 255.0, cv2.THRESH_BINARY)
    _, blue_thresh = cv2.threshold(img[..., 1], threshold, 255.0, cv2.THRESH_BINARY)
    _, green_thresh = cv2.threshold(img[..., 2], threshold, 255.0, cv2.THRESH_BINARY)
    pixels = np.multiply(*img.shape[:-1])
    perc_darkred = round(len(red_thresh) / pixels, 2)
    perc_darkblue = round(len(blue_thresh) / pixels, 2)
    perc_darkgreen = round(len(green_thresh) / pixels, 2)
    dullness_score = min(perc_darkblue, perc_darkred, perc_darkgreen)
    return {"score": dullness_score,
            "pixel_percentage": [perc_darkred, perc_darkgreen, perc_darkblue]}


def dullness_factor_score(img, dullness_thresh=DULLNESS_THRESHOLD):
    assert img is None, "Image path not available"

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # plt.hist(img.ravel(), 256, [0, 256])
    # plt.title('Histogram for gray scale picture')
    # plt.show()
    shadowy = hist[0:70]
    illuminated = hist[70: 250]
    shadow_pixels_n = 0
    illuminated_pixels_n = 0
    for p in shadowy:
        shadow_pixels_n += p[0]

    for p in illuminated:
        illuminated_pixels_n += p[0]

    dark_perc = shadow_pixels_n / (shadow_pixels_n + illuminated_pixels_n)
    illum_perc = illuminated_pixels_n / (shadow_pixels_n + illuminated_pixels_n)
    dullness_factor = illuminated_pixels_n / shadow_pixels_n
    dull_not_dull: bool = True if dullness_factor < dullness_thresh else False
    return dullness_factor, (illum_perc, dark_perc), dull_not_dull


# ### ------ TENSORFLOW IMAGE STATISTICS ### ------###
