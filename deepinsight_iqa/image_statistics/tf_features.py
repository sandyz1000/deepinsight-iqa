import os
import math
import typing
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops import io_ops
DULLNESS_THRESHOLD = 0.15
BRIGHTSPOT_THRESHOLD = 252
DARKSPOT_THRESHOLD = 20


def laplacian(image, size):
    if len(image.shape) == 3 and image.shape[2] == 3:  # convert rgb to grayscale
        image = tf.image.convert_image_dtype(image, tf.dtypes.float32)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.convert_image_dtype(image, tf.dtypes.float32)
        image = tf.squeeze(image, 2)

    image = tf.convert_to_tensor(image, dtype=tf.float32)

    new = tf.image.convert_image_dtype(image, tf.dtypes.float32)
    new = tf.expand_dims(new, 2)
    image = tf.expand_dims(new, 0)

    fil = np.ones([size, size])
    fil[int(size / 2), int(size / 2)] = 1.0 - size**2
    fil = tf.convert_to_tensor(fil, tf.float32)
    fil = tf.stack([fil] * 1, axis=2)
    fil = tf.expand_dims(fil, 3)

    result = tf.nn.depthwise_conv2d(image, fil, strides=[1, 1, 1, 1], padding="SAME")
    result = tf.squeeze(result, 0)
    result = tf.squeeze(result, 2)

    result = result.numpy()
    minM = np.min(result)
    maxM = np.max(result)
    output = (result.astype('float') - minM) * 255 / (maxM - minM)
    return output


def laplacian_of_gaussian(image, filtersize, sigma):
    if len(image.shape) == 3 and image.shape[2] == 3:
        raise TypeError('Please input Grayscaled Image.')
    elif len(image.shape) > 3:
        raise TypeError('Incorrect number of channels.')
    n_channels = 1
    image = tf.expand_dims(image, 2)

    w = math.ceil(sigma * filtersize)
    w_range = int(math.floor(w / 2))

    y = x = tf.range(-w_range, w_range + 1, 1)
    Y, X = tf.meshgrid(x, y)
    z = tf.cast(tf.add(tf.square(X), tf.square(Y)), tf.float32)
    nom = tf.subtract(z, 2 * (sigma**2))
    denom = 2 * math.pi * (sigma**6)
    exp = tf.exp(-z / 2 * (sigma**2))
    fil = tf.divide(tf.multiply(nom, exp), denom)

    fil = tf.stack([fil] * n_channels, axis=2)
    fil = tf.expand_dims(fil, 3)

    new = tf.image.convert_image_dtype(image, tf.dtypes.float32)
    new = tf.expand_dims(new, 0)
    res = tf.nn.depthwise_conv2d(new, fil, strides=[1, 1, 1, 1], padding="SAME")
    res = tf.squeeze(res, 0)
    res = tf.squeeze(res, 2)

    result = res.numpy()
    minM = np.min(result)
    maxM = np.max(result)
    output = (result.astype('float') - minM) * 255 / (maxM - minM)
    return output


def basic_threshold(image: typing.Union[np.ndarray, tf.Tensor], threshold: int):
    image = tf.convert_to_tensor(image, name="image")

    rank = image.shape.rank
    if rank != 2 and rank != 3:
        raise ValueError("Image should be either 2 or 3-dimensional.")

    if not isinstance(threshold, int):
        raise ValueError("Threshold value must be an integer.")

    if rank == 3:
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.convert_image_dtype(image, tf.dtypes.uint8)
        image = tf.squeeze(image, 2)

    final = tf.where(image > threshold, True, False)
    return final


def adaptive_thresholding(image: typing.Union[np.ndarray, tf.Tensor]):
    """
    Thresholding using Histogram method
    """
    image = tf.convert_to_tensor(image, name="image")
    window = 40
    rank = image.shape.rank
    if rank != 2 and rank != 3:
        raise ValueError("Image should be either 2 or 3-dimensional.")

    if not isinstance(window, int):
        raise ValueError("Window size value must be an integer.")

    r, c = image.shape
    if window > min(r, c):
        raise ValueError("Window size should be lesser than the size of the image.")

    if rank == 3:
        image = tf.image.rgb_to_grayscale(image)
        image = tf.squeeze(image, 2)

    image = tf.image.convert_image_dtype(image, tf.dtypes.float32)

    i = 0
    final = tf.zeros((r, c))
    while i < r:
        j = 0
        r1 = min(i + window, r)
        while j < c:
            c1 = min(j + window, c)
            cur = image[i:r1, j:c1]
            thresh = tf.reduce_mean(cur)
            new = tf.where(cur > thresh, 255.0, 0.0)

            s1 = [x for x in range(i, r1)]
            s2 = [x for x in range(j, c1)]
            X, Y = tf.meshgrid(s2, s1)
            ind = tf.stack([tf.reshape(Y, [-1]), tf.reshape(X, [-1])], axis=1)

            final = tf.tensor_scatter_nd_update(final, ind, tf.reshape(new, [-1]))
            j += window
        i += window
    return final


def otsu_thresholding(image):
    """ Thresholding using Otsu's method """
    image = tf.convert_to_tensor(image, name="image")

    rank = image.shape.rank
    if rank != 2 and rank != 3:
        raise ValueError("Image should be either 2 or 3-dimensional.")

    if image.dtype != tf.int32:
        image = tf.cast(image, tf.int32)

    r, c = image.shape
    hist = tf.math.bincount(image, dtype=tf.int32)

    if len(hist) < 256:
        hist = tf.concat([hist, [0] * (256 - len(hist))], 0)

    current_max, threshold = 0, 0
    total = r * c

    spre = [0] * 256
    sw = [0] * 256
    spre[0] = int(hist[0])

    for i in range(1, 256):
        spre[i] = spre[i - 1] + int(hist[i])
        sw[i] = sw[i - 1] + (i * int(hist[i]))

    for i in range(256):
        if total - spre[i] == 0:
            break

        meanB = 0 if int(spre[i]) == 0 else sw[i] / spre[i]
        meanF = (sw[255] - sw[i]) / (total - spre[i])
        varBetween = (total - spre[i]) * spre[i] * ((meanB - meanF)**2)

        if varBetween > current_max:
            current_max = varBetween
            threshold = i

    final = tf.where(image > threshold, 255, 0)
    return final


def gaussFilter(fx: int, fy: int, sigma: int):
    """ Gaussian Filter
    """
    x = tf.range(-int(fx / 2), int(fx / 2) + 1, 1)
    Y, X = tf.meshgrid(x, x)

    sigma = -2 * (sigma**2)
    z = tf.cast(tf.add(tf.square(X), tf.square(Y)), tf.float32)
    k = 2 * tf.exp(tf.divide(z, sigma))
    k = tf.divide(k, tf.reduce_sum(k))
    return k


def gaussian_blur(image: tf.Tensor, filtersize: typing.List[int], sigma: int):
    n_channels = 3
    if len(image.shape) == 3 and image.shape[2] != 3:
        raise TypeError('Incorrect number of channels.')

    elif len(image.shape) == 3 and image.shape[2] == 3:
        n_channels = 3

    elif len(image.shape) == 2:
        image = tf.expand_dims(image, 2)
        n_channels = 1

    fx, fy = filtersize[0], filtersize[1]
    fil = gaussFilter(fx, fy, sigma)
    fil = tf.stack([fil] * n_channels, axis=2)
    fil = tf.expand_dims(fil, 3)

    new = tf.image.convert_image_dtype(image, tf.dtypes.float32)
    new = tf.expand_dims(new, 0)
    res = tf.nn.depthwise_conv2d(new, fil, strides=[1, 1, 1, 1], padding="SAME")

    res = tf.squeeze(res, 0)

    if n_channels == 1:
        res = tf.squeeze(res, 2)

    res = tf.image.convert_image_dtype(res, tf.dtypes.uint8)
    return res


def average_filter(image: tf.Tensor, filtersize: typing.List[int]):
    """Averaging Filter
    """
    n_channels = 3
    if len(image.shape) == 3 and image.shape[2] != 3:
        raise TypeError('Incorrect number of channels.')

    elif len(image.shape) == 3 and image.shape[2] == 3:
        n_channels = 3

    elif len(image.shape) == 2:
        image = tf.expand_dims(image, 2)
        n_channels = 1

    fx, fy = filtersize[0], filtersize[1]
    fil = tf.ones([fx, fy, n_channels], tf.float32) / (fx * fy)
    fil = tf.expand_dims(fil, 3)

    new = tf.image.convert_image_dtype(image, tf.dtypes.float32)
    new = tf.expand_dims(new, 0)
    res = tf.nn.depthwise_conv2d(new, fil, strides=[1, 1, 1, 1], padding="SAME")

    res = tf.squeeze(res, 0)

    if n_channels == 1:
        res = tf.squeeze(res, 2)

    res = tf.image.convert_image_dtype(res, tf.dtypes.uint8)
    return res


def tf_equalize_histogram(image):
    values_range = tf.constant([0., 255.], dtype=tf.float32)
    histogram = tf.histogram_fixed_width(tf.image.convert_image_dtype(image, tf.dtypes.float32), values_range, 256)
    cdf = tf.cumsum(histogram)
    cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]

    img_shape = tf.shape(image)
    pix_cnt = img_shape[-3] * img_shape[-2]
    px_map = tf.round(tf.cast(cdf - cdf_min, tf.float32) * 255. / tf.cast(pix_cnt - 1, tf.float32))
    px_map = tf.cast(px_map, tf.uint8)

    eq_hist = tf.expand_dims(tf.gather_nd(px_map, tf.cast(image, tf.int32)), 2)
    return eq_hist


def get_gradients(image, sigma):
    image = gaussian_blur(image, [5, 5], sigma)
    image = tf.image.convert_image_dtype(image, tf.dtypes.float32)
    image = tf.expand_dims(image, 0)
    image = tf.expand_dims(image, 3)
    sobel = tf.image.sobel_edges(image)
    sobel = tf.squeeze(sobel, 0)
    gx = sobel[:, :, :, 0]
    gx = tf.squeeze(gx, 2)

    gy = sobel[:, :, :, 1]
    gy = tf.squeeze(gy, 2)

    magnitude = tf.sqrt(tf.add(tf.square(gx), tf.square(gy)))
    gradient = tf.atan2(gy, gx)
    return magnitude, gradient


def nonmaxsupress(Gm, Gd):
    nms = np.zeros(Gm.shape)
    h, w = Gm.shape
    Gm = Gm.numpy()
    Gd = Gd.numpy()
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            angle = np.rad2deg(Gd[i, j]) % 180
            mag = Gm[i, j]

            if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                dx, dy = 0, 1
            elif (22.5 <= angle < 67.5):
                dx, dy = 1, -1
            elif (67.5 <= angle < 112.5):
                dx, dy = 1, 0
            elif (112.5 <= angle < 157.5):
                dx, dy = 1, 1

            if mag > Gm[i - dx, j - dy] and mag > Gm[i + dx, j + dy]:
                nms[i, j] = mag
    return nms


def canny_edge(image: tf.Tensor, sigma: int, low: int = 0.1, high: int = 0.3):
    """ Canny Edge Detector
    """
    def threshold2(NMS, low, high):
        weak, strong = 50, 255
        res1 = np.copy(NMS)
        res = np.asarray(res1, dtype=np.int32)
        res[res1 > high] = strong
        res[[res1 <= high] or [res1 >= low]] = weak
        res[res1 < low] = 0
        res = tf.convert_to_tensor(res, dtype=tf.uint8)
        return res
    if len(image.shape) == 3 and image.shape[2] == 3:
        raise TypeError('Please input Grayscaled Image.')
    elif len(image.shape) > 3:
        raise TypeError('Incorrect number of channels.')

    # Canny edge detector
    Gm, Gd = get_gradients(image, sigma)
    nms = nonmaxsupress(Gm, Gd)
    ret = threshold2(nms, low, high)
    return ret


def read_img(img_path):
    # Return tensor object for the image
    image = io_ops.read_file(img_path)
    color2 = tf.io.decode_png(image, channels=3, dtype=tf.dtypes.uint8, name=None)
    img = tf.image.convert_image_dtype(color2, tf.dtypes.float32)
    return img


# ## --- Method to be used for color analysis --- ##
def color_analysis(img: typing.Union[tf.Tensor, np.ndarray]):
    assert isinstance(img, tf.Tensor) or isinstance(img, np.ndarray), \
        "Not a valid image format, should be tf.tensor or numpy.ndarray"

    class DullnessScore:
        def __init__(self, color_prob, is_dull, dull_ratio, illumina_prob) -> None:
            self.is_dull = is_dull
            self.dull_ratio = dull_ratio
            self.color_prob = color_prob
            self.illumina_prob = illumina_prob

        def _to_dict(self):
            return {"isdull": self.is_dull, "dull_ratio": self.dull_ratio,
                    "color_prob": self.color_prob, "illumina_prob": self.illumina_prob}

    def bright_spot(img: tf.Tensor, threshold: int = BRIGHTSPOT_THRESHOLD):
        """
        Detect excess Illumination
        NOTE: Check other thresholding technique, OTSU,
        """
        assert img is not None, "Image tensor not available"
        img = basic_threshold(img, threshold)
        is_excess = img.numpy().any()
        return is_excess

    def dark_spot(img: tf.Tensor, threshold: int = DARKSPOT_THRESHOLD):
        """
        Detect Dark spot in an image
        """
        img = ~basic_threshold(img, threshold)
        is_excess = img.numpy().any()
        return is_excess

    def dullness(color_img: tf.Tensor, color_thresh=int(50.0),
                 dullness_thresh=DULLNESS_THRESHOLD,
                 shadow_range=(0, 70), illumi_range=(70, 250)):
        """
        Dull score base on color thresholding and image histogram illumination
        """
        assert color_img is not None, "Image tensor not available"
        size = tuple(color_img.shape)
        red = tf.expand_dims(basic_threshold(color_img[..., 0], color_thresh), -1)
        green = tf.expand_dims(basic_threshold(color_img[..., 1], color_thresh), -1)
        blue = tf.expand_dims(basic_threshold(color_img[..., 2], color_thresh), -1)
        total_pixels = tf.constant(np.prod(size[:-1]), dtype=tf.float32)
        pxl_density = tf.cast(tf.concat([red, green, blue], axis=2), tf.float32)
        pxl_density = tf.reshape(pxl_density, (-1, 3))
        color_prob = tf.divide(tf.reduce_sum(pxl_density, axis=0), total_pixels)

        # Gray scale dullness score
        edges = [i / (256 - 0) for i in range(256)]
        gray = tf.image.rgb_to_grayscale(color_img, name="dull-image")
        gray = tf.image.convert_image_dtype(gray, tf.dtypes.float32)
        hist = tfp.stats.histogram(gray, edges)

        shadow = tf.math.reduce_sum(hist[shadow_range[0]:shadow_range[1]])
        illuminate = tf.math.reduce_sum(hist[illumi_range[0]: illumi_range[1]])
        dark_prob = tf.divide(shadow, tf.add(shadow, illuminate))
        light_prob = tf.divide(illuminate, tf.add(shadow, illuminate))
        dull_odd = tf.math.divide(light_prob, dark_prob)
        is_dull = True if dull_odd < dullness_thresh else False

        # Calculate percentage of color
        response = DullnessScore(
            color_prob=color_prob.numpy().tolist(),
            is_dull=is_dull,
            dull_ratio=dull_odd.numpy().tolist(),
            illumina_prob={"dark": dark_prob.numpy().tolist(), "light": light_prob.numpy().tolist()}
        )
        return response._to_dict()

    img = tf.convert_to_tensor(img, dtype=tf.float32)
    # size = tf.constant(img.shape, dtype=tf.int32)

    # NOTE: cut the images into multiple proportion as complete average may give bias results,
    # It should be no of random crop and then find the scoring of Individual crop

    # halves = (size[0] / 2, size[1] / 2)
    # im1 = tf.image.crop_to_bounding_box(img, 0, 0, size[0], halves[1])
    # im2 = tf.image.crop_to_bounding_box(img, 0, halves[1], size[0], size[1])
    # images = [im1, im2]

    dullness_score = dullness(img)
    spots = {"dark": dark_spot(img), "bright": bright_spot(img)}
    return dullness_score, spots


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
    fft = tf.signal.fft2d(image, name=None)
    fftShift = tf.signal.fftshift(fft)
    if vis:
        magnitude = np.log(np.abs(fftShift))
        visualize(image, magnitude)
    # TODO: Mask pixel to zero for the given radius
    # tf.tensor_scatter_nd_update
    # fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = tf.signal.ifftshift(fftShift)
    recon = tf.signal.ifft2d(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = tf.math.log(np.abs(recon))
    mean = tf.math.reduce_mean(magnitude)

    return (mean, mean <= thresh)


def bluriness_score(im: typing.Union[tf.Tensor, np.ndarray]):
    """
    Image Blurrness
    ==================
    To measure the image blurrness, I refered to the following paper:
    "Diatom Autofocusing in Brightfield Microscopy: A Comparative Study".

    In this paper the author Pech-Pacheco et al. has provided variance of the Laplacian Filter
    which can be used to measure if the image blurryness score.

    In this technique, the single channel of an image is convolved  with the the laplacian filter.
    If the specified value is less than a threshold value, then image is blurry otherwise not.

    """
    assert isinstance(im, tf.Tensor) or isinstance(im, np.ndarray), \
        "Not a valid image format, should be tf.tensor or numpy.ndarray"

    im = tf.cast(im, tf.float32)
    edge_laplace = laplacian(im, 3)
    # variance = tfp.stats.variance(
    #     edge_laplace, sample_axis=0, keepdims=False, name=None
    # )
    # max_ = tf.math.reduce_max(edge_laplace)
    blur = 1 - tf.divide(tf.math.reduce_variance(edge_laplace), 255.0)
    return blur.numpy()


def average_pixel_width(im: typing.Union[tf.Tensor, np.ndarray]):
    """
    Uniform Images (with no pixel variations)
    ==================
    Feature 3 - Average Pixel Width (using edge detection)

    Some images may contain no pixel variation and are entirely uniform.
    Average Pixel Width is a measure which indicates the amount of edges present in the image.
    If this number comes out to be very low, then the image is most likely a uniform image and may
    not represent right content.

    To compute this measure, we need to use skimage's Canny Detection
    """
    assert isinstance(im, tf.Tensor) or isinstance(im, np.ndarray), \
        "Not a valid image format, should be tf.tensor or numpy.ndarray"

    canny = canny_edge(im, sigma=3)
    apw = (float(tf.math.reduce_sum(canny)) / (canny.shape[0] * canny.shape[1]))
    return apw * 100


def get_dominant_color(img: typing.Union[tf.Tensor, np.ndarray]):
    """
    Dominant Color
    ==================
    
    4. What are the key colors used in the image ?

    Colors used in the images play a significant role in garnering the attraction from users.
    Additional features related to colors such as Dominant and Average colors can be created.

    """
    import cv2
    from scipy.stats import itemfreq
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
    return dominant_color.tolist()


# 5. Dimensions of the Image
# Too Big Images or Too Small Images might not be very good for generating good attraction.
# Users may skip viewing a very large or very small sized image. Hence for advertisers it is
# important to set precise dimensions and size of the image. Hence we can create additional features.

# - Image width
# - Image height
# - Image size

def get_average_color(img: typing.Union[tf.Tensor, np.ndarray]):
    img = tf.image.convert_image_dtype(img, tf.dtypes.float32)
    x = tf.reshape(img, (-1, 3))
    avg_color = tf.math.reduce_mean(x, axis=0)
    return avg_color


def get_size(filepath):
    st = os.stat(filepath)
    return st.st_size


def rgbtogray(image: tf.Tensor):
    image = tf.squeeze(tf.image.rgb_to_grayscale(image))
    image = tf.image.convert_image_dtype(image, tf.dtypes.float32)
    return image
