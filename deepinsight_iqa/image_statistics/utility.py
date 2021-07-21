from PIL import Image as IMG
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
from threading import Lock


def plot_image(img_path):
    im1 = IMG.open(img_path)
    im2 = im1.convert(mode='L')
    im = np.asarray(im2)

    edges1 = feature.canny(im, sigma=1)
    edges2 = feature.canny(im, sigma=3)

    # display results
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)

    ax1.imshow(im, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('noisy image', fontsize=20)

    ax2.imshow(edges1, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

    ax3.imshow(edges2, cmap=plt.cm.gray)
    ax3.axis('off')
    ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

    fig.tight_layout()

    plt.show()


def thread_safe_memoize(func):
    cache = {}
    session_lock = Lock()

    def memoizer(*args, **kwargs):
        with session_lock:
            key = str(args) + str(kwargs)
            if key not in cache:
                cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoizer
