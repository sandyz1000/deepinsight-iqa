from .utility import thread_safe_memoize
from . import tf_features as feat
import cv2


@thread_safe_memoize
def predict(img_path, blur=True, dominant_color=True, aesthetic=True, pixel_width=True, **kwargs):
    img_stats = dict()
    img = feat.read_img(img_path)

    if kwargs.get('aesthetic', False):
        dullness_score, spots = feat.color_analysis(img)
        img_stats['spots'] = spots
        img_stats['dullness_score'] = dullness_score

    if kwargs.get('blur', False):
        blur_score = feat.bluriness_score(feat.rgbtogray(img))
        img_stats['blur_score'] = blur_score

    if kwargs.get('pixel_width', False):
        avg_pixel_width = feat.average_pixel_width(feat.rgbtogray(img))
        img_stats['avg_pixel_width'] = avg_pixel_width

    if kwargs.get('dominant_color', False):
        dominant_color = feat.get_dominant_color(cv2.imread(img_path))
        img_stats['dominant_color'] = dominant_color

    return img_stats


def _main():
    import argparse
    parser = argparse.ArgumentParser("Get bluriness score")
    parser.add_argument("--img-path", required=True, type=str, help="Image file path")
    parser.add_argument("--color_analysis", default=True, type=bool, help="Perform color analysis")
    parser.add_argument("--blur_score", default=True, type=bool, help="Get Bluriness score")
    parser.add_argument("--dominant_color", default=True, type=bool, help="Get dominant color")
    parser.add_argument("--avg-pixel", default=True, type=bool, help="Avg pixel in an image")
    args = vars(parser.parse_args())
    print(predict(**args))


# if __name__ == "__main__":
#     _main()
