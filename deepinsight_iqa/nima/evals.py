import os
import glob
import sys
from .utils import calc_mean_score, save_json
from .handlers.model_builder import Nima
from ..data_pipeline.nima_gen.nima_datagen import NimaDataGenerator as TestDataGenerator


def init_model(base_model_name="MobileNet", weights_file=None):
    try:
        nima = Nima(base_model_name, weights=None)
        nima.build()
        if weights_file:
            nima.nima_model.load_weights(weights_file)
        return nima
    except Exception as e:
        print("Unable to load NIMA weights", str(e))
        sys.exit(1)


def image_file_to_json(img_path):
    img_dir = os.path.dirname(img_path)
    img_id = os.path.basename(img_path).split('.')[0]

    return img_dir, [{'image_id': img_id}]


def image_dir_to_json(img_dir, img_type='jpg'):
    img_paths = glob.glob(os.path.join(img_dir, '*.' + img_type))

    samples = []
    for img_path in img_paths:
        img_id = os.path.basename(img_path).split('.')[0]
        samples.append({'image_id': img_id})

    return samples


def evaluation(nima,
               image_source,
               predictions_file=None,
               n_classes=10,
               batch_size=64,
               img_format='jpg',
               multiprocessing_data_load=False):
    # TODO: Save output and Calculate metric and save to csv

    if os.path.isfile(image_source):
        image_dir, samples = image_file_to_json(image_source)
    else:
        image_dir = image_source
        samples = image_dir_to_json(image_dir, img_type='jpg')

    # initialize data generator
    data_generator = TestDataGenerator(samples, image_dir, batch_size, n_classes, nima.preprocessing_function(),
                                       img_format=img_format, do_train=False, shuffle=False)

    # get predictions
    predictions = nima.nima_model.predict_generator(
        data_generator, workers=8, use_multiprocessing=multiprocessing_data_load, verbose=1)

    # calc mean scores and add to samples
    for i, sample in enumerate(samples):
        sample['mean_score_prediction'] = calc_mean_score(predictions[i])

    if predictions_file is not None:
        save_json(samples, predictions_file)

    return samples
