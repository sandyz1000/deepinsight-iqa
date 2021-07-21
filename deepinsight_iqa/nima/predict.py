import os
import glob
import sys
from .utils import calc_mean_score, save_json, thread_safe_memoize
from .handlers.model_builder import Nima
from ..data_pipeline.nima_gen.nima_datagen import NimaDataGenerator as TestDataGenerator
import tensorflow as tf


@thread_safe_memoize
def init_model():
    try:
        print("##### initiating NIMA model ")
        #MODEL_PATH = '/home/ssg0283/Documents/PIAXR_BATNCHES/scrachPad/pixar-scratchpad/iqa/weights/nima'
        MODEL_PATH = '/data1/weights/iqa/nima'
        base_model_name = "MobileNet"
        weights_file = os.path.join(MODEL_PATH, base_model_name, "weights_mobilenet_technical_0.11.hdf5")

        # build model and load weights
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=800)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print("Invalid device or cannot modify virtual devices once initialized")

        nima = Nima(base_model_name, weights=None)
        nima.build()
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


def predict(nima, image, predictions_file=None, img_format='jpg'):
    # load samples
    # if os.path.isfile(image_source):
    #     image_dir, samples = image_file_to_json(image_source)
    # else:
    #     image_dir = image_source
    #     samples = image_dir_to_json(image_dir, img_type='jpg')

    # initialize data generator
    n_classes = 10
    batch_size = 64
    samples = []
    sample = {"imgage_id": "img_1"}
    samples.append(sample)
    data_generator = TestDataGenerator(samples, image, batch_size, n_classes,
                                       nima.preprocessing_function(), img_format=img_format)

    # get predictions
    predictions = nima.nima_model.predict_generator(data_generator, workers=1, use_multiprocessing=False, verbose=1)

    # calc mean scores and add to samples
    for i, sample in enumerate(samples):
        sample['mean_score_prediction'] = calc_mean_score(predictions[i])

    # print(json.dumps(samples, indent=2))

    if predictions_file is not None:
        save_json(samples, predictions_file)

    return samples


def _main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-is', '--image-source', help='image directory or file', required=True)
    parser.add_argument('-pf', '--predictions-file', help='file with predictions', required=False, default=None)
    args = vars(parser.parse_args())
    args['nima'] = init_model()
    predict(**args)


# if __name__ == '__main__':
#     _main()
