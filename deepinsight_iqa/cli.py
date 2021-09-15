import os
import json
import click
import sys
from functools import partial
from deepinsight_iqa.data_pipeline import (TFDatasetType, TFRecordDataset)
TRAINING_MODELS = ["objective", "subjective"]


@click.command()
@click.option('-t', '--dataset_type', required=True,
              show_choices=TFDatasetType._list_field(), help="Dataset Type, must be from the given options")
@click.option('-i', '--image-dir', help='directory with image files', required=True)
@click.option('-f', '--input-file', required=True, help='input csv/json file')
def prepare_tf_record(dataset_type, image_dir, input_file):
    kls = getattr(TFDatasetType, dataset_type, None)
    assert kls is not None, "Invalid attribute for the datatype"
    tfrecord = kls()
    tfrecord_path = tfrecord.write_tfrecord_dataset(image_dir, input_file)
    return tfrecord_path


@click.command()
@click.option('-m', '--algo', required=True, show_choices=["nima", "diqa"], help="Pass algorithm to train")
@click.option('-c', '--conf_file', help='train job directory with samples and config file', required=True)
@click.option('-b', '--base_dir', help='Directory where logs and weight can be stored/found', default=os.getcwd())
@click.option('-i', '--image-filepath', help='directory with image files', required=True)
def predict(algo, conf_file, base_dir, image_filepath):
    # Script use to predict Image quality using Deep image quality assesement
    cfg = parse_config(base_dir, conf_file)
    if algo == "nima":
        from deepinsight_iqa.nima import predict

        return 0
    elif algo == "diqa":
        from deepinsight_iqa.diqa.predict import Prediction
        prediction = Prediction(
            model_dir=cfg['model_dir'], subjective_weightfname=cfg['subjective_weightfname'],
            base_model_name=cfg['base_model_name']
        )
        score = prediction.predict(image_filepath)
        print("Score: ", score)
        return 0


def parse_config(job_dir, config_file):
    os.makedirs(os.path.join(job_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(job_dir, 'logs'), exist_ok=True)
    config = json.load(open(os.path.join(job_dir, config_file), 'r'))
    return config


@click.command()
@click.option('-m', '--algo', required=True, show_choices=["nima", "diqa"], help="Pass algorithm to train")
@click.option('-t', '--train_model', default='all', show_choices=["all"] + TRAINING_MODELS[:],
              help="Arguments to mention if network need to be train completely or partially")
@click.option('-c', '--conf_file', help='train job directory with samples and config file', required=True)
@click.option('-b', '--base_dir', help='Directory where logs and weight can be stored/found',
              default=os.getcwd())
@click.option('-f', '--input-file', required=True, help='input csv/json file')
@click.option('-i', '--image-dir', help='directory with image files', required=True)
@click.option('-p', '--pretrained_model', show_choices=TRAINING_MODELS,
              type=str, help='Set pretrained to start training using pretrained n/w',
              default=None)
def train(algo, train_model, conf_file, base_dir, input_file, image_dir, pretrained_model=None):
    cfg = parse_config(base_dir, conf_file)
    if algo == "nima":
        from deepinsight_iqa.nima.train import train_nima
        train_nima(cfg, image_dir, base_dir, input_file)

    elif algo == "diqa":
        from deepinsight_iqa.diqa.train import train_diqa
        train_diqa(
            cfg, image_dir, input_file,
            pretrained_model=pretrained_model, train_model=train_model
        )


@click.command()
@click.option('-m', '--algo', required=True, show_choices=["nima", "diqa"], help="Pass algorithm to train")
@click.option('-c', '--conf_file', help='train job directory with samples and config file', required=True)
@click.option('-b', '--base_dir', help='Directory where logs and weight can be stored/found', default=os.getcwd())
@click.option('-f', '--input-file', required=True, help='input csv/json file')
@click.option('-i', '--image-dir', help='directory with image files', required=True)
def evaluate(algo, conf_file, base_dir, input_file, image_dir):
    cfg = parse_config(base_dir, conf_file)
    if algo == "nima":
        from deepinsight_iqa.nima import evals

        return 0
    elif algo == "diqa":
        from deepinsight_iqa.diqa import evals
        from deepinsight_iqa.diqa.utils.tf_imgutils import image_preprocess
        from deepinsight_iqa.diqa.data import get_iqa_combined_datagen, get_iqa_tfds

        return 0


@click.group()
def main():
    return 0


main.add_command(prepare_tf_record, "prepare_tf_record")
main.add_command(predict, "predict")
main.add_command(train, "train")
main.add_command(evaluate, "evaluate")


if __name__ == "__main__":
    sys.exit(main())
