import os
import json
import click
import sys
from deepinsight_iqa.data_pipeline import (TFDatasetType, TFRecordDataset)


@click.command()
@click.option('-t', '--dataset_type', required=True,
              show_choices=TFDatasetType._list_field(), help="Dataset Type, must be from the given options")
@click.option('-i', '--image-dir', help='directory with image files', required=True)
@click.option('-f', '--input-file', required=True, help='input csv/json file')
def prepare_tf_record(dataset_type, image_dir, input_file):
    kls = getattr(TFDatasetType, dataset_type, None)
    assert kls is not None, "Invalid attribute for the datatype"
    tfrecord: TFRecordDataset = kls()
    tfrecord_path = tfrecord.write_tfrecord_dataset(image_dir, input_file)
    return tfrecord_path


@click.command()
@click.option('-m', '--algo', required=True, show_choices=["nima", "diqa"], help="Pass algorithm to train")
@click.option('-c', '--conf_file', help='train job directory with samples and config file', required=True)
@click.option('-b', '--base_dir', help='Directory where logs and weight can be stored/found', default=os.getcwd())
@click.option('-f', '--input-file', required=True, help='input csv/json file')
@click.option('-i', '--image-dir', help='directory with image files', required=True)
def predict(algo, conf_file, base_dir, input_file, image_dir):
    # Script use to predict Image quality using Deep image quality assesement
    if algo == "nima":
        cfg = parse_config(base_dir, conf_file)
        from deepinsight_iqa.nima import predict

        return 0
    elif algo == "diqa":
        from deepinsight_iqa.diqa import diqa

        return 0


def parse_config(job_dir, config_file):
    os.makedirs(os.path.join(job_dir, 'weights'), exist_ok=True)
    os.makedirs(os.paisth.join(job_dir, 'logs'), exist_ok=True)
    config = json.load(open(config_file, 'r'))
    return config


@click.command()
@click.option('-m', '--algo', required=True, show_choices=["nima", "diqa"], help="Pass algorithm to train")
@click.option('-c', '--conf_file', help='train job directory with samples and config file', required=True)
@click.option('-b', '--base_dir', help='Directory where logs and weight can be stored/found', default=os.getcwd())
@click.option('-f', '--input-file', required=True, help='input csv/json file')
@click.option('-i', '--image-dir', help='directory with image files', required=True)
def train(algo, conf_file, base_dir, input_file, image_dir):
    cfg = parse_config(base_dir, conf_file)
    if algo == "nima":
        from deepinsight_iqa.nima import train
        samples_file = os.path.join(base_dir, input_file)
        samples = json.load(open(samples_file, 'r'))
        trainer = train.Train(samples=samples, job_dir=base_dir, image_dir=image_dir, **cfg)
        trainer.train()
        return 0

    elif algo == "diqa":
        from deepinsight_iqa.diqa import train
        dataset_type = cfg.pop('dataset_type')
        trainer = train.Train(image_dir, input_file, dataset_type, **cfg)
        trainer.train()
        return 0


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

        return 0


@click.group()
def main():
    return 0


main.add_command(train, "train")
main.add_command(evaluate, "evaluate")


if __name__ == "__main__":
    sys.exit(main())