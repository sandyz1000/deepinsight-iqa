import os
import json
import click
import sys
from typing import Dict, List, Any
from deepinsight_iqa.data_pipeline import (TFDatasetType, TFRecordDataset)
from deepinsight_iqa.diqa.data import get_iqa_datagen
from deepinsight_iqa.diqa.utils.tf_imgutils import image_preprocess
from deepinsight_iqa.diqa.trainer import Trainer as DiqaTrainer
from deepinsight_iqa.nima.train import Train as NimaTrainer

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


def train_nima(cfg, image_dir, base_dir, input_file):
    samples_file = os.path.join(base_dir, input_file)
    samples = json.load(open(samples_file, 'r'))
    trainer = NimaTrainer(samples=samples, job_dir=base_dir, image_dir=image_dir, **cfg)
    trainer.train()
    return 0


def train_diqa(
    cfg, 
    image_dir, 
    input_file, 
    weight_path=None, 
    network='subjective'
):
    dataset_type = cfg.pop('dataset_type', None)
    model_dir = cfg.pop('model_dir', 'weights/diqa')
    # NOTE: Based on dataset_type init the corresponding datagenerator
    input_file = input_file if os.path.exists(input_file) else os.path.join(image_dir, input_file)
    if dataset_type:
        train_tfds, valid_tfds = get_iqa_datagen(
            image_dir,
            input_file,
            dataset_type=dataset_type,
            image_preprocess=image_preprocess,
            input_size=cfg['input_size'],
            do_augment=cfg['use_augmentation'],
            channel_dim=cfg['channel_dim'],
            batch_size=cfg['batch_size']
        )
    else:
        train_tfds, valid_tfds = get_iqa_datagen(
            image_dir,
            input_file,
            image_preprocess=image_preprocess,
            input_size=cfg['input_size'],
            do_augment=cfg['use_augmentation'],
            channel_dim=cfg['channel_dim'],
            batch_size=cfg['batch_size']
        )
    
    network = network if network else cfg.pop('network', 'subjective')
    trainer = DiqaTrainer(
        train_tfds,
        valid_datagen=valid_tfds,
        model_dir=model_dir,
        network=network,
        **cfg
    )
    
    if network == "objective":
        trainer.train_objective()
        return 0
        
    trainer.train_final()


@click.command()
@click.option('-m', '--algo', required=True, show_choices=["nima", "diqa"], help="Pass algorithm to train")
@click.option('-n', '--network', required=False, show_choices=TRAINING_MODELS[:],
              help="Arguments to mention if network need to be train completely or partially")
@click.option('-c', '--conf_file', help='train job directory with samples and config file', required=True)
@click.option('-b', '--base_dir', help='Directory where logs and weight can be stored/found',
              default=os.getcwd())
@click.option('-f', '--input-file', required=True, help='input csv/json file')
@click.option('-i', '--image-dir', help='directory with image files', required=True)
@click.option('-p', '--weight_path', type=str, help='Set pretrained to start training using pretrained n/w')
def train(
    algo: str,
    network: str,
    conf_file: str,
    base_dir: str,
    input_file: str,
    image_dir: str,
    weight_path: str = None
):
    cfg = parse_config(base_dir, conf_file)  # type: Dict
    if algo == "nima":
        train_nima(cfg, image_dir, base_dir, input_file)

    elif algo == "diqa":
        print(f"Setting pretrained model type to {weight_path}")
        train_diqa(
            cfg,
            image_dir,
            input_file,
            weight_path=weight_path,
            network=network
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
