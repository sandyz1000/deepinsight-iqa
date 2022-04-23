import os
import json
import click
import sys
from typing import Dict, List, Any
from pathlib import Path
from deepinsight_iqa.nima.predict import Prediction as NimaPrediction
from deepinsight_iqa.diqa.predict import Prediction as DiqaPrediction
from deepinsight_iqa.diqa.evals import Evaluation as DiqaEvaluation
from deepinsight_iqa.nima.evals import Evaluation as NimaEvaluation
from deepinsight_iqa.data_pipeline import (TFDatasetType, TFRecordDataset)
from deepinsight_iqa.diqa.data import get_iqa_datagen
from deepinsight_iqa.diqa.utils.tf_imgutils import image_preprocess
from deepinsight_iqa.diqa.trainer import Trainer as DiqaTrainer
from deepinsight_iqa.nima.train import Train as NimaTrainer
from deepinsight_iqa.diqa import TRAINING_MODELS, SUBJECTIVE_NW, OBJECTIVE_NW


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
@click.option('-b', '--base_dir', help='Directory where logs and weight can be stored/found', default="weigths/diqa")
@click.option('-w', '--weight_file', help='Pretrained weights file name', required=True)
@click.option('-i', '--image_filepath', help='directory with image files', required=True)
def predict(algo, conf_file, base_dir, weight_file, image_filepath):
    # Script use to predict Image quality using Deep image quality assesement
    cfg = parse_config(conf_file)
    if algo == "nima":
        prediction = NimaPrediction(
            weights_file=Path(base_dir, weight_file),
            base_model_name=cfg['bottleneck']
        )
        score = prediction.predict(image_filepath, predictions_file='output.json')

    elif algo == "diqa":

        cf_model_dir = cfg.pop('model_dir', 'weights/diqa')
        cf_network = cfg.pop('network', SUBJECTIVE_NW)

        network = network if network else cf_network
        model_dir = base_dir if base_dir else cf_model_dir

        prediction = DiqaPrediction(
            model_dir=model_dir,
            weight_filename=weight_file,
            model_type=cfg['model_type'],
            network=network
        )
        score = prediction.predict(image_filepath)
    
    print("Score: ", score)
    return 0


def parse_config(config_file):
    config = json.load(Path(config_file).open('r'))
    return config


@click.command()
@click.option('-m', '--algo', required=True, show_choices=["nima", "diqa"], help="Pass algorithm to train")
@click.option('-n', '--network', required=False, show_choices=TRAINING_MODELS[:],
              help="Arguments to mention if network need to be train completely or partially")
@click.option('-c', '--conf_file', help='train job directory with samples and config file', required=True)
@click.option('-b', '--base_dir', help='Directory where logs and weight can be stored/found', default="weigths/diqa")
@click.option('-w', '--weight_file', help='Pretrained weights file name', default=None)
@click.option('-f', '--input-file', required=True, help='input csv/json file')
@click.option('-i', '--image-dir', help='directory with image files', required=True)
def train(
    algo: str,
    network: str,
    conf_file: str,
    base_dir: str,
    weight_file: str,
    input_file: str,
    image_dir: str,
):
    from deepinsight_iqa.common.utility import set_gpu_limit
    # set_gpu_limit(10)
    cfg = parse_config(conf_file)  # type: Dict

    def train_diqa(
        image_dir,
        input_file,
        model_dir=None,
        network=None,
        weight_fname=None
    ):
        dataset_type = cfg.pop('dataset_type', None)
        model_dir = model_dir if model_dir \
            else cfg.pop('model_dir', 'weights/diqa')

        # NOTE: Based on dataset_type init the corresponding datagenerator
        input_file = input_file if os.path.exists(input_file) \
            else os.path.join(image_dir, input_file)

        train_generator, valid_generator = get_iqa_datagen(
            image_dir,
            input_file,
            dataset_type=dataset_type,
            image_preprocess=image_preprocess,
            do_train=True,
            input_size=cfg['input_size'],
            do_augment=cfg['use_augmentation'],
            channel_dim=cfg['channel_dim'],
            batch_size=cfg['batch_size']
        )

        trainer = DiqaTrainer(
            train_generator,
            valid_datagen=valid_generator,
            model_dir=model_dir,
            network=network,
            weight_file=weight_fname,
            **cfg
        )

        if network == OBJECTIVE_NW:
            trainer.train_objective()
            return 0

        trainer.train_final()
        trainer.save_weights()

    def train_nima(cfg, image_dir, base_dir, input_file):
        samples_file = os.path.join(base_dir, input_file)
        samples = json.load(open(samples_file, 'r'))
        trainer = NimaTrainer(samples=samples, job_dir=base_dir, image_dir=image_dir, **cfg)
        trainer.train()
        return 0

    if algo == "nima":
        train_nima(cfg, image_dir, base_dir, input_file)

    elif algo == "diqa":
        print(f"Setting pretrained model type to {base_dir}")

        cf_model_dir = cfg.pop('model_dir', 'weights/diqa')
        cf_network = cfg.pop('network', SUBJECTIVE_NW)

        network = network if network else cf_network
        model_dir = base_dir if base_dir else cf_model_dir

        train_diqa(
            image_dir,
            input_file,
            model_dir=model_dir,
            network=network,
            weight_fname=weight_file
        )


@click.command()
@click.option('-m', '--algo', required=True, show_choices=["nima", "diqa"], help="Pass algorithm to train")
@click.option('-c', '--conf_file', help='train job directory with samples and config file', required=True)
@click.option('-b', '--base_dir', help='Directory where logs and weight can be stored/found', default="weigths/diqa")
@click.option('-w', '--weight_file', help='Pretrained weights file name', required=True)
@click.option('-f', '--input_csv', help='input csv/json file', required=True)
@click.option('-i', '--image_dir', help='directory with image files', required=True)
def evaluate(algo, conf_file, base_dir, weight_file, input_csv, image_dir):
    cfg = parse_config(conf_file)
    if algo == "nima":

        evaluator = NimaEvaluation(
            weights_file=Path(base_dir, weight_file),
            base_model_name=cfg['bottleneck']
        )

        outputs = evaluator(image_dir)

    elif algo == "diqa":

        model_dir = cfg.pop('model_dir', 'weights/diqa')

        model_dir = base_dir if base_dir else model_dir

        evaluator = DiqaEvaluation(
            model_dir=model_dir,
            weight_filename=weight_file,
            model_type=cfg['model_type'],
            batch_size=cfg['batch_size'],
            kwargs=cfg
        )
        score = evaluator(image_dir, input_csv)

    print(">>> Evaluation Complete >>> ")


@click.group()
def main():
    return 0


main.add_command(prepare_tf_record, "prepare_tf_record")
main.add_command(predict, "predict")
main.add_command(train, "train")
main.add_command(evaluate, "evaluate")


if __name__ == "__main__":
    sys.exit(main())
