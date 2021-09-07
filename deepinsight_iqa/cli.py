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
            model_dir=cfg['model_dir'], final_wts_filename=cfg['final_wts_filename'],
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


def _train_diqa(cfg, image_dir, input_file, pretrained_model=None, train_model='all'):
    from deepinsight_iqa.diqa.train import Trainer
    from deepinsight_iqa.diqa.data import get_combine_datagen, get_iqa_datagen
    from deepinsight_iqa.diqa.utils.tf_imgutils import image_preprocess

    dataset_type = cfg.pop('dataset_type', None)
    # NOTE: Based on dataset_type init the corresponding datagenerator
    input_file = input_file if os.path.exists(input_file) else os.path.join(image_dir, input_file)
    if dataset_type:
        train_tfds, valid_tfds = get_iqa_datagen(
            image_dir, input_file, dataset_type,
            do_augment=cfg['use_augmentation'],
            image_preprocess=image_preprocess, input_size=cfg.pop('input_size'), **cfg
        )
    else:
        train_tfds, valid_tfds = get_combine_datagen(
            image_dir, input_file, do_augment=cfg['use_augmentation'],
            image_preprocess=image_preprocess, input_size=cfg.pop('input_size'), **cfg
        )

    trainer = Trainer(train_tfds, valid_iter=valid_tfds, **cfg)
    if pretrained_model:
        trainer.loadweights(pretrained_model)

    obj_trainer = partial(trainer.train_objective, model=trainer.diqa.objective)
    sub_trainer = partial(trainer.train_subjective, model=trainer.diqa.subjective)
    if train_model == "all":
        (func() for func in [obj_trainer, sub_trainer])
    else:
        sub_trainer() if train_model == "subjective" else obj_trainer()

    return 0


def _train_nima(cfg, image_dir, base_dir, input_file):
    from deepinsight_iqa.nima.train import Train
    samples_file = os.path.join(base_dir, input_file)
    samples = json.load(open(samples_file, 'r'))
    trainer = Train(samples=samples, job_dir=base_dir, image_dir=image_dir, **cfg)
    trainer.train()
    return 0


@click.command()
@click.option('-m', '--algo', required=True, show_choices=["nima", "diqa"], help="Pass algorithm to train")
@click.option('-t', '--train_model', default='all', show_choices=["all"] + TRAINING_MODELS[:],
              help="Arguments to mention if network need to be train completely or partially")
@click.option('-c', '--conf_file', help='train job directory with samples and config file', required=True)
@click.option('-b', '--base_dir', help='Directory where logs and weight can be stored/found',
              default=os.getcwd())
@click.option('-f', '--input-file', required=True, help='input csv/json file')
@click.option('-i', '--image-dir', help='directory with image files', required=True)
@click.option('-p', '--pretrained_model_name', show_choices=TRAINING_MODELS,
              type=str, help='Set pretrained to start training using pretrained n/w',
              default=None)
def train(algo, train_model, conf_file, base_dir, input_file, image_dir, pretrained_model_name=None):
    cfg = parse_config(base_dir, conf_file)
    if algo == "nima":
        _train_nima(cfg, image_dir, base_dir, input_file)

    elif algo == "diqa":
        _train_diqa(
            cfg, image_dir, input_file,
            pretrained_model=pretrained_model_name, train_model=train_model
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
        from deepinsight_iqa.diqa.data import get_combine_datagen, get_iqa_datagen

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
