import os
import json
import click
import importlib
_BASE_DIRNAME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def executor(dstype=("tid2013", "live"), use_pretrained=False, use_augmentation=False,
             batch_size=1, epochs=1):
    """ 
    Executor that will be used to initiate training of IQA dataset (TID2013, LIVE)

    TODO: Add image augmentation and train objective_error_map model with keras Data generator
    """
    dstype = DatasetType.TID if dstype == "tid2013" else (DatasetType.LIV if dstype == "live" else None)
    ds = DiqaDataset(dataset_type=dstype).fetch()

    model = init_train(ds, model_path=None, custom=False, epochs=epochs, batch=batch_size,
                       use_pretrained=use_pretrained, log_dir='logs')
    # TODO: Use keras or tf v2.2.0 API for training

    # objective_score = train_objective_model(
    #     ds, model_path=OBJECTIVE_MODEL_PATH, use_pretrained=use_pretrained
    # )

    # subjective_score, history = train_subjective_score(
    #     ds, objective_score, model_path=SUBJECTIVE_MODEL_PATH, use_pretrained=use_pretrained
    # )

    # TODO: Evaluate model (Write method for evaluation)


def parse_config(job_dir, config_file):
    ensure_dir_exists(os.path.join(job_dir, 'weights'))
    ensure_dir_exists(os.paisth.join(job_dir, 'logs'))
    config = load_json(config_file)
    return config


@click.command()
@click.option('-m', '--algo', required=True, show_choices=["lbp", "nima", "diqa"], help="Pass algorithm to train")
@click.option('-c', '--conf_file', help='train job directory with samples and config file', required=True)
@click.option('-f', '--input-file', required=True, help='input csv/json file')
@click.option('-i', '--image-dir', help='directory with image files', required=True)
def train(algo, conf_file, input_file, image_dir):
    assert algo in ["lbp", "nima", "diqa"], "Invalid algorithm args for training"
    job_dir = _BASE_DIRNAME
    cfg = parse_config(job_dir, conf_file)
    train_mod = importlib.import_module(f".{algo}.train")
    if algo == "nima":
        samples_file = os.path.join(job_dir, input_file)
        samples = load_json(samples_file)
        train_mod.train(samples=samples, job_dir=job_dir, image_dir=image_dir, **cfg)
        return 0

    elif algo == "lbp":
        return 0

    elif algo == "diqa":
        trainer = train_mod.TrainDeepIMAWithGenerator(image_dir, input_file, **cfg)
        trainer.keras_init_train()
        return 0


@click.command()
@click.option('-m', '--algo', required=True, show_choices=["lbp", "nima", "diqa"], help="Pass algorithm to train")
@click.option('-c', '--conf_file', help='train job directory with samples and config file', required=True)
@click.option('-f', '--input-file', required=True, help='input csv/json file')
@click.option('-i', '--image-dir', help='directory with image files', required=True)
def evaluate(algo, conf_file, input_file, image_dir):
    assert algo in ["lbp", "nima", "diqa"], "Invalid algorithm args for evals"
    job_dir = _BASE_DIRNAME
    cfg = parse_config(job_dir, conf_file)
    evals_mod = importlib.import_module(f".{algo}.evals")


@click.group()
def main():
    return 0


main.add_command(train, "train")
main.add_command(evaluate, "evaluate")