import json
import argparse
import numpy as np
import pandas as pd
from maxentropy.skmaxent import MinDivergenceModel
import logging
import glob2
import os
logger = logging.getLogger(__name__)
CHOICES = ["mos_to_prob", "prob_to_mos"]


def normalize_labels(labels):
    labels_np = np.array(labels)
    return labels_np / labels_np.sum()


def calc_mean_score(score_dist):
    # Expectation
    score_dist = normalize_labels(score_dist)
    return (score_dist * np.arange(1, 11)).sum()


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, target_file, prefix=""):
    with open(f"{prefix}_{target_file}", 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)


# the maximised distribution must satisfy the mean for each sample
def get_features():
    def f0(x):
        return x

    return [f0]


def get_max_entropy_distribution(mean: pd.Series):
    SAMPLESPACE = np.arange(10)
    features = get_features()

    model = MinDivergenceModel(features, samplespace=SAMPLESPACE, algorithm='CG')

    # set the desired feature expectations and fit the model
    X = np.array([[mean]])
    model.fit(X)

    return model.probdist()


def get_dataframe(mean_raw_file):
    df = pd.read_csv(mean_raw_file, skiprows=0, header=None, sep=' ')
    df.columns = ['distorted_path', 'reference_path', 'mos']
    return df


def parse_raw_data(df):
    samples = []
    for i, row in df.iterrows():
        max_entropy_dist = get_max_entropy_distribution(row['mos'])
        samples.append({'image_id': row['distorted_path'].split('.')[0], 'label': max_entropy_dist.tolist()})

    # split data into test and train set
    indices = np.random.shuffle(np.arange(len(samples)))
    train_size = len(indices) * 0.7
    train_samples = [samples[x] for x in indices[:train_size]]
    test_samples = [samples[x] for x in indices[train_size:]]
    return train_samples, test_samples


def mos_to_prob(source_file, target_file):
    """ Calculate probability dist from MOS and save to json
    """
    df = get_dataframe(source_file)
    train_samples, test_samples = parse_raw_data(df)
    for sample, filename in [(train_samples, "train.json"), (test_samples, "test.json")]:
        save_json(sample, filename, prefix=target_file)
    logger.info(f'Done! Saved JSON at {target_file}')


def prob_to_mos(src_path, target_file):
    """ Merge/Convert json file to single CSV
    Arguments:
        args {[type]} -- [description]
    """
    ava_labels = []
    for filename in glob2.glob(os.path.join(src_path, "*.json")):
        ava_labels = +[{"image_id": row['image_id'], "label": calc_mean_score(row['label'])}
                       for row in load_json(filename).items()]
    df = pd.DataFrame.from_dict(ava_labels)
    df = df.iloc[np.random.permutation(len(df))]
    df.columns = ["image_id", "label"]
    df.to_csv(target_file)
    logger.info(f'Done! Saved CSV at {target_file} location')


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conversion_type", choices=CHOICES, required=True,
                        help="Convert mos to probability distribution and vice-versa.")
    parser.add_argument('-sf', '--source-path', required=True,
                        help='csv/json file path of raw mos_with_names file')
    parser.add_argument('-tf', '--target-path', required=True,
                        help='file path of json/csv labels file to be saved')
    return parser.parse_args()


def _cli():
    args = _parse_arguments()
    if args.conversion_type == "mos_to_prob":
        mos_to_prob(args.source_path, args.target_path)
    elif args.conversion_type == "prob_to_mos":
        prob_to_mos(args.source_path, args.target_path)
    else:
        raise ValueError("Invalid choices args")


if __name__ == '__main__':
    _cli()
