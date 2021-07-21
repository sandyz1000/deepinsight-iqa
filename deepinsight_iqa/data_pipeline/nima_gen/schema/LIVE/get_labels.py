import os
import argparse
import numpy as np
import pandas as pd
import json
from maxentropy.skmaxent import MinDivergenceModel


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
    df = pd.read_csv(mean_raw_file, skiprows=1, header=None, sep=',')
    df.columns = ['distortion', 'index', 'distorted_path',
                  'reference_path', 'dmos', 'dmos_realigned', 'dmos_realigned_std']
    return df


def parse_raw_data(df):
    samples = []
    for i, row in df.iterrows():
        max_entropy_dist = get_max_entropy_distribution(row['dmos'])
        samples.append({'image_id': os.path.splitext(row['distorted_path'])[0],
                        'label': max_entropy_dist.tolist()})

    # split data into test and train set
    indices = np.arange(len(samples))
    np.random.shuffle()
    train_size = int(len(indices) * 0.7)
    train_samples = [samples[x] for x in indices[:train_size]]
    test_samples = [samples[x] for x in indices[train_size:]]
    return train_samples, test_samples


def _parse_arguments():
    parser = argparse.ArgumentParser()
    # SRC FILENAME: dmos.csv
    parser.add_argument(
        '-sf', '--source_csv_file', type=str, required=True, help='file path of raw mos_with_names file')
    parser.add_argument(
        '-tf', '--target_prefix', type=str, default="live_labels", help='file path of json labels file to be saved')

    return parser.parse_args()


def _cli():
    args = _parse_arguments()
    df = get_dataframe(args.source_csv_file)
    train_samples, test_samples = parse_raw_data(df)
    save_json(train_samples, "train.json", prefix=args.target_prefix)
    save_json(test_samples, "test.json", prefix=args.target_prefix)
    print('Done! Saved JSON with prefix {}'.format(args.target_prefix))


if __name__ == '__main__':
    _cli()
