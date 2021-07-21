from threading import Lock
import os
import json
import numpy as np


def load_samples(samples_file):
    return load_json(samples_file)


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, target_file):
    with open(target_file, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)


def normalize_labels(labels):
    labels_np = np.array(labels)
    return labels_np / labels_np.sum()


def calc_mean_score(score_dist):
    # Expectation
    score_dist = normalize_labels(score_dist)
    return (score_dist * np.arange(1, 11)).sum()


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def thread_safe_memoize(func):
    cache = {}
    session_lock = Lock()

    def memoizer(*args, **kwargs):
        with session_lock:
            key = str(args) + str(kwargs)
            if key not in cache:
                cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoizer


def load_config(config_file):
    config = load_json(config_file)
    return config
