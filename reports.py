import os
import sys
from concurrent.futures import ThreadPoolExecutor
sys.path.append(os.path.abspath('./src'))
from src import image_statistic_est, nima_prediction, diqa_prediction, lbp_prediction, brisque_prediction
import numpy as np
import csv

nima_model = nima_prediction.init_model()
diqa_model = diqa_prediction.init_model()


def prediction(img_path, label):
    if label == "GOOD":
        label = 1
    elif label == 'BAD':
        label = 0
    else:
        return "label is not correct"
    file = img_path.split('/')[-1]
    # Fetch common image statistic based on image intensity
    gen_score = image_statistic_est.predict(img_path)

    # BRISQUE Score
    bris_score = brisque_prediction.predict(img_path)
    bris_score = float(100 - bris_score) / 10.0

    # LBP Score
    lbp_score = lbp_prediction.predict(img_path)[0]

    # NIMA mean score
    nima_score = nima_prediction.predict(nima_model, img_path)[0]['mean_score_prediction']

    # DEEP IMAGE ASSESEMENT SCORE
    diqa_score = diqa_prediction.predict(diqa_model, img_path)

    # NOTE: We are using cumulative mean score but it should be later implemented using voting regressor
    weights = np.array([0.3, 0.3, 0.3, 0.1])
    scores = np.array([diqa_score, nima_score, lbp_score, bris_score])
    cumulative = scores.dot(weights.T)

    return {"file": file,
            "brisque_score": bris_score,
            "lbp_score": lbp_score,
            "nima_score": nima_score,
            "diqa_score": diqa_score,
            "blur_factor": gen_score['blur_score'],
            "label": label
            }


def write_csv(rows, csv_path, delimeter=","):
    csv_path = os.path.join(csv_path, 'train.csv')
    """
    Write results in a csv_path
    """
    colnames = rows[0].keys()
    with open(csv_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=colnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def get_results(datasetpath):
    rows = []
    for root, dirs, files in os.walk(datasetpath):
        for _dir in dirs:
            dirpath = os.path.join(root, _dir)
            files = os.listdir(dirpath)
            for file in files:
                path = os.path.join(root, _dir, file)
                result = prediction(path, _dir)
                rows.append(result)
    return rows


def create_csv(datasetpath, csvpath):
    rows = get_results(datasetpath)
    write_csv(rows, csvpath)


if __name__ == "__main__":
    import argparse
    from pprint import pprint
    parser = argparse.ArgumentParser("Use cumulative score for image quality")
    parser.add_argument("--dataset", required=True, help="Pass image path for evaluation")
    parser.add_argument("--csvpath", required=True, help="Pass image path for evaluation")

    args = vars(parser.parse_args())
    pprint(create_csv(args['dataset'], args['csvpath']))
