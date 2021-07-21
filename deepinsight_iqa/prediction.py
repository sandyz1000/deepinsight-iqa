import os
import sys
from concurrent.futures import ThreadPoolExecutor
from . import image_statistics, nima, diqa, lbp, brisque
import numpy as np
import csv
import argparse
from pprint import pprint

nima_model = nima.predict.init_model()
diqa_model = diqa.predict.init_model()


def prediction(img_path):
    # Fetch common image statistic based on image intensity
    gen_score = image_statistics.predict(img_path)

    # BRISQUE Score
    bris_score = brisque.predict.predict(img_path)
    bris_score = float(100 - bris_score) / 10.0

    # LBP Score
    lbp_score = lbp.predict(img_path)[0]

    # NIMA mean score
    nima_score = nima.predict.predict(nima_model, img_path)[0]['mean_score_prediction']

    # DEEP IMAGE ASSESEMENT SCORE
    diqa_score = diqa.predict.predict(diqa_model, img_path)

    # NOTE: We are using cumulative mean score but it should be later implemented using voting regressor
    weights = np.array([0.3, 0.3, 0.3, 0.1])
    scores = np.array([diqa_score, nima_score, lbp_score, bris_score])
    cumulative = scores.dot(weights.T)
    
    return {"scores": {
            "brisque": bris_score,
            "lbp_score": lbp_score,
            "nima_score": nima_score,
            "diqa_score": diqa_score
            },
            "cum_score": cumulative,
            "image_statistics": gen_score}


def parse_arguments():
    parser = argparse.ArgumentParser("Use cumulative score for image quality")
    parser.add_argument("--im", required=True, help="Pass image path for evaluation")
    args = vars(parser.parse_args())
    pprint(prediction(args['im']))
