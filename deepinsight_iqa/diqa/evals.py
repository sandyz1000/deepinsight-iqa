import os
import numpy as np
from .utils import image_preprocess
from .model import Diqa
from scipy.stats import pearsonr, spearmanr
from ..data_pipeline.diqa_datagen import DiqaDataGenerator, DatasetType, DiqaDataset
import pandas as pd


def evaluation(image_dir, model_path, csv_path, 
               custom=False, batch_size=1, out_path="report.csv"):
    """
    Use pearson and spearman correlation matrix for evaluation, we can also use RMSE error to 
    calculate the mean differences
    """

    diqa = Diqa(custom=custom)
    assert csv_path.split(".")[-1:] == 'csv', "Not a valid file extension"

    df = pd.read_csv(csv_path)
    test_generator = DiqaDataGenerator(df,
                                       image_dir,
                                       batch_size,
                                       img_preprocessing=image_preprocess,
                                       shuffle=False)

    nb_samples = len(test_generator)
    diqa.subjective_score_model.load_weights(model_path)
    predictions = diqa.subjective_score_model.predict_generator(test_generator, steps=nb_samples)

    # TODO: Use prediction to create evaluation metrics
