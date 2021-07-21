import csv
import os


def read_csv(csv_path, delimeter=","):
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimeter)
        for row in reader:
            yield row


def write_csv(rows, csv_path, delimeter=","):
    """ 
    Write results in a csv_path
    """
    colnames = rows.keys()
    with open(csv_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=colnames, delimeter=delimeter)
        writer.writeheader()
        for row in range(len(rows)):
            writer.writerow(row)


def train_voting_regressor(csvpath):
    """
    Train voting regressor for finding optimal parameters for calculating mean score for ensemble models
    :param img_dir: [description]
    :param csvpath: [description]
    """
    # from sklearn.ensemble import VotingRegressor
    # er = VotingRegressor([('nima', em), ('diqa', diqa), ('lbp', lbp), ('brisque', bris)])
    # er.fit(X, y).predict(X)
    # X = [('bris', x1), ('lbp', x2), ('nima', x3), ('diqa', x4), ('blur', x5)]
    # y = (0, 1)
    # from sklearn.ensemble import VotingClassifier
    # from sklearn.linear_model import LogisticRegression
    # clf = Logistic()
    # clf.fit(X, y)
    # clf.predict(X)
    # y = (0, 1)
    # clf.get_params()
    # w1, w2, w3, w4, w5, w0 -> bias

    pass


def batch_prediction(img_dir, csvpath, outpath):
    # TODO: Use iterator in batch for prediction
    # ThreadPoolExecutor
    rows = []
    for row in read_csv(csvpath):
        dist_path = os.path.join(img_dir, row['distorted_path'])
        ref_path = os.path.join(img_dir, row['reference_path'])
        # Fetch common image statistic based on image intensity
        gen_score = image_statistics.predict(dist_path)

        bris_score = brisque_prediction.predict(dist_path)
        bris_score = float(100 - bris_score) / 10.0

        # LBP Score
        lbp_score = lbp_prediction.predict(dist_path)[0]

        # NIMA mean score
        nima_score = nima_prediction.predict(nima_model, dist_path)[0]['mean_score_prediction']

        # DEEP IMAGE ASSESEMENT SCORE
        diqa_score = diqa_prediction.predict(diqa_model, dist_path)[0]
        rows.append({"dist_path": dist_path,
                     "ref_path": ref_path,
                     "brisque_score": bris_score,
                     "lbp_score": lbp_score,
                     "nima_score": nima_score,
                     "diqa_score": diqa_score,
                     "blur_factor": gen_score['bluriness']['blur'],
                     "mos": row['reference_path']
                     })

    # Save all score to columns in csv
    write_csv(rows, outpath)

    