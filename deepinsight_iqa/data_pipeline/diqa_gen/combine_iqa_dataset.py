import os
import pandas as pd
from typing import Dict, Tuple


def combine_deepiqa_dataset(data_dir: str, csvspathmap: Dict[str, str], output_csv: str) -> None:
    """ Combine all csv to single csv file that can be used by the data-generator
    """
    from functools import partial

    def _tid2013_data_parser(img_dir, *row):
        """
        Higher value of MOS (0 - minimal, 9 - maximal) corresponds to higher visual
        quality of the image.

        Rescale range to the range [0, 1], where 0 denotes the lowest quality (largest perceived distortion).
        """
        distorted_image, reference_image, mos = row
        return (
            os.path.join(img_dir, distorted_image),
            os.path.join(img_dir, reference_image),
            mos / 10
        )

    def _csiq_data_parser(img_dir, *row):
        """
        All of the distorted versions of each reference image were viewed simultaneously
        across the monitor array. Each subject horizontally positioned these images across
        the monitor array such that the horizontal distance between every pair of images
        reflected the difference in perceived quality between them.
        As a final step, across-image ratings were performed to obtain a "realignment" of
        the within-image ratings; this realignment experiment was a separate, but identical,
        experiment in which observers placed subsets of all the images linearly in space.

        The ratings were converted to z-scores, realigned, outliers removed, averaged across subjects,
        and then normalized to span the range [0, 1], where 1 denotes the lowest quality (largest perceived distortion).

        Subtract dmos from 1 to convert the score to common scale i.e. 0 denotes the lowest quality and vice-versa
        """
        image, dst_idx, dst_type, dst_lev, dmos_std, dmos = row
        dst_type = "".join(dst_type.split())
        dst_img_path = os.path.join(
            img_dir, 'dst_imgs', dst_type, f"{image}.{dst_type}.{dst_lev}.png"
        )
        ref_img_path = os.path.join(img_dir, 'src_imgs', f"{image}.png")

        return dst_img_path, ref_img_path, 1 - dmos

    def _liveiqa_data_parser(img_dir, *row: Tuple):
        """
        Difference Mean Opinion Score (DMOS) value for each distorted image:
        The raw scores for each subject is the difference scores (between the test and the reference) 
        and then Z-scores and then scaled and shifted to the full range (1 to 100).

        Rescale range to the range [0, 1] and subtract from 1 to covert it to common scale,
        where 0 denotes the lowest quality (largest perceived distortion) and vice-versa
        """
        distortion, index, distorted_path, reference_path, dmos, dmos_realigned, dmos_realigned_std = row
        return (
            os.path.join(img_dir, distorted_path),
            os.path.join(img_dir, reference_path),
            1 - (dmos / 100)
        )

    _FUNC_MAPPING = {
        "tid2013": _tid2013_data_parser,
        "csiq": _csiq_data_parser,
        "live": _liveiqa_data_parser
    }

    assert set(csvspathmap.keys()) == set(_FUNC_MAPPING.keys()), "Invalid csvpath to function map"
    cols = ['index', 'distorted_image', 'reference_image', 'mos']
    dataset = pd.DataFrame(columns=cols)
    for dataset_name, csvpath in csvspathmap.items():
        # csv_name = os.path.basename(csvpath)
        folder_name = os.path.dirname(csvpath)
        ddf = pd.read_csv(os.path.join(data_dir, csvpath))
        funcparser = partial(_FUNC_MAPPING[dataset_name], folder_name)
        current = pd.DataFrame([funcparser(*row) for idx, row in ddf.iterrows()], columns=cols)
        dataset = dataset.append(current, ignore_index=True)
    output_csv = os.path.join(data_dir, output_csv)
    dataset.to_csv(output_csv)
