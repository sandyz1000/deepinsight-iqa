__version__ = "1.0.0"
__author__ = "Sandip Dey"

from deepinsight_iqa.nima import predict as nima_prediction
from deepinsight_iqa.diqa import predict as diqa_prediction

__all__ = [
    "nima_prediction",
    "diqa_prediction"
]
