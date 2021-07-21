__version__ = "1.0.0"
__author__ = "Sandip Dey"

from abc import ABCMeta, abstractmethod


class init_training(metaclass=ABCMeta):

    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(init_training, cls).__new__(cls)
        return cls._instances[cls]

    @abstractmethod
    def __train(self):
        pass

# from .image_statistics import predict as image_statistic_est
# from .brisque import predict as brisque_prediction
# from .lbp import predict as lbp_prediction
# from .nima import predict as nima_prediction
# from .diqa import predict as diqa_prediction




# __all__ = ["image_statistic_est", "nima_prediction", "lbp_prediction", "brisque_prediction", "diqa_prediction"]
