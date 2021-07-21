import os
from easydict import EasyDict

__C = EasyDict()
cfg = __C


__C.BASE_DIR = os.path.abspath(os.path.dirname(__file__))
__C.OBJECTIVE_MODEL_NAME = "objective-model-mobilenetv2.tid2013.h5"
__C.SUBJECTIVE_MODEL_NAME = "subjective-model-mobilenetv2.tid2013.h5"

