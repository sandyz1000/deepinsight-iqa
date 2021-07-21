import os

DATASETS_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKSUMS_PATH = os.path.join(DATASETS_PATH, 'url_checksums')
CHECKSUMS_PATH = os.path.normpath(CHECKSUMS_PATH)

from .live_iqa import LiveIQA
from .tid2013 import Tid2013
from .tfrecords import Tid2013RecordDataset, CSIQRecordDataset, LiveRecordDataset, AVARecordDataset
import enum

__all__ = ['Tid2013', 'LiveIQA', 'Tid2013RecordDataset', 'CSIQRecordDataset', 'LiveRecordDataset', 'AVARecordDataset']


class DatasetType(enum.Enum):
    LIV = "live"
    TID = "tid2013"
    CSIQ = "csiq"
    AVA = "ava"


class TFDataset:
    def __init__(self, dataset_type: str = None) -> None:
        _kls = self.__switcher(dataset_type)
        assert _kls is not None, "Invalid dataset type"
        self.builder = _kls()
        self.builder.download_and_prepare()

    def __switcher(self, dataset_type):
        switcher = {DatasetType.TID: Tid2013,
                    DatasetType.LIV: LiveIQA,
                    DatasetType.CSIQ: CSIQ,
                    DatasetType.AVA: AVA}
        return switcher.get(dataset_type, None)

    def fetch(self, shuffle=1024, batch=1):
        ds = self.builder.as_dataset(shuffle_files=True)['train']
        ds = ds.shuffle(shuffle).batch(batch)
        return ds
