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

