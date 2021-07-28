from .dataset.tfrecords import (
    Tid2013RecordDataset, AVARecordDataset,
    CSIQRecordDataset, LiveRecordDataset, TFRecordDataset
)
import enum


class TFDatasetType(enum.Enum):
    csiq = CSIQRecordDataset,
    tid2013 = Tid2013RecordDataset,
    ava = AVARecordDataset,
    live = LiveRecordDataset

    @classmethod
    def _list_field(cls):
        return [
            v.name for k, v in cls._value2member_map_.items()
        ]
