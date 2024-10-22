from .base import Exporter
from .coco_exporter import COCOExporter
from .yolov8_exporter import YOLOv8Exporter
from .pascal_voc_exporter import PascalVOCExporter
from .excel_exporter import ExcelExporter
from .tfrecord_exporter import TFRecordExporter
from .dataset_coco_exporter import DatasetCocoExporter

__all__ = [
    'Exporter',
    'COCOExporter',
    'YOLOv8Exporter',
    'PascalVOCExporter',
    'ExcelExporter',
    'TFRecordExporter',
    'DatasetCocoExporter',
    'get_exporter'
]

def get_exporter(format_):
    if format_ == "coco":
        return COCOExporter()
    elif format_ == "dataset_coco":
        return DatasetCocoExporter()
    elif format_ == "yolov8":
        return YOLOv8Exporter()
    elif format_ == "pascal_voc":
        return PascalVOCExporter()
    elif format_ == "excel":
        return ExcelExporter()
    elif format_ == "tfrecord":
        return TFRecordExporter()
    else:
        raise ValueError(f"Unknown format: {format_}")