import pytest
from icevision.all import *
from copy import deepcopy


@pytest.fixture()
def record():
    Record = type("Record", (BBoxesRecordMixin, LabelsRecordMixin, BaseRecord), {})
    record = Record()

    record.set_imageid(1)
    record.set_image_size(3, 3)
    record.add_labels([1, 2])
    record.add_bboxes([BBox.from_xyxy(1, 2, 4, 4), BBox.from_xyxy(1, 2, 1, 3)])

    return record


def test_record_autofix(record):
    success_dict = record.autofix()

    assert record.labels == [1]
    assert record.bboxes == [BBox.from_xyxy(1, 2, 3, 3)]


def test_autofix_records(record):
    records = [record, deepcopy(record)]
    autofix_records(records)

    for record in records:
        assert record.labels == [1]
        assert record.bboxes == [BBox.from_xyxy(1, 2, 3, 3)]