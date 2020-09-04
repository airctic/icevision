import pytest
from icevision.all import *


def test_coco_parser(samples_source):
    parser = parsers.coco(
        annotations_file=samples_source / "annotations.json",
        img_dir=samples_source / "images",
    )

    records = parser.parse()[0]
    assert len(records) == 5

    record = records[0]
    assert record["imageid"] == 0
    assert record["filepath"] == samples_source / "images/000000128372.jpg"
    assert record["height"] == 427
    assert record["width"] == 640

    assert record["labels"] == [6, 1, 1, 1, 1, 1, 1, 1, 31, 31, 1, 3, 31, 1, 31, 31]
    assert pytest.approx(record["bboxes"][0].xywh) == [0, 73.89, 416.44, 305.13]

    assert "masks" in record
