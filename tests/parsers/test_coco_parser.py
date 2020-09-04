import pytest
from icevision.all import *

source = Path(__file__).absolute().parent.parent.parent / "samples"


def test_info_parser():
    parser = test_utils.sample_image_info_parser()
    infos = parser.parse()[0]
    assert len(infos) == 6
    expected = {
        "imageid": 0,
        "filepath": source / "images/000000128372.jpg",
        "height": 427,
        "width": 640,
    }
    assert infos[0] == expected


def test_coco_annotation_parser():
    parser = test_utils.sample_annotation_parser()
    annotations = parser.parse()[0]
    annotation = annotations[0]
    assert len(annotations) == 5
    assert annotation["imageid"] == 0
    assert annotation["labels"] == [4]
    assert pytest.approx(annotation["bboxes"][0].xyxy) == [
        175.14,
        175.68,
        496.2199,
        415.68,
    ]


def test_coco_parser(records):
    r = records[0]
    assert len(records) == 5
    assert (r["height"], r["width"]) == (427, 640)
    assert r["imageid"] == 0
    assert r["bboxes"][0].xywh == [0.0, 73.89, 416.44, 305.13]
    assert r["filepath"] == source / "images/000000128372.jpg"
