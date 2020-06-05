import pytest
from mantisshrimp.imports import Path, json, plt
from mantisshrimp import *

source = Path(__file__).absolute().parent.parent.parent / "samples"


def test_info_parser():
    parser = test_utils.sample_image_info_parser()
    infos = parser.parse()[0]
    assert len(infos) == 6
    expected = {
        "imageid": 128372,
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
    assert annotation["imageid"] == 343934
    assert annotation["label"] == [4]
    assert pytest.approx(annotation["bbox"][0].xyxy) == [
        175.14,
        175.68,
        496.2199,
        415.68,
    ]


@pytest.mark.skip
def test_category_parser():
    catparser = test_utils.sample_category_parser()
    with np_local_seed(42):
        catmap = catparser.parse_dicted(show_pbar=False)
    assert catmap.cats[0].name == "background"
    assert len(catmap) == 81
    assert catmap.cats[2] == Category(2, "bicycle")
    assert catmap.id2i[42] == 38


def test_coco_parser(records):
    r = records[2]
    assert len(records) == 5
    assert (r["height"], r["width"]) == (427, 640)
    assert r["imageid"] == 128372
    assert r["bbox"][0].xywh == [0.0, 73.89, 416.44, 305.13]
    assert r["filepath"] == source / "images/000000128372.jpg"
