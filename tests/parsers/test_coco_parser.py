import pytest
from icevision.all import *


@pytest.fixture
def coco_dir():
    return Path(__file__).absolute().parent.parent.parent / "samples"


@pytest.fixture
def coco_bbox_parser(coco_dir):
    return parsers.coco(coco_dir / "annotations.json", coco_dir / "images", mask=False)


@pytest.fixture
def coco_mask_parser(coco_dir):
    return parsers.coco(coco_dir / "annotations.json", coco_dir / "images", mask=True)


def test_bbox_parser(coco_dir, coco_bbox_parser):
    records = coco_bbox_parser.parse(data_splitter=SingleSplitSplitter())[0]
    assert len(records) == 5

    record = records[0]
    assert record.imageid == 0
    assert record.filepath == coco_dir / "images/000000343934.jpg"
    assert record.width == 640
    assert record.height == 480

    assert record.labels == [4]
    assert pytest.approx(record.bboxes[0].xyxy) == (175.14, 175.68, 496.2199, 415.68)
    assert record.iscrowds == [0]
    assert pytest.approx(record.areas) == [43522.805]


def test_mask_parser(coco_mask_parser):
    records = coco_mask_parser.parse(data_splitter=SingleSplitSplitter())[0]
    assert len(records) == 5

    record = records[0]
    assert len(record.masks) == 1
    assert record.masks[0].points[0][:2] == [457.3, 258.92]
