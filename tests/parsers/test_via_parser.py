import pytest
from icevision.all import *


@pytest.fixture
def via_bbox_parser(via_dir, via_bbox_class_map):
    return parsers.via(via_dir / "via_bbox.json", via_dir, via_bbox_class_map)


def test_bbox_parser(via_dir, via_bbox_parser):
    records = via_bbox_parser.parse(data_splitter=SingleSplitSplitter())[0]
    assert len(records) == 2

    record = records[0]
    assert record.imageid == 0
    assert record.filepath == via_dir / "IMG_4908.jpg"
    assert record.width == 640
    assert record.height == 480

    assert record.labels == [4, 1, 3, 2]
    # Polygon
    assert record.bboxes[0].xyxy == (104, 120, 155, 176)
    # Rect
    assert record.bboxes[3].xyxy == (263, 55, 308, 103)
