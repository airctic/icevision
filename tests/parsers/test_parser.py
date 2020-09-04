import pytest
from icevision.all import *


@pytest.fixture()
def data():
    return [
        {"id": 1, "labels": [1], "bboxes": [[1, 2, 3, 4]]},
        {"id": 42, "labels": [2, 1], "bboxes": [[1, 2, 3, 4], [4, 3, 2, 1]]},
    ]


class SimpleParser(parsers.Parser, parsers.LabelsMixin, parsers.BBoxesMixin):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        yield from self.data

    def imageid(self, o) -> Hashable:
        return o["id"]

    def labels(self, o) -> List[int]:
        return o["labels"]

    def bboxes(self, o) -> List[BBox]:
        return [BBox.from_xyxy(*pnts) for pnts in o["bboxes"]]


def test_parser(data):
    parser = SimpleParser(data)

    records = parser.parse()[0]
    assert len(records) == 2

    record = records[1]
    assert set(record.keys()) == {"imageid", "labels", "bboxes"}
    assert record["imageid"] == 1
    assert record["labels"] == [2, 1]
    assert record["bboxes"] == [BBox.from_xyxy(1, 2, 3, 4), BBox.from_xyxy(4, 3, 2, 1)]


def test_parser_annotation_len_mismatch(data):
    class BrokenParser(SimpleParser):
        def labels(self, o) -> List[int]:
            return o["labels"][:1]

    parser = BrokenParser(data)

    with pytest.raises(RuntimeError) as err:
        records = parser.parse(data)
    assert "inconsistent number of annotations" in str(err)
