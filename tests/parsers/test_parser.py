import pytest
from icevision.all import *


@pytest.fixture()
def data():
    return [
        {"id": 1, "filepath": __file__, "labels": [1], "bboxes": [[1, 2, 3, 4]]},
        {
            "id": 42,
            "filepath": __file__,
            "labels": [2, 1],
            "bboxes": [[1, 2, 3, 4], [10, 20, 30, 40]],
        },
        {"id": 3, "filepath": "none.txt", "labels": [1], "bboxes": [[1, 2, 3, 4]]},
    ]


class SimpleParser(
    parsers.Parser, parsers.FilepathMixin, parsers.LabelsMixin, parsers.BBoxesMixin
):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        yield from self.data

    def imageid(self, o) -> Hashable:
        return o["id"]

    def filepath(self, o) -> Union[str, Path]:
        return o["filepath"]

    def image_width_height(self, o) -> Tuple[int, int]:
        return (100, 100)

    def labels(self, o) -> List[int]:
        return o["labels"]

    def bboxes(self, o) -> List[BBox]:
        return [BBox.from_xyxy(*pnts) for pnts in o["bboxes"]]


def test_parser(data):
    parser = SimpleParser(data)

    records = parser.parse(data_splitter=SingleSplitSplitter())[0]
    assert len(records) == 2

    record = records[1]
    assert set(record.keys()) == {
        "imageid",
        "filepath",
        "height",
        "width",
        "labels",
        "bboxes",
    }
    assert record["imageid"] == 1
    assert record["filepath"] == Path(__file__)
    assert record["labels"] == [2, 1]
    assert record["bboxes"] == [
        BBox.from_xyxy(1, 2, 3, 4),
        BBox.from_xyxy(10, 20, 30, 40),
    ]


@pytest.mark.skip
def test_parser_annotation_len_mismatch(data):
    class BrokenParser(SimpleParser):
        def labels(self, o) -> List[int]:
            return o["labels"][:1]

    parser = BrokenParser(data)

    with pytest.raises(RuntimeError) as err:
        records = parser.parse(data_splitter=SingleSplitSplitter())[0]
    assert "inconsistent number of annotations" in str(err)


@pytest.mark.skip
@pytest.mark.parametrize(
    "bbox_xyxy", [[1, 2, 1, 4], [3, 2, 1, 4], [1, 2, 3, 2], [1, 4, 3, 2]]
)
def test_parser_invalid_data(data, bbox_xyxy):
    invalid_sample = [{"id": 3, "labels": [1], "bboxes": [bbox_xyxy]}]

    parser = SimpleParser(data + invalid_sample)

    with pytest.raises(InvalidDataError) as err:
        records = parser.parse(data_splitter=SingleSplitSplitter())[0]
