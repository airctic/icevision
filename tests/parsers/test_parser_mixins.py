import pytest
from icevision.all import *


@pytest.fixture
def all_parser_mixins_cls():
    class AllParserMixins(
        parsers.ImageidMixin,
        parsers.FilepathMixin,
        parsers.SizeMixin,
        parsers.LabelsMixin,
        parsers.BBoxesMixin,
        parsers.MasksMixin,
        parsers.AreasMixin,
        parsers.IsCrowdsMixin,
    ):
        def imageid(self, o) -> int:
            return 42

        def filepath(self, o) -> Union[str, Path]:
            return __file__

        def image_width_height(self, o) -> Tuple[int, int]:
            return (480, 420)

        def labels(self, o) -> List[int]:
            return [0]

        def bboxes(self, o) -> List[BBox]:
            return [BBox.from_xyxy(1, 2, 3, 4)]

        def masks(self, o) -> List[Mask]:
            return [MaskArray(np.zeros((1, 420, 480)))]

        def areas(self, o) -> List[float]:
            return [4.2]

        def iscrowds(self, o) -> List[bool]:
            return [False]

    return AllParserMixins


def test_all_parser_mixins(all_parser_mixins_cls):
    mixins = all_parser_mixins_cls()

    Record = create_mixed_record(mixins.record_mixins())
    record = Record()

    mixins.parse_fields(None, record)

    assert record["imageid"] == 42
    assert record["filepath"] == Path(__file__)
    assert record["height"] == 420
    assert record["width"] == 480
    assert record["labels"] == [0]
    assert record["bboxes"] == [BBox.from_xyxy(1, 2, 3, 4)]
    assert record["masks"].erles == [{"size": [420, 480], "counts": b"PlT6"}]
    assert record["areas"] == [4.2]
    assert record["iscrowds"] == [False]


def test_all_parser_mixins_broken_filepath(all_parser_mixins_cls):
    class BrokenFilepath(all_parser_mixins_cls):
        def filepath(self, o) -> Union[str, Path]:
            return "path.none"

    mixins = BrokenFilepath()

    Record = create_mixed_record(mixins.record_mixins())
    record = Record()

    with pytest.raises(AbortParseRecord):
        mixins.parse_fields(None, record)
