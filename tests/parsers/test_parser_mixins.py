from icevision.all import *


def test_all_parser_mixins():
    class TestAllParserMixins(
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
            return "path"

        def image_height(self, o) -> int:
            return 420

        def image_width(self, o) -> int:
            return 480

        def labels(self, o) -> List[int]:
            return [0]

        def bboxes(self, o) -> List[BBox]:
            return [BBox.from_xyxy(1, 2, 3, 4)]

        def masks(self, o) -> List[Mask]:
            return [MaskArray(np.array([]))]

        def areas(self, o) -> List[float]:
            return [4.2]

        def iscrowds(self, o) -> List[bool]:
            return [False]

    mixins = TestAllParserMixins()

    Record = create_mixed_record(mixins.record_mixins())
    record = Record()

    mixins.parse_fields(None, record)

    assert record["imageid"] == 42
    assert record["filepath"] == Path("path")
    assert record["height"] == 420
    assert record["width"] == 480
    assert record["labels"] == [0]
    assert record["bboxes"] == [BBox.from_xyxy(1, 2, 3, 4)]
    assert all(record["masks"][0].data == MaskArray(np.array([])).data)
    assert record["areas"] == [4.2]
    assert record["iscrowds"] == [False]
