import pytest
from icevision.all import *


@pytest.fixture
def dummy_parser_all_mixins():
    class AllParserMixins(
        parsers.Parser,
        parsers.ClassMapMixin,
        parsers.ImageidMixin,
        parsers.FilepathMixin,
        parsers.SizeMixin,
        parsers.LabelsMixin,
        parsers.BBoxesMixin,
        parsers.MasksMixin,
        parsers.KeyPointsMixin,
        parsers.AreasMixin,
        parsers.IsCrowdsMixin,
    ):
        def __iter__(self) -> Any:
            raise NotImplementedError

        def imageid(self, o) -> int:
            return 42

        def filepath(self, o) -> Union[str, Path]:
            return __file__

        def image_width_height(self, o) -> Tuple[int, int]:
            return (480, 420)

        def labels(self, o) -> List[Hashable]:
            return ["a"]

        def bboxes(self, o) -> List[BBox]:
            return [BBox.from_xyxy(1, 2, 3, 4)]

        def masks(self, o) -> List[Mask]:
            return [MaskArray(np.zeros((1, 420, 480)))]

        def keypoints(self, o) -> List[KeyPoints]:
            return [KeyPoints((1, 1, 1), None)]

        def areas(self, o) -> List[float]:
            return [4.2]

        def iscrowds(self, o) -> List[bool]:
            return [False]

    return AllParserMixins


def test_parser_components(dummy_parser_all_mixins):
    parser = dummy_parser_all_mixins()
    comp_groups = component_registry.get_components_groups(parser.components)
    assert set(comp_groups) == set(
        (
            "classmap",
            "imageid",
            "filepath",
            "size",
            "label",
            "bbox",
            "mask",
            "keypoint",
            "area",
            "iscrowd",
        )
    )


def test_all_parser_mixins(dummy_parser_all_mixins):
    parser = dummy_parser_all_mixins()

    Record = parser.record_class()
    record = Record()

    parser.parse_fields(None, record)

    assert record["imageid"] == 42
    assert record["filepath"] == Path(__file__)
    assert record["height"] == 420
    assert record["width"] == 480
    assert record["labels"] == [1]
    assert record["bboxes"] == [BBox.from_xyxy(1, 2, 3, 4)]
    assert record["masks"].erles == [{"size": [420, 480], "counts": b"PlT6"}]
    assert record["areas"] == [4.2]
    assert record["iscrowds"] == [False]
    assert record["keypoints"] == [KeyPoints((1, 1, 1), None)]


def test_all_parser_mixins_broken_filepath(dummy_parser_all_mixins):
    class BrokenFilepath(dummy_parser_all_mixins):
        def filepath(self, o) -> Union[str, Path]:
            return "path.none"

    parser = BrokenFilepath()

    Record = parser.record_class()
    record = Record()

    with pytest.raises(AbortParseRecord):
        parser.parse_fields(None, record)
