from icevision.all import *


def test_all_parser_mixins():
    class TestAllParserMixins(
        parsers.ImageidMixin,
        parsers.FilepathMixin,
        parsers.SizeMixin,
        parsers.LabelsMixin,
        parsers.BBoxesMixin,
        parsers.MasksMixin,
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

        def labels(self, o):
            return 0

        def bboxes(self, o) -> BBox:
            return BBox.from_xyxy(1, 2, 3, 4)

        def masks(self, o) -> MaskArray:
            return MaskArray(np.array([]))

        def iscrowds(self, o) -> bool:
            return 0

    test = TestAllParserMixins()
    info_parse_funcs = {
        "imageid": test.imageid,
        "height": test.image_height,
        "width": test.image_width,
        "filepath": test.filepath,
    }
    annotation_parse_funcs = {
        "labels": test.labels,
        "bboxes": test.bboxes,
        "masks": test.masks,
        "iscrowds": test.iscrowds,
    }
    assert test.collect_info_parse_funcs() == info_parse_funcs
    assert test.collect_annotation_parse_funcs() == annotation_parse_funcs
