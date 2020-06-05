from mantisshrimp import *
from mantisshrimp.imports import Union, Path, np


def test_all_parser_mixins():
    class TestAllParserMixins(
        ImageidParserMixin,
        FilepathParserMixin,
        SizeParserMixin,
        LabelParserMixin,
        BBoxParserMixin,
        MaskParserMixin,
        IsCrowdParserMixin,
    ):
        def imageid(self, o) -> int:
            return 42

        def filepath(self, o) -> Union[str, Path]:
            return "path"

        def height(self, o) -> int:
            return 420

        def width(self, o) -> int:
            return 480

        def label(self, o):
            return 0

        def bbox(self, o) -> BBox:
            return BBox.from_xyxy(1, 2, 3, 4)

        def mask(self, o) -> MaskArray:
            return MaskArray(np.array([]))

        def iscrowd(self, o) -> bool:
            return 0

    test = TestAllParserMixins()
    info_parse_funcs = {
        "imageid": test.imageid,
        "height": test.height,
        "width": test.width,
        "filepath": test.filepath,
    }
    annotation_parse_funcs = {
        "label": test.label,
        "bbox": test.bbox,
        "mask": test.mask,
        "iscrowd": test.iscrowd,
    }
    assert test.collect_info_parse_funcs() == info_parse_funcs
    assert test.collect_annotation_parse_funcs() == annotation_parse_funcs
