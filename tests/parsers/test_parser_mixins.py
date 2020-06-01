from mantisshrimp import *
from mantisshrimp.imports import Union, Path


def test_all_parser_mixins():
    class TestAllParserMixins(
        ImageidParserMixin, FilepathParserMixin, SizeParserMixin, SplitParserMixin
    ):
        def imageid(self, o) -> int:
            return 42

        def filepath(self, o) -> Union[str, Path]:
            return "path"

        def height(self, o) -> int:
            return 420

        def width(self, o) -> int:
            return 480

        def split(self, o) -> int:
            return 0

    test = TestAllParserMixins()
    parse_funcs = {
        "split": test.split,
        "height": test.height,
        "width": test.width,
        "filepath": test.filepath,
        "imageid": test.imageid,
    }
    assert test.collect_parse_funcs() == parse_funcs
