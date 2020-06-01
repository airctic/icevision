__all__ = [
    "ImageidParserMixin",
    "FilepathParserMixin",
    "SizeParserMixin",
    "SplitParserMixin",
]

from mantisshrimp.imports import *


class ParserMixin(ABC):
    @abstractmethod
    def collect_parse_funcs(self, funcs=None):
        return funcs or {}


class ImageidParserMixin(ParserMixin):
    def collect_parse_funcs(self, funcs=None):
        funcs = super().collect_parse_funcs(funcs)
        return {"imageid": self.imageid, **funcs}

    @abstractmethod
    def imageid(self, o) -> int:
        pass


class FilepathParserMixin(ParserMixin):
    def collect_parse_funcs(self, funcs=None):
        funcs = super().collect_parse_funcs(funcs)
        return {"filepath": self.filepath, **funcs}

    @abstractmethod
    def filepath(self, o) -> Union[str, Path]:
        pass


class SizeParserMixin(ParserMixin):
    def collect_parse_funcs(self, funcs=None):
        funcs = super().collect_parse_funcs(funcs)
        return {"height": self.height, "width": self.width, **funcs}

    @abstractmethod
    def height(self, o) -> int:
        pass

    @abstractmethod
    def width(self, o) -> int:
        pass


class SplitParserMixin(ParserMixin):
    def collect_parse_funcs(self, funcs=None):
        funcs = super().collect_parse_funcs(funcs)
        return {"split": self.split, **funcs}

    @abstractmethod
    def split(self, o) -> int:
        pass


# testing
# class Test(FilepathParserMixin, ImageidParserMixin):
#     def filepath(self, o) -> Union[str, Path]:
#         return "path"
#
#     def imageid(self, o) -> int:
#         return 42
