__all__ = ["DefaultImageInfoParser"]

from mantisshrimp.imports import *
from mantisshrimp.parsers.parser import *
from mantisshrimp.parsers.mixins import *


class DefaultImageInfoParser(
    Parser, FilepathParserMixin, SizeParserMixin, ABC,
):
    pass
