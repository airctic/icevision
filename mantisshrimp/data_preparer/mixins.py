__all__ = ["PreparerMixin", "ImgPreparerMixin", "MaskPreparerMixin"]

from mantisshrimp.imports import *
from mantisshrimp.utils import *
from mantisshrimp.core import *


class PreparerMixin(ABC):
    @abstractmethod
    def prepare(self, record):
        pass


class ImgPreparerMixin(PreparerMixin):
    def prepare(self, record):
        super().prepare(record)
        if "filepath" in record:
            record["img"] = open_img(record["filepath"])


class MaskPreparerMixin(PreparerMixin):
    def prepare(self, record):
        super().prepare(record)
        if "mask" in record:
            record["mask"] = MaskArray.from_masks(
                record["mask"], record["height"], record["width"]
            )
