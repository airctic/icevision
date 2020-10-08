__all__ = ["BaseRecord"]

from icevision.imports import *
from collections.abc import MutableMapping
from copy import copy
from .record_mixins import RecordMixin


# TODO: MutableMapping because of backwards compatability
class BaseRecord(MutableMapping, RecordMixin):
    def clean(self):
        # For loop the annotations
        pass

    def copy(self) -> "BaseRecord":
        return copy(self)

    def load(self) -> "BaseRecord":
        record = copy(self)
        record._load()
        return record

    # backwards compatiblity: implemented method to behave like a dict
    def __getitem__(self, key):
        return self.as_dict()[key]

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        delattr(self, key)

    def __iter__(self):
        yield from self.as_dict()

    def __len__(self):
        return len(self.as_dict())


# class AnnotationParserMixin(ParserMixin):
#     def collect_parse_funcs(self, funcs):
#         funcs = super().collect_info_parse_funcs(funcs)
#         return [*funcs]

# class BBoxesMixin(AnnotationParserMixin):
#     """Adds `bboxes` method to parser"""
#     def collect_parse_funcs(self, funcs=None):
#         funcs = super().collect_annotation_parse_funcs(funcs)
#         return [self._bboxes, *funcs]

#     def _bboxes(self, o, record):
#         bboxes = self.bboxes(o)
#         record.add_annotation('bboxes', bboxes)

#     @abstractmethod
#     def bboxes(self, o) -> List[BBox]:
#         pass

# class BBoxMixin:
#     def __init__(self):
#         super().__init__()
#         self.bboxes = []

#     def add_bboxes(self, bboxes:List[BBox]):
#         self.add_annotation
#         self.bboxes.extend(bboxes)

# class ImageidMixin:
#     def __init__(self):
#         super().__init__()
#         self.imageid = None

#     def set_imageid(self, imageid: int):
#         self.imageid = imageid

# class Record(BBoxMixin, LabelMixin): # Same mixins as parser
#     def __init__(self):
#         self.metadata = []
#         self.annotations = []
#         super().__init__()

#     def clean(self):
#         # needs to know what are annotations and what is metadata

# record.set_imageid()
# record.add_bbox()

# class ImageidMixin(ParserMixin):
#     """Adds `imageid` method to parser"""

#     def collect_info_parse_funcs(self, funcs=None):
#         funcs = super().collect_info_parse_funcs(funcs)
#         return {"imageid": self.imageid, **funcs}

#     def

#     @abstractmethod
#     def imageid(self, o) -> Hashable:
#         pass

# class BBoxesMixin(ParserMixin):
#     """Adds `bboxes` method to parser"""
#     def collect_annotation_parse_funcs(self, funcs=None):
#         funcs = super().collect_annotation_parse_funcs(funcs)
#         return {"bboxes": self._bboxes, **funcs}

#     def _bboxes(self, o, record):
#         bboxes = self.bboxes(o)
#         record.add_bboxes(bboxes)

#     @abstractmethod
#     def bboxes(self, o) -> List[BBox]:
#         pass