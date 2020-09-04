__all__ = ["parser", "ImageParser", "AnnotationParser", "BirdMaskFile"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision import parsers


def parser(data_dir: Union[str, Path], class_map: ClassMap) -> parsers.ParserInterface:
    image_parser = ImageParser(data_dir)
    annotation_parser = AnnotationParser(data_dir, class_map)
    return parsers.CombinedParser(image_parser, annotation_parser)


class ImageParser(parsers.Parser, parsers.FilepathMixin):
    def __init__(self, data_dir):
        self.image_filepaths = get_image_files(data_dir)

    def __iter__(self) -> Any:
        yield from self.image_filepaths

    def filepath(self, o) -> Union[str, Path]:
        return o

    def imageid(self, o) -> Hashable:
        return o.stem


class AnnotationParser(
    parsers.Parser, parsers.MasksMixin, parsers.BBoxesMixin, parsers.LabelsMixin
):
    def __init__(self, data_dir, class_map):
        self.mat_filepaths = get_files(
            data_dir / "annotations-mat", extensions=[".mat"]
        )
        self.class_map = class_map

    def __iter__(self) -> Any:
        yield from self.mat_filepaths

    def masks(self, o) -> List[Mask]:
        return [BirdMaskFile(o)]

    def bboxes(self, o) -> List[BBox]:
        import scipy.io

        mat = scipy.io.loadmat(str(o))
        bbox = mat["bbox"]
        xyxy = [int(bbox[pos]) for pos in ["left", "top", "right", "bottom"]]
        return [BBox.from_xyxy(*xyxy)]

    def imageid(self, o) -> Hashable:
        return o.stem

    def labels(self, o) -> List[int]:
        class_name = o.parent.name
        return [self.class_map.get_name(class_name)]


class BirdMaskFile(MaskFile):
    def to_mask(self, h, w):
        import scipy.io

        mat = scipy.io.loadmat(str(self.filepath))
        return MaskArray(mat["seg"])[None]
