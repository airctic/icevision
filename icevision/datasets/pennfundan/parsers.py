__all__ = ["parser", "PennFundanParser"]

from icevision.imports import *
from icevision import *


def parser(data_dir) -> parsers.ParserInterface:
    return PennFundanParser(data_dir=data_dir)


class PennFundanParser(parsers.MaskRCNN, parsers.FilepathMixin, parsers.SizeMixin):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.filenames = get_files(data_dir / "Annotation", extensions=".txt")

    def __iter__(self):
        yield from self.filenames

    def __len__(self):
        return len(self.filenames)

    def prepare(self, o):
        self._imageid = getattr(self, "_imageid", 0) + 1
        self.lines = L(o.read().split("\n"))
        self._bboxes = []

        for line in self.lines:
            if line.startswith("Image filename"):
                filename = re.findall(r'"(.*)"', line)[0]
                self._filepath = self.data_dir.parent / filename

            elif line.startswith("Image size (X x Y x C)"):
                size_str = re.search(r"\d{3,4}\sx\s\d{3,4}\sx\s3", line).group()
                self._size = [int(o) for o in size_str.split("x")]

            elif line.startswith("Objects with ground truth"):
                # number of objects is the first number that shows in the line
                self._num_objects = int(re.findall("\d+", line)[0])

            elif line.startswith("Pixel mask for object"):
                mask_filename = re.findall(r'"(.+)"', line.split(":")[-1])[0]
                self._mask_filepath = self.data_dir.parent / mask_filename

            if line.startswith("Bounding box"):
                # find bbox coordinates in line and covert to a list
                point_pairs_str = re.findall(r"(\d+,\s\d+)", line)
                points = []
                for pairs in point_pairs_str:
                    for point in pairs.split(","):
                        points.append(int(point))

                bbox = BBox.from_xyxy(*points)
                self._bboxes.append(bbox)

    def imageid(self, o) -> int:
        return self._imageid

    def filepath(self, o) -> Union[str, Path]:
        return self._filepath

    def width(self, o) -> int:
        return self._size[0]

    def height(self, o) -> int:
        return self._size[1]

    def labels(self, o) -> List[int]:
        return [1] * self._num_objects

    def iscrowds(self, o) -> List[bool]:
        return [False] * self._num_objects

    def masks(self, o) -> List[Mask]:
        return [MaskFile(self._mask_filepath)]

    def bboxes(self, o) -> List[BBox]:
        return self._bboxes
