from mantisshrimp.imports import *
from mantisshrimp import *


class PennFundanParser(DefaultImageInfoParser, MaskRCNNParser):
    def __init__(self, source):
        self.source = source
        self.filenames = get_files(source / "Annotation", extensions=".txt")

    def __iter__(self):
        yield from self.filenames

    def prepare(self, o):
        self._imageid = getattr(self, "_imageid", 0) + 1
        self.lines = L(o.read().split("\n"))
        self._bboxes = []

        for line in self.lines:
            if line.startswith("Image filename"):
                filename = re.findall(r'"(.*)"', line)[0]
                self._filepath = self.source.parent / filename

            elif line.startswith("Image size (X x Y x C)"):
                size_str = re.search(r"\d{3,4}\sx\s\d{3,4}\sx\s3", line).group()
                self._size = [int(o) for o in size_str.split("x")]

            elif line.startswith("Objects with ground truth"):
                # number of objects is the first number that shows in the line
                self._num_objects = int(re.findall("\d+", line)[0])

            elif line.startswith("Pixel mask for object"):
                mask_filename = re.findall(r'"(.+)"', line.split(":")[-1])[0]
                self._mask_filepath = self.source.parent / mask_filename

            elif line.startswith("Bounding Box"):
                points_str = re.findall(r"(\d+,\s\d+)", line)
                points_int = [int(point) for point in points_str.split(",")]
                points_arr = np.array(points_int).flat
                bbox = BBox.from_xyxy(*points_arr)
                self._bboxes.append(bbox)

    def imageid(self, o) -> int:
        return self._imageid

    def filepath(self, o) -> Union[str, Path]:
        return self._filepath

    def height(self, o) -> int:
        return self._size[1]

    def width(self, o) -> int:
        return self._size[0]

    def label(self, o) -> List[int]:
        return [1] * self._num_objects

    def iscrowd(self, o) -> List[bool]:
        return [False] * self._num_objects

    def mask(self, o) -> List[Mask]:
        return [MaskFile(self._mask_filepath)]

    def bbox(self, o) -> List[BBox]:
        return self._bboxes


source = Path("/Users/lgvaz/.data/PennFudanPed/")
parser = PennFundanParser(source)

splitter = RandomSplitter([0.8, 0.2])

train_records, valid_records = parser.parse(splitter)
parser.filenames

len(train_records)
len(valid_records)
