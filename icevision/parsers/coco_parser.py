__all__ = [
    "coco",
    "COCOBaseParser",
    "COCOBBoxParser",
    "COCOMaskParser",
    "COCOKeyPointsParser",
    "COCOKeypointsMetadata",
]

from icevision.imports import *
from icevision.core import *
from icevision.utils import *
from icevision.parsers import *


def coco(
    annotations_file: Union[str, Path],
    img_dir: Union[str, Path],
    mask: bool = True,
) -> Parser:
    parser_cls = COCOMaskParser if mask else COCOBBoxParser
    return parser_cls(annotations_file, img_dir)


class COCOBaseParser(
    Parser, FilepathMixin, SizeMixin, LabelsMixin, AreasMixin, IsCrowdsMixin
):
    def __init__(
        self, annotations_filepath: Union[str, Path], img_dir: Union[str, Path]
    ):
        self.annotations_dict = json.loads(Path(annotations_filepath).read_bytes())
        self.img_dir = Path(img_dir)

        self._imageid2info = {o["id"]: o for o in self.annotations_dict["images"]}

    def __iter__(self):
        yield from self.annotations_dict["annotations"]

    def __len__(self):
        return len(self.annotations_dict["annotations"])

    def prepare(self, o):
        self._info = self._imageid2info[o["image_id"]]

    def imageid(self, o) -> int:
        return o["image_id"]

    def filepath(self, o) -> Path:
        return self.img_dir / self._info["file_name"]

    def image_width_height(self, o) -> Tuple[int, int]:
        return get_image_size(self.filepath(o))

    def labels(self, o) -> List[int]:
        return [o["category_id"]]

    def areas(self, o) -> List[float]:
        return [o["area"]]

    def iscrowds(self, o) -> List[bool]:
        return [o["iscrowd"]]


class COCOBBoxParser(COCOBaseParser, BBoxesMixin):
    def bboxes(self, o) -> List[BBox]:
        return [BBox.from_xywh(*o["bbox"])]


class COCOMaskParser(COCOBBoxParser, MasksMixin):
    def masks(self, o) -> List[MaskArray]:
        seg = o["segmentation"]
        if o["iscrowd"]:
            return [RLE.from_coco(seg["counts"])]
        else:
            return [Polygon(seg)]


class COCOKeyPointsParser(COCOBBoxParser, KeyPointsMixin):
    def keypoints(self, o) -> List[KeyPoints]:
        return (
            [KeyPoints.from_xyv(o["keypoints"], COCOKeypointsMetadata)]
            if sum(o["keypoints"]) > 0
            else []
        )

    def labels(self, o) -> List[int]:
        return [o["category_id"]] if sum(o["keypoints"]) > 0 else []

    def areas(self, o) -> List[float]:
        return [o["area"]] if sum(o["keypoints"]) > 0 else []

    def iscrowds(self, o) -> List[bool]:
        return [o["iscrowd"]] if sum(o["keypoints"]) > 0 else []

    def bboxes(self, o) -> List[BBox]:
        return [BBox.from_xywh(*o["bbox"])] if sum(o["keypoints"]) > 0 else []


class COCOConnectionsColor:
    head = (220, 30, 200)
    torso = (100, 240, 100)
    right_arm = (170, 170, 100)
    left_arm = (120, 120, 230)
    left_leg = (230, 120, 130)
    right_leg = (230, 180, 190)


class COCOKeypointsMetadata(KeypointsMetadata):
    labels = (
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )

    connections = (
        KeypointConnection(0, 1, COCOConnectionsColor.head),
        KeypointConnection(0, 2, COCOConnectionsColor.head),
        KeypointConnection(1, 2, COCOConnectionsColor.head),
        KeypointConnection(1, 3, COCOConnectionsColor.head),
        KeypointConnection(2, 4, COCOConnectionsColor.head),
        KeypointConnection(3, 5, COCOConnectionsColor.head),
        KeypointConnection(4, 6, COCOConnectionsColor.head),
        KeypointConnection(5, 6, COCOConnectionsColor.torso),
        KeypointConnection(5, 7, COCOConnectionsColor.left_arm),
        KeypointConnection(5, 11, COCOConnectionsColor.torso),
        KeypointConnection(6, 8, COCOConnectionsColor.right_arm),
        KeypointConnection(6, 12, COCOConnectionsColor.torso),
        KeypointConnection(7, 9, COCOConnectionsColor.left_arm),
        KeypointConnection(8, 10, COCOConnectionsColor.right_arm),
        KeypointConnection(11, 12, COCOConnectionsColor.torso),
        KeypointConnection(13, 11, COCOConnectionsColor.left_leg),
        KeypointConnection(14, 12, COCOConnectionsColor.right_leg),
        KeypointConnection(15, 13, COCOConnectionsColor.left_leg),
        KeypointConnection(16, 14, COCOConnectionsColor.right_leg),
    )
