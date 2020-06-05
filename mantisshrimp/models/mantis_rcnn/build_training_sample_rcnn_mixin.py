__all__ = ["BuildTrainingSampleFasterRCNNMixin", "BuildTrainingSampleMaskRCNNMixin"]

from mantisshrimp.imports import *
from mantisshrimp.core import *


class BuildTrainingSampleFasterRCNNMixin(ABC):
    def build_training_sample(
        self,
        imageid: int,
        img: np.ndarray,
        label: List[int],
        bbox: List[BBox],
        **kwargs,
    ):
        x = im2tensor(img)
        _fake_box = [0, 1, 2, 3]
        y = {
            "image_id": tensor(imageid, dtype=torch.int64),
            "labels": tensor(label or [0], dtype=torch.int64),
            "boxes": tensor([o.xyxy for o in bbox] or [_fake_box], dtype=torch.float),
            "area": tensor([o.area for o in bbox] or [4]),
        }
        return x, y


class BuildTrainingSampleMaskRCNNMixin(BuildTrainingSampleFasterRCNNMixin, ABC):
    def build_training_sample(
        self,
        imageid: int,
        img: np.ndarray,
        label: List[int],
        bbox: List[BBox],
        iscrowd: List[bool],
        mask: MaskArray,
        **kwargs,
    ):
        x, y = super().build_training_sample(
            imageid=imageid, img=img, label=label, bbox=bbox, iscrowd=iscrowd
        )
        y["masks"] = tensor(mask.data, dtype=torch.uint8)
        y["iscrowd"] = tensor(iscrowd or [0], dtype=torch.uint8)
        return x, y
