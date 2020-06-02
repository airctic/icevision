__all__ = ["AlbuTransform"]

from ..imports import *
from ..utils import *
from ..core import *
from .transform import *


class AlbuTransform(Transform):
    def __init__(self, tfms):
        import albumentations as A

        self.bbox_params = A.BboxParams(format="pascal_voc", label_fields=["label"])
        super().__init__(tfms=A.Compose(tfms, bbox_params=self.bbox_params))

    def apply(
        self,
        img: np.ndarray,
        label=None,
        bbox: List[BBox] = None,
        mask: MaskArray = None,
        iscrowd: List[int] = None,
        **kwargs
    ):
        # Substitue label with list of idxs, so we can also filter out iscrowd in case any bbox is removed
        # TODO: Same should be done if a mask is completely removed from the image (if bbox is not given)
        params = {"image": img}
        params["label"] = list(range_of(label)) if label is not None else []
        params["bboxes"] = [o.xyxy for o in bbox] if bbox is not None else []
        if mask is not None:
            params["masks"] = mask.data

        d = self.tfms(**params)

        out = {"img": d["image"]}
        if label is not None:
            out["label"] = [label[i] for i in d["label"]]
        if bbox is not None:
            out["bbox"] = [BBox.from_xyxy(*points) for points in d["bboxes"]]
        if mask is not None:
            out["mask"] = MaskArray(np.stack(d["masks"]))
        if iscrowd is not None:
            out["iscrowd"] = [iscrowd[i] for i in d["label"]]
        return out
