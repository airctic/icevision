__all__ = ["AlbuTransform"]

from mantisshrimp.imports import *
from mantisshrimp.core import *
from mantisshrimp.transforms.transform import *


class AlbuTransform(Transform):
    def __init__(self, tfms):
        import albumentations as A

        self.bbox_params = A.BboxParams(format="pascal_voc", label_fields=["labels"])
        super().__init__(tfms=A.Compose(tfms, bbox_params=self.bbox_params))

    def apply(
        self,
        img: np.ndarray,
        labels=None,
        bboxes: List[BBox] = None,
        masks: MaskArray = None,
        iscrowds: List[int] = None,
        **kwargs
    ):
        # Substitue labels with list of idxs, so we can also filter out iscrowds in case any bboxes is removed
        # TODO: Same should be done if a masks is completely removed from the image (if bboxes is not given)
        params = {"image": img}
        params["labels"] = list(range_of(labels)) if labels is not None else []
        params["bboxes"] = [o.xyxy for o in bboxes] if bboxes is not None else []
        if masks is not None:
            params["masks"] = masks.data

        d = self.tfms(**params)

        out = {"img": d["image"]}
        out["height"], out["width"], _ = out["img"].shape

        if labels is not None:
            out["labels"] = [labels[i] for i in d["labels"]]
        if bboxes is not None:
            out["bboxes"] = [BBox.from_xyxy(*points) for points in d["bboxes"]]
        if masks is not None:
            out["masks"] = MaskArray(np.stack(d["masks"]))
        if iscrowds is not None:
            out["iscrowds"] = [iscrowds[i] for i in d["labels"]]
        return out
