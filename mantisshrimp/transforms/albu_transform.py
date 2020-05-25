__all__ = ['AlbuTransform']

from ..imports import *
from ..utils import *
from ..core import *
from .transform import *

class AlbuTransform(Transform):
    def __init__(self, tfms):
        import albumentations as A
        self.bbox_params = A.BboxParams(format='pascal_voc', label_fields=['labels'])
        super().__init__(tfms=A.Compose(tfms, bbox_params=self.bbox_params))

    def apply(self, img, labels, bboxes=None, masks=None, iscrowds=None, **kwargs):
        # Substitue labels with list of idxs, so we can also filter out iscrowd in case any bbox is removed
        # TODO: Same should be done if a mask is completely removed from the image (if bbox is not given)
        d = self.tfms(image=img, labels=list(range_of(labels)),
                      masks=masks.data if masks else None,
                      bboxes=lmap(lambda o: o.xyxy, bboxes))
        return {
            'img': d['image'],
            'labels': [labels[i] for i in d['labels']],
            'bboxes': lmap(lambda o: BBox.from_xyxy(*o), d['bboxes']),
            'masks': ifnotnone(d['masks'], lambda o: MaskArray(np.stack(o))),
            'iscrowds': lmap(iscrowds.__getitem__, d['labels']),
        }
