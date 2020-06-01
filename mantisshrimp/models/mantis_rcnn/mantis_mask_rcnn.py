__all__ = ["MantisMaskRCNN"]

from ...imports import *
from ...core import *
from .rcnn_param_groups import *
from .mantis_rcnn import *


class MantisMaskRCNN(MantisRCNN):
    @delegates(MaskRCNN.__init__)
    def __init__(self, n_class, h=256, pretrained=True, metrics=None, **kwargs):
        super().__init__(metrics=metrics)
        self.n_class, self.h, self.pretrained = n_class, h, pretrained
        self.m = maskrcnn_resnet50_fpn(pretrained=self.pretrained, **kwargs)
        in_features = self.m.roi_heads.box_predictor.cls_score.in_features
        self.m.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.n_class)
        in_features_mask = self.m.roi_heads.mask_predictor.conv5_mask.in_channels
        self.m.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, self.h, self.n_class
        )

    def forward(self, images, targets=None):
        return self.m(images, targets)

    def model_splits(self):
        return split_rcnn_model(self.m)

    @staticmethod
    def item2training_sample(item: Item):
        x = im2tensor(item.img)
        _fake_box = [0, 1, 2, 3]
        y = {
            "image_id": tensor(item.imageid, dtype=torch.int64),
            "labels": tensor(item.labels or [0], dtype=torch.int64),
            "boxes": tensor(
                [o.xyxy for o in item.bboxes] or [_fake_box], dtype=torch.float
            ),
            "area": tensor([o.area for o in item.bboxes] or [4]),
            "iscrowd": tensor(item.iscrowds or [0], dtype=torch.uint8),
            "masks": tensor(item.masks.data, dtype=torch.uint8),
        }
        return x, y
