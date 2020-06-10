__all__ = ["MantisFasterRCNN"]

from mantisshrimp.imports import *
from mantisshrimp.core import *
from mantisshrimp.models.mantis_rcnn.rcnn_param_groups import *
from mantisshrimp.models.mantis_rcnn.mantis_rcnn import *
from mantisshrimp.backbones import *


class MantisFasterRCNN(MantisRCNN):
    @delegates(FasterRCNN.__init__)
    def __init__(
        self,
        n_class,
        h=256,
        backbone="default",
        pretrained=True,
        metrics=None,
        **kwargs,
    ):
        super().__init__(metrics=metrics)
        self.n_class, self.h, self.backbone, self.pretrained = (
            n_class,
            h,
            backbone,
            pretrained,
        )

        if self.backbone == "defualt":
            # Creates the default fasterrcnn as given in pytorch. Trained on COCO dataset
            self.m = fasterrcnn_resnet50_fpn(pretrained=self.pretrained, **kwargs)
            in_features = self.m.roi_heads.box_predictor.cls_score.in_features
            self.m.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, self.n_class
            )

        elif self.backbone == "hub":
            # We need to integrate pytorch hub backbones here.
            pass

        elif self.backbone == "custom":
            # User would need to write their own backbone.
            pass

        else:
            # Creates the custom backbone model trained on ImageNet.
            try:
                self.base_model = create_torchvision_backbone(
                    backbone=self.backbone, pretrained=pretrained
                )
                self.m = FasterRCNN(
                    backbone=self.base_model, num_classes=self.n_class, **kwargs,
                )
            except NotImplementedError:
                raise ("Invalid Backbone")

    def forward(self, images, targets=None):
        return self.m(images, targets)

    def model_splits(self):
        return split_rcnn_model(self.m)

    @staticmethod
    def build_training_sample(
        imageid: int, img: np.ndarray, label: List[int], bbox: List[BBox], **kwargs,
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
