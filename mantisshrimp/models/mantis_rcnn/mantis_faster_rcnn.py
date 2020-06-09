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
        else:
            # Creates the custom backbone model trained on ImageNet.
            try:
                self.base_model = create_torchvision_backbone(
                    backbone=self.backbone, is_pretrained=pretrained
                )
                self.ft_anchor_generator = AnchorGenerator(
                    sizes=((32, 64, 128)), aspect_ratios=((0.5, 1.0, 2.0))
                )
                self.ft_mean = [0.485, 0.456, 0.406]  # imagenet mean and std
                self.ft_std = [0.229, 0.224, 0.225]
                self.m = FasterRCNN(
                    backbone=self.base_model,
                    num_classes=self.n_class,
                    image_mean=self.ft_mean,
                    image_std=self.ft_std,
                )
            except NotImplementedError:
                print("Invalid Backbone")

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
