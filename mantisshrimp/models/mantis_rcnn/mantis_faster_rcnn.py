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
        backbone=None,
        pretrained=True,
        fpn=True,
        metrics=None,
        **kwargs,
    ):
        super().__init__(metrics=metrics)
        self.n_class, self.h, self.backbone, self.pretrained, self.fpn = (
            n_class,
            h,
            backbone,
            pretrained,
            fpn
        )

        supported_resnet_models = ['resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']

        if self.backbone is None:
            # Creates the default fasterrcnn as given in pytorch. Trained on COCO dataset
            self.m = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=self.n_classes, 
            pretrained_backbone=True, **kwargs)
            in_features = self.m.roi_heads.box_predictor.cls_score.in_features
            self.m.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.n_class)
        
        elif isinstance(self.backbone , str):
            # Giving string as a backbone, which is either supported resnet or backbone
            if self.fpn is True:
                # Creates a torchvision resnet model with fpn added
                # Will need to add support for other models with fpn as well
                if self.backbone in supported_resnet_models:
                    self.m = resnet_fpn_backbone(backbone_name=self.backbone, pretrained=False)
                    self.m = FasterRCNN(self.backbone, self.n_class, **kwargs)
                else: 
                    raise NotImplementedError("FPN for non resnets is not supported yet")

            else:
                # This does not create fpn backbone, it is supported for all models
                self.base_model = create_torchvision_backbone(backbone=backbone, pretrained=self.pretrained)
                self.m = FasterRCNN(backbone=self.base_model, num_classes=self.n_class, **kwargs)
        
        elif isinstance(self.backbone, torch.nn.modules.container.Sequential):
            # Trying to create the backbone from CNN passed.
            try:
                self.base_model = self.backbone
                self.m = FasterRCNN(backbone=self.base_model, num_classes=self.n_class, **kwargs)
            except Exception:
                raise ("Could not parse your CNN as RCNN backbone")

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
