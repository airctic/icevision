__all__ = ["MantisFasterRCNN"]

from mantisshrimp.imports import *
from mantisshrimp.core import *
from mantisshrimp.models.mantis_rcnn.rcnn_param_groups import *
from mantisshrimp.models.mantis_rcnn.mantis_rcnn import *
from mantisshrimp.backbones import *


class MantisFasterRCNN(MantisRCNN):
    """
    Creates a flexible Faster RCNN implementation based on torchvision library.
    Args: 
    n_class (int) : number of classes. Do not have class_id "0" it is reserved as background. n_class = number of classes to label + 1 for background.
    backbone (str or torch.nn.Module): If none creates a default resnet50_fpn model trained on MS COCO 2017
    Supported backones are: "resnet18", "resnet34","resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d", "wide_resnet50_2", "wide_resnet101_2",
    as resnets with fpn backbones.
    Without fpn backbones supported are: "resnet18", "resnet34", "resnet50","resnet101", "resnet152", "resnext101_32x8d", "mobilenet", "vgg11", "vgg13", "vgg16", "vgg19",
    pretrained (bool): Creates a pretrained backbone with imagenet weights.
    """

    @delegates(FasterRCNN.__init__)
    def __init__(
        self, n_class: int, backbone: nn.Module = None, metrics=None, **kwargs,
    ):
        super().__init__(metrics=metrics)
        self.n_class = n_class
        self.backbone = backbone
        self.supported_resnet_fpn_models = [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "resnext50_32x4d",
            "resnext101_32x8d",
            "wide_resnet50_2",
            "wide_resnet101_2",
        ]

        self.supported_non_fpn_models = [
            "mobilenet",
            "vgg11",
            "vgg13",
            "vgg16",
            "vgg19",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            ## "resnext50_32x4d",
            "resnext101_32x8d",
            ## "wide_resnet50_2",
            ## "wide_resnet101_2",
        ]

        if backbone is None:
            # Creates the default fasterrcnn as given in pytorch. Trained on COCO dataset
            self.m = fasterrcnn_resnet50_fpn(
                pretrained=True, num_classes=n_class, **kwargs,
            )
            in_features = self.m.roi_heads.box_predictor.cls_score.in_features
            self.m.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_class)
        else:
            self.m = FasterRCNN(backbone, num_classes=n_class, **kwargs)

    @staticmethod
    def get_backbone_by_name(
        name: str, fpn: bool = True, pretrained: bool = True
    ) -> nn.Module:
        # Giving string as a backbone, which is either supported resnet or backbone
        if fpn:
            # Creates a torchvision resnet model with fpn added
            # Will need to add support for other models with fpn as well
            # Passing pretrained True will initiate backbone which was trained on ImageNet
            if name in MantisFasterRCNN.supported_resnet_fpn_models:
                # It returns BackboneWithFPN model
                backbone = resnet_fpn_backbone(name, pretrained=pretrained)
            else:
                raise ValueError(f"{name} with FPN not supported")
        else:
            # This does not create fpn backbone, it is supported for all models
            if name in MantisFasterRCNN.supported_non_fpn_models:
                backbone = create_torchvision_backbone(name, pretrained=pretrained)
            else:
                raise ValueError(f"{name} not supported")
        return backbone
    
    def forward(self, images, targets=None):
        return self.m(images, targets)

    def model_splits(self):
        return split_rcnn_model(self.m)

    @staticmethod
    def build_training_sample(
        imageid: int, img: np.ndarray, label: List[int], bbox: List[BBox], **kwargs,
    ):
        x = im2tensor(img)
        # inject values when annotations are empty are disconsidered
        # because we mark label as 0 (background)
        _fake_box = [0, 1, 2, 3]
        y = {
            "image_id": tensor(imageid, dtype=torch.int64),
            "labels": tensor(label or [0], dtype=torch.int64),
            "boxes": tensor([o.xyxy for o in bbox] or [_fake_box], dtype=torch.float),
        }
        return x, y
