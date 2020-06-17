__all__ = ["MantisMaskRCNN"]

from mantisshrimp.imports import *
from mantisshrimp.core import *
from mantisshrimp.models.mantis_rcnn.rcnn_param_groups import *
from mantisshrimp.models.mantis_rcnn.mantis_rcnn import *
from mantisshrimp.models.mantis_rcnn.mantis_faster_rcnn import *
from mantisshrimp.backbones import *


class MantisMaskRCNN(MantisRCNN):
    @delegates(MaskRCNN.__init__)
    def __init__(
        self, num_classes: int, backbone: nn.Module = None, metrics=None, **kwargs,
    ):
        super().__init__(metrics=metrics)
        self.num_classes = num_classes
        self.backbone = backbone

        if backbone is None:
            # Creates the default fasterrcnn as given in pytorch. Trained on COCO dataset
            self.m = maskrcnn_resnet50_fpn(pretrained=True, **kwargs,)
            in_features = self.m.roi_heads.box_predictor.cls_score.in_features

            self.m.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            in_features_mask = self.m.roi_heads.mask_predictor.conv5_mask.in_channels
            self.m.roi_heads.mask_predictor = MaskRCNNPredictor(
                in_features_mask, self.num_classes
            )

        else:
            self.m = MaskRCNN(backbone, num_classes=num_classes, **kwargs)

    @staticmethod
    def get_backbone_by_name(
        name: str, fpn: bool = True, pretrained: bool = True
    ) -> nn.Module:
        """
        Args:
            backbone (str): If none creates a default resnet50_fpn model trained on MS COCO 2017
                Supported backones are: "resnet18", "resnet34","resnet50", "resnet101", "resnet152",
                 "resnext50_32x4d", "resnext101_32x8d", "wide_resnet50_2", "wide_resnet101_2", as resnets with fpn backbones.
                Without fpn backbones supported are: "resnet18", "resnet34", "resnet50","resnet101",
                 "resnet152", "resnext101_32x8d", "mobilenet", "vgg11", "vgg13", "vgg16", "vgg19",
            pretrained (bool): Creates a pretrained backbone with imagenet weights.
        """
        # Giving string as a backbone, which is either supported resnet or backbone
        if fpn:
            # Creates a torchvision resnet model with fpn added
            # It returns BackboneWithFPN model
            backbone = resnet_fpn_backbone(name, pretrained=pretrained)
        else:
            # This does not create fpn backbone, it is supported for all models
            backbone = create_torchvision_backbone(name, pretrained=pretrained)
        return backbone

    def forward(self, images, targets=None):
        return self.m(images, targets)

    def model_splits(self):
        return split_rcnn_model(self.m)

    @staticmethod
    def build_training_sample(
        imageid: int,
        img: np.ndarray,
        label: List[int],
        bbox: List[BBox],
        mask: MaskArray,
        **kwargs,
    ):
        x, y = MantisFasterRCNN.build_training_sample(
            imageid=imageid, img=img, label=label, bbox=bbox,
        )
        y["masks"] = tensor(mask.data, dtype=torch.uint8)
        return x, y
