__all__ = ["model"]

from icevision.imports import *
from icevision.models.torchvision_models.utils import *
from icevision.backbones import resnet_fpn

from torchvision.models.detection.retinanet import (
    retinanet_resnet50_fpn,
    RetinaNet,
    RetinaNetHead,
)


def model(
    num_classes: int,
    backbone: Optional[nn.Module] = None,
    remove_internal_transforms: bool = True,
    pretrained: bool = True,
    **retinanet_kwargs
) -> nn.Module:
    if backbone is None:
        model = retinanet_resnet50_fpn(
            pretrained=pretrained, pretrained_backbone=pretrained, **retinanet_kwargs
        )
        model.head = RetinaNetHead(
            in_channels=model.backbone.out_channels,
            num_anchors=model.head.classification_head.num_anchors,
            num_classes=num_classes,
        )
        resnet_fpn.patch_param_groups(model.backbone)
    else:
        model = RetinaNet(
            backbone=backbone, num_classes=num_classes, **retinanet_kwargs
        )

    patch_retinanet_param_groups(model)

    if remove_internal_transforms:
        remove_internal_model_transforms(model)

    return model
