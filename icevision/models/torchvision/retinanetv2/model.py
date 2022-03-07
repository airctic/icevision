__all__ = ["model"]

from icevision.imports import *
from icevision.models.torchvision.utils import *
from icevision.models.torchvision.backbone_config import TorchvisionBackboneConfigV2
from icevision.models.torchvision.retinanetv2.backbones.resnet_fpn_utils import (
    patch_param_groups,
)
from torch.hub import load_state_dict_from_url
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.retinanet import (
    retinanet_resnet50_fpn,
    RetinaNet,
    RetinaNetHead,
)


def model(
    num_classes: int,
    backbone: Optional[TorchvisionBackboneConfigV2] = None,
    remove_internal_transforms: bool = True,
    pretrained: bool = True,
    **retinanet_kwargs,
) -> nn.Module:
    if backbone is None:
        model = retinanet_resnet50_fpn(
            pretrained=pretrained, pretrained_backbone=pretrained, **retinanet_kwargs
        )
        patch_param_groups(model.backbone)
    else:

        model = RetinaNet(
            backbone=backbone.backbone, num_classes=91, **retinanet_kwargs
        )

        if pretrained and backbone.weights_url:
            print(f"Loading pretrained model from {backbone.weights_url}")
            state_dict = load_state_dict_from_url(backbone.weights_url)
            model.load_state_dict(state_dict, strict=False)
            overwrite_eps(model, 0.0)  # Not sure if this is needed

    model.head = RetinaNetHead(
        in_channels=model.backbone.out_channels,
        num_anchors=model.head.classification_head.num_anchors,
        num_classes=num_classes,
    )
    patch_retinanet_param_groups(model)

    if remove_internal_transforms:
        remove_internal_model_transforms(model)

    return model
