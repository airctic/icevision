__all__ = ["resnet50_fpn", "resnet18_fpn"]

from icevision.models.torchvision.retinanetv2.backbones.resnet_fpn_utils import (
    patch_param_groups,
)
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.retinanet import LastLevelP6P7
from icevision.models.torchvision.retinanetv2.backbones.backbone_config import (
    RetinanetV2TorchVisionBackboneConfig,
)


def _resnet_fpn(name: str, pretrained: bool = True, **kwargs):
    model = resnet_fpn_backbone(
        backbone_name=name,
        pretrained=pretrained,
        returned_layers=[2, 3, 4],
        extra_blocks=LastLevelP6P7(256, 256),
        **kwargs
    )
    patch_param_groups(model)
    return model


resnet50_fpn = RetinanetV2TorchVisionBackboneConfig(
    backbone=_resnet_fpn("resnet50"),
    weights_url="https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth",
)

resnet18_fpn = RetinanetV2TorchVisionBackboneConfig(
    backbone=_resnet_fpn("resnet18"),
    weights_url="",  # No pretrained weights found for this model.
)
