__all__ = ["mobilenet_v2_fpn"]

from turtle import back
from icevision.models.torchvision.retinanetv2.backbones.mobilenet_fpn_utils import (
    patch_param_groups,
)
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from icevision.models.torchvision.backbone_config import TorchvisionBackboneConfigV2
from torchvision.models.detection.retinanet import LastLevelP6P7


class RetinanetV2TorchVisionBackboneConfig(TorchvisionBackboneConfigV2):
    def __init__(self, backbone, **kwargs):
        self.backbone = backbone
        super().__init__(model_name="retinanet", **kwargs)


def _mobilenet_fpn(name: str, pretrained: bool = True, fpn=True, **kwargs):
    model = mobilenet_backbone(
        backbone_name=name,
        pretrained=pretrained,
        fpn=True,
        returned_layers=[2, 3, 4],
        extra_blocks=LastLevelP6P7(256, 256),
        **kwargs
    )
    patch_param_groups(model)
    return model


mobilenet_v2_fpn = RetinanetV2TorchVisionBackboneConfig(
    backbone=_mobilenet_fpn("mobilenet_v2", pretrained=True),
    weights_url="",
)
