from icevision.backbones.timm.mobilenet import *
from icevision.models.mmdet.backbones.timm.common import *
from mmdet.models.builder import BACKBONES


@BACKBONES.register_module(force=True)
class MobileNetv3Large100(MMDetTimmBackbone):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_mobilenetv3_large_100(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )

