from icevision.backbones.timm.mobilenet import *
from icevision.models.mmdet.backbones.timm.common import *
from mmdet.models.builder import BACKBONES


@BACKBONES.register_module(force=True)
class ResNest14D(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_resnest14d(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )


@BACKBONES.register_module(force=True)
class ResNest26D(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_resnest26d(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )


@BACKBONES.register_module(force=True)
class ResNest50D(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_resnest50d(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )


@BACKBONES.register_module(force=True)
class ResNest50D_1S4x24D(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_resnest50d_1s4x24d(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )


@BACKBONES.register_module(force=True)
class ResNest50D_4S2x40D(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_resnest50d_4s2x40d(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )


@BACKBONES.register_module(force=True)
class ResNest101E(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_resnest101e(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )


@BACKBONES.register_module(force=True)
class ResNest200E(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_resnest200e(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )


@BACKBONES.register_module(force=True)
class ResNest269E(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_resnest269e(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )
