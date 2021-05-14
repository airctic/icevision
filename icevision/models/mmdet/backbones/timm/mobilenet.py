from icevision.backbones.timm.mobilenet import *
from icevision.models.mmdet.backbones.timm.common import *
from mmdet.models.builder import BACKBONES


@BACKBONES.register_module(force=True)
class MobileNetV2_100(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_mobilenetv2_100(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )


@BACKBONES.register_module(force=True)
class MobileNetV2_110D(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_mobilenetv2_110d(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )


@BACKBONES.register_module(force=True)
class MobileNetV2_120D(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_mobilenetv2_120d(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )


@BACKBONES.register_module(force=True)
class MobileNetv2_140(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_mobilenetv2_140(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )


@BACKBONES.register_module(force=True)
class MobileNetV3_Large_075(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_mobilenetv3_large_075(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )


@BACKBONES.register_module(force=True)
class MobileNetv3_Large_100(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_mobilenetv3_large_100(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )


@BACKBONES.register_module(force=True)
class MobileNetV3_RW(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_mobilenetv3_rw(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )


@BACKBONES.register_module(force=True)
class MobileNetV3_Small_075(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_mobilenetv3_small_075(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )


@BACKBONES.register_module(force=True)
class MobileNetV3_Small_100(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_mobilenetv3_small_100(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )


@BACKBONES.register_module(force=True)
class TF_MobileNetV3_Large_075(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_tf_mobilenetv3_large_075(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )


@BACKBONES.register_module(force=True)
class TF_MobileNetV3_Large_100(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_tf_mobilenetv3_large_100(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )


@BACKBONES.register_module(force=True)
class TF_MobileNetV3_Large_Minimal_100(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_tf_mobilenetv3_large_minimal_100(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )


@BACKBONES.register_module(force=True)
class TF_MobileNetV3_Small_075(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_tf_mobilenetv3_small_075(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )


@BACKBONES.register_module(force=True)
class TF_MobileNetV3_Small_100(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_tf_mobilenetv3_small_100(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )


@BACKBONES.register_module(force=True)
class TF_MobileNetV3_Small_Minimal_100(MMDetTimmBase):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__(pretrained=pretrained, out_indices=out_indices, **kwargs)
        self.model = ice_tf_mobilenetv3_small_minimal_100(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )
