from icevision.backbones.timm.mobilenet import *


@BACKBONES.register_module(force=True)
class MobileNetv3Large100(nn.Module):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__()
        self.model = ice_mobilenetv3_large_100(
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):  # should return a tuple
        return self.model(x)
