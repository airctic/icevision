__all__ = [
    "resnest50_fpn",
    "resnest101_fpn",
    "resnest200_fpn",
    "resnest269_fpn",
]

from icevision.imports import *
from icevision.utils import *
from icevision.backbones.resnet_fpn_utils import patch_param_groups
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.detection.backbone_utils import (
    resnet_fpn_backbone,
    BackboneWithFPN,
)

from icevision.soft_dependencies import SoftDependencies

if SoftDependencies.resnest:
    import resnest.torch


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# TODO: resnest_name?
def resnest_fpn_backbone(
    backbone_fn,
    pretrained,
    # norm_layer=misc_nn_ops.FrozenBatchNorm2d,
    trainable_layers=3,
    returned_layers=None,
    extra_blocks=None,
):
    """
    Constructs a specified ResNest backbone with FPN on top. Freezes the specified number of layers in the backbone.

    Examples::

        >>> from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        >>> backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)
        >>> # get some dummy image
        >>> x = torch.rand(1,3,64,64)
        >>> # compute the output
        >>> output = backbone(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('0', torch.Size([1, 256, 16, 16])),
        >>>    ('1', torch.Size([1, 256, 8, 8])),
        >>>    ('2', torch.Size([1, 256, 4, 4])),
        >>>    ('3', torch.Size([1, 256, 2, 2])),
        >>>    ('pool', torch.Size([1, 256, 1, 1]))]

    Arguments:
        backbone_name (string): resnet architecture. Possible values are 'ResNet', 'resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        norm_layer (torchvision.ops): it is recommended to use the default value. For details visit:
            (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        pretrained (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    backbone = backbone_fn(pretrained=pretrained)

    backbone.fc = Identity()
    backbone.avgpool = Identity()

    # select layers that wont be frozen
    assert trainable_layers <= 5 and trainable_layers >= 0
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][
        :trainable_layers
    ]
    # freeze layers only if pretrained backbone is used
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(
        backbone,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks=extra_blocks,
    )


def _resnest_fpn(name, pretrained: bool = True, **kwargs):
    model = resnest_fpn_backbone(backbone_fn=name, pretrained=pretrained, **kwargs)
    patch_param_groups(model)

    return model


def resnest50_fpn(pretrained: bool = True, **kwargs):
    return _resnest_fpn(resnest.torch.resnest50, pretrained=pretrained, **kwargs)


def resnest101_fpn(pretrained: bool = True, **kwargs):
    return _resnest_fpn(resnest.torch.resnest101, pretrained=pretrained, **kwargs)


def resnest200_fpn(pretrained: bool = True, **kwargs):
    return _resnest_fpn(resnest.torch.resnest200, pretrained=pretrained, **kwargs)


def resnest269_fpn(pretrained: bool = True, **kwargs):
    return _resnest_fpn(resnest.torch.resnest269, pretrained=pretrained, **kwargs)
