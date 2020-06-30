__all__ = ["create_backbone_adaptive", "create_backbone_features"]

from mantisshrimp.imports import *

# Use this when you have Adaptive Pooling layer in End.
# When Model.features is not applicable.
def create_backbone_adaptive(model, out_channels: int = None):
    # Out channels is optional can pass it if user knows it
    # print(list(model.children())[-1].in_features)
    modules_total = list(model.children())
    modules = modules_total[:-1]
    ft_backbone = nn.Sequential(*modules)
    # print(ft_backbone)

    if out_channels is None:
        modules_out_channels = modules_total[-1].in_features
        ft_backbone.out_channels = modules_out_channels
    else:
        ft_backbone.out_channels = out_channels

    return ft_backbone


# Use this when model.features function is available as in mobilenet and vgg
# Again it is mandatory for out_channels here
def create_backbone_features(model, out_channels: int):
    ft_backbone = model.features
    ft_backbone.out_channels = out_channels
    return ft_backbone


# Use this when adaptive avg_pooling is not available or you want to pass custom CNN
# This can work with adaptive case too. So it is a bit generic case.
def create_backbone_generic(model, out_channels: int):
    # Here out_channels is a mandatory argument.
    modules_total = list(model.children())
    modules = modules_total[:-1]
    ft_backbone = nn.Sequential(*modules)
    ft_backbone.out_channels = out_channels
    return ft_backbone
