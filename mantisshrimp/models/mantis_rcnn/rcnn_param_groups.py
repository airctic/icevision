__all__ = ["check_all_model_params_in_groups", "resnet_fpn_backbone_param_groups"]

from mantisshrimp.imports import *


def check_all_model_params_in_groups(model, param_groups):
    num_model_params = len(list(model.parameters()))
    num_param_groups_params = len(list(nn.Sequential(*param_groups).parameters()))
    if num_model_params != num_param_groups_params:
        raise RuntimeError(
            f"{num_model_params} params in model but only {num_param_groups_params} "
            "in parameter group"
        )


def resnet_fpn_backbone_param_groups(model):
    body = model.body
    param_groups = []
    param_groups += [nn.Sequential(body.conv1, body.bn1)]
    param_groups += [getattr(body, l) for l in list(body) if l.startswith("layer")]
    param_groups += [model.fpn]
    check_all_model_params_in_groups(model, param_groups)
    return param_groups
