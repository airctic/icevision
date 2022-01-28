__all__ = [
    "param_groups",
    "patch_param_groups",
]

from icevision.imports import *
from icevision.utils import *


def param_groups(model: nn.Module) -> List[nn.Parameter]:
    body = model.body

    layers = []
    layers += [nn.Sequential(body.conv1, body.bn1)]
    layers += [getattr(body, l) for l in list(body) if l.startswith("layer")]
    layers += [model.fpn]

    _param_groups = [list(layer.parameters()) for layer in layers]
    check_all_model_params_in_groups2(model, _param_groups)

    return _param_groups


def patch_param_groups(model: nn.Module) -> None:
    model.param_groups = MethodType(param_groups, model)
