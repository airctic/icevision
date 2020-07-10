__all__ = [
    "to_np",
    "requires_grad",
    "model_device",
    "params",
    "check_all_model_params_in_groups2",
]

from mantisshrimp.imports import *


def to_np(t):
    return t.detach().cpu().numpy()


def requires_grad(model, layer):
    return list(model.parameters())[layer].requires_grad


def model_device(model):
    return first(model.parameters()).device


def params(m):
    return list(m.parameters())


def check_all_model_params_in_groups2(
    model: nn.Module, param_groups: List[List[nn.Parameter]]
):
    num_params = len([param for group in param_groups for param in group])
    num_params_expected = len(list(model.parameters()))

    if num_params != num_params_expected:
        raise RuntimeError(
            f"{num_params_expected} params in model but only {num_params} "
            "in parameter group"
        )
