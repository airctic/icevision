__all__ = [
    "remove_internal_model_transforms",
    "patch_rcnn_param_groups",
    "patch_retinanet_param_groups",
]

from icevision.imports import *
from icevision.utils import *
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN


def _noop_normalize(image: Tensor) -> Tensor:
    return image


def _noop_resize(
    image: Tensor, target: Optional[Dict[str, Tensor]]
) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
    return image, target


def remove_internal_model_transforms(model: GeneralizedRCNN):
    model.transform.normalize = _noop_normalize
    model.transform.resize = _noop_resize


def patch_param_groups(
    model: nn.Module,
    head_layers: List[nn.Module],
    backbone_param_groups: List[List[nn.Parameter]],
):
    def param_groups(model: nn.Module) -> List[List[nn.Parameter]]:
        head_param_groups = [list(layer.parameters()) for layer in head_layers]

        _param_groups = backbone_param_groups + head_param_groups
        check_all_model_params_in_groups2(model, _param_groups)

        return _param_groups

    model.param_groups = MethodType(param_groups, model)


def patch_rcnn_param_groups(model: nn.Module):
    return patch_param_groups(
        model=model,
        head_layers=[model.rpn, model.roi_heads],
        backbone_param_groups=model.backbone.param_groups(),
    )


def patch_retinanet_param_groups(model: nn.Module):
    return patch_param_groups(
        model=model,
        head_layers=[model.head],
        backbone_param_groups=model.backbone.param_groups(),
    )
