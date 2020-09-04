__all__ = ["remove_internal_model_transforms", "patch_param_groups"]

from icevision.imports import *
from icevision.utils import *
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN


def remove_internal_model_transforms(model: GeneralizedRCNN):
    def noop_normalize(image):
        return image

    def noop_resize(image, target):
        return image, target

    model.transform.normalize = noop_normalize
    model.transform.resize = noop_resize


def patch_param_groups(
    model: nn.Module, backbone_param_groups: List[List[nn.Parameter]]
):
    def param_groups(model: nn.Module) -> List[List[nn.Parameter]]:
        head_layers = [model.rpn, model.roi_heads]
        head_param_groups = [list(layer.parameters()) for layer in head_layers]

        _param_groups = backbone_param_groups + head_param_groups
        check_all_model_params_in_groups2(model, _param_groups)

        return _param_groups

    model.param_groups = MethodType(param_groups, model)
