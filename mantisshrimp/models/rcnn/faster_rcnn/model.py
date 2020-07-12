from mantisshrimp.imports import *
from mantisshrimp.utils import *
from mantisshrimp.backbones import resnet_fpn


def remove_transforms_from_model(model: GeneralizedRCNN):
    def noop_normalize(image):
        return image

    def noop_resize(image, target):
        return image, target

    model.transform.normalize = noop_normalize
    model.transform.resize = noop_resize


def model(
    num_classes: int,
    backbone: Optional[nn.Module] = None,
    remove_internal_transforms: bool = True,
    **faster_rcnn_kwargs
) -> nn.Module:
    """ FasterRCNN model given by torchvision

    Args:
        num_classes (int): Number of classes.
        backbone (nn.Module): Backbone model to use. Defaults to a resnet50_fpn model.

    Return:
        nn.Module
    """
    if backbone is None:
        model = fasterrcnn_resnet50_fpn(pretrained=True, **faster_rcnn_kwargs)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        backbone_param_groups = resnet_fpn.param_groups(model.backbone)
    else:
        model = FasterRCNN(backbone, num_classes=num_classes, **faster_rcnn_kwargs)

        backbone_param_groups = backbone.param_groups()

    def param_groups(model: nn.Module) -> List[List[nn.Parameter]]:
        head_layers = [model.rpn, model.roi_heads]
        head_param_groups = [list(layer.parameters()) for layer in head_layers]

        _param_groups = backbone_param_groups + head_param_groups
        check_all_model_params_in_groups2(model, _param_groups)

        return _param_groups

    model.param_groups = MethodType(param_groups, model)

    if remove_internal_transforms:
        remove_transforms_from_model(model)

    return model
