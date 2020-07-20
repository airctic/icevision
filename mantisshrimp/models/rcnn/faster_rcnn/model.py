from mantisshrimp.imports import *
from mantisshrimp.backbones import resnet_fpn
from mantisshrimp.models.rcnn.utils import *

from torchvision.models.detection.faster_rcnn import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN,
    FastRCNNPredictor,
)


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

    patch_param_groups(model=model, backbone_param_groups=backbone_param_groups)

    if remove_internal_transforms:
        remove_internal_model_transforms(model)

    return model
