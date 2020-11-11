__all__ = ["model"]

from icevision.imports import *
from icevision.backbones import resnet_fpn
from icevision.models.torchvision_models.utils import *

from torchvision.models.detection.faster_rcnn import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN,
    FastRCNNPredictor,
)


def model(
    num_classes: int,
    backbone: Optional[nn.Module] = None,
    remove_internal_transforms: bool = True,
    pretrained: bool = True,
    **faster_rcnn_kwargs
) -> nn.Module:
    """FasterRCNN model implemented by torchvision.

    # Arguments
        num_classes: Number of classes.
        backbone: Backbone model to use. Defaults to a resnet50_fpn model.
        remove_internal_transforms: The torchvision model internally applies transforms
        like resizing and normalization, but we already do this at the `Dataset` level,
        so it's safe to remove those internal transforms.
        pretrained: Argument passed to `fastercnn_resnet50_fpn` if `backbone is None`.
        By default it is set to True: this is generally used when training a new model (transfer learning).
        `pretrained = False`  is used during inference (prediction) for cases where the users have their own pretrained weights.
        **faster_rcnn_kwargs: Keyword arguments that internally are going to be passed to
        `torchvision.models.detection.faster_rcnn.FastRCNN`.

    # Returns
        A Pytorch `nn.Module`.
    """
    if backbone is None:
        model = fasterrcnn_resnet50_fpn(
            pretrained=pretrained, pretrained_backbone=pretrained, **faster_rcnn_kwargs
        )

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        resnet_fpn.patch_param_groups(model.backbone)
    else:
        model = FasterRCNN(backbone, num_classes=num_classes, **faster_rcnn_kwargs)

    patch_rcnn_param_groups(model=model)

    if remove_internal_transforms:
        remove_internal_model_transforms(model)

    return model
