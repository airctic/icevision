from icevision.imports import *
from icevision.backbones import resnet_fpn
from icevision.models.torchvision_models.utils import *

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import (
    maskrcnn_resnet50_fpn,
    MaskRCNN,
    MaskRCNNPredictor,
)


def model(
    num_classes: int,
    backbone: Optional[nn.Module] = None,
    remove_internal_transforms: bool = True,
    pretrained: bool = True,
    **mask_rcnn_kwargs
) -> nn.Module:
    """MaskRCNN model implemented by torchvision.

    # Arguments
        num_classes: Number of classes.
        backbone: Backbone model to use. Defaults to a resnet50_fpn model.
        remove_internal_transforms: The torchvision model internally applies transforms
        like resizing and normalization, but we already do this at the `Dataset` level,
        so it's safe to remove those internal transforms.
        pretrained: Argument passed to `maskrcnn_resnet50_fpn` if `backbone is None`.
        By default it is set to True: this is generally used when training a new model (transfer learning).
        `pretrained = False`  is used during inference (prediction) for cases where the users have their own pretrained weights.
        **mask_rcnn_kwargs: Keyword arguments that internally are going to be passed to
        `torchvision.models.detection.mask_rcnn.MaskRCNN`.

    # Return
        A Pytorch `nn.Module`.
    """
    if backbone is None:
        model = maskrcnn_resnet50_fpn(
            pretrained=pretrained, pretrained_backbone=pretrained, **mask_rcnn_kwargs
        )

        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_channels=in_features_mask, dim_reduced=256, num_classes=num_classes
        )

        resnet_fpn.patch_param_groups(model.backbone)
    else:
        model = MaskRCNN(backbone, num_classes=num_classes, **mask_rcnn_kwargs)

    patch_rcnn_param_groups(model=model)

    if remove_internal_transforms:
        remove_internal_model_transforms(model)

    return model
