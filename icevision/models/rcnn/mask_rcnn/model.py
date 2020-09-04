from icevision.imports import *
from icevision.backbones import resnet_fpn
from icevision.models.rcnn.utils import *

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
    **mask_rcnn_kwargs
) -> nn.Module:
    """MaskRCNN model implemented by torchvision.

    # Arguments
        num_classes: Number of classes.
        backbone: Backbone model to use. Defaults to a resnet50_fpn model.
        remove_internal_transforms: The torchvision model internally applies transforms
        like resizing and normalization, but we already do this at the `Dataset` level,
        so it's safe to remove those internal transforms.
        **mask_rcnn_kwargs: Keyword arguments that internally are going to be passed to
        `torchvision.models.detection.mask_rcnn.MaskRCNN`.

    # Return
        A Pytorch `nn.Module`.
    """
    if backbone is None:
        model = maskrcnn_resnet50_fpn(pretrained=True, **mask_rcnn_kwargs)

        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_channels=in_features_mask, dim_reduced=256, num_classes=num_classes
        )

        backbone_param_groups = resnet_fpn.param_groups(model.backbone)
    else:
        model = MaskRCNN(backbone, num_classes=num_classes, **mask_rcnn_kwargs)
        backbone_param_groups = backbone.param_groups()

    patch_param_groups(model=model, backbone_param_groups=backbone_param_groups)

    if remove_internal_transforms:
        remove_internal_model_transforms(model)

    return model
