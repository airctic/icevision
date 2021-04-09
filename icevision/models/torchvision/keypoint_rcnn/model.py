__all__ = ["model"]

from icevision.imports import *
from icevision.backbones import resnet_fpn
from icevision.models.torchvision.utils import *
from icevision.models.torchvision.backbones.backbone_config import (
    TorchvisionBackboneConfig,
)

from torchvision.models.detection.keypoint_rcnn import (
    keypointrcnn_resnet50_fpn,
    KeypointRCNNPredictor,
    KeypointRCNN,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def model(
    num_keypoints: int,
    num_classes: int = 2,
    backbone: Optional[TorchvisionBackboneConfig] = None,
    remove_internal_transforms: bool = True,
    **keypoint_rcnn_kwargs
) -> nn.Module:
    """KeypointRCNN model implemented by torchvision.

    # Arguments
        num_keypoints: Number of keypoints (e.g. 17 in case of COCO).
        num_classes: Number of classes (including background).
        backbone: Backbone model to use. Defaults to a resnet50_fpn model.
        remove_internal_transforms: The torchvision model internally applies transforms
        like resizing and normalization, but we already do this at the `Dataset` level,
        so it's safe to remove those internal transforms.
        **keypoint_rcnn_kwargs: Keyword arguments that internally are going to be passed to
        `torchvision.models.detection.keypoint_rcnn.KeypointRCNN`.

    # Returns
        A Pytorch `nn.Module`.
    """
    if backbone is None:
        model = keypointrcnn_resnet50_fpn(
            pretrained=True, pretrained_backbone=True, **keypoint_rcnn_kwargs
        )

        in_channels = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
        model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(
            in_channels, num_keypoints
        )

        if num_classes != 2:
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        resnet_fpn.patch_param_groups(model.backbone)
    else:
        model = KeypointRCNN(
            backbone.backbone,
            num_classes=num_classes,
            num_keypoints=num_keypoints,
            **keypoint_rcnn_kwargs
        )

    patch_rcnn_param_groups(model=model)

    if remove_internal_transforms:
        remove_internal_model_transforms(model)

    return model
