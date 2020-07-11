from mantisshrimp.imports import *
from mantisshrimp.core import *
from mantisshrimp.models.mantis_rcnn.rcnn_param_groups import *
from mantisshrimp.models.mantis_rcnn.mantis_rcnn import *

SplitModelInterface = Callable[[nn.Module], List[nn.Parameter]]


def model(
    num_classes: int,
    backbone: nn.Module,
    backbone_param_groups: Optional[List[nn.Parameter]] = None,
    **faster_rcnn_kwargs
):
    if backbone is None:
        # Creates the default fasterrcnn as given in pytorch. Trained on COCO dataset
        model = fasterrcnn_resnet50_fpn(pretrained=True, **faster_rcnn_kwargs)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        backbone_param_groups = resnet_fpn_backbone_param_groups(model.backbone)
    else:
        model = FasterRCNN(backbone, num_classes=num_classes, **faster_rcnn_kwargs)
        backbone_param_groups = backbone.param_groups(backbone)
        # backbone_param_groups = backbone_param_groups or [backbone]

    param_groups = backbone_param_groups + [model.rpn, model.roi_heads]
    check_all_model_params_in_groups(model, param_groups)

    # if remove_internal_transforms:
    #     self._remove_transforms_from_model(self.model)
