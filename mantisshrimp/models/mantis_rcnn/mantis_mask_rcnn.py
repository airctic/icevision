__all__ = ["MantisMaskRCNN"]

from mantisshrimp.imports import *
from mantisshrimp.core import *
from mantisshrimp.models.mantis_rcnn.rcnn_param_groups import *
from mantisshrimp.models.mantis_rcnn.mantis_rcnn import *
from mantisshrimp.models.mantis_rcnn.mantis_faster_rcnn import *
from mantisshrimp.backbones import *


class MantisMaskRCNN(MantisRCNN):
    @delegates(MaskRCNN.__init__)
    def __init__(
        self,
        num_classes: int,
        backbone: nn.Module = None,
        param_groups: List[nn.Module] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes

        if backbone is None:
            # Creates the default fasterrcnn as given in pytorch. Trained on COCO dataset
            self.model = maskrcnn_resnet50_fpn(pretrained=True, **kwargs)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes
            )
            in_features_mask = (
                self.model.roi_heads.mask_predictor.conv5_mask.in_channels
            )
            self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
                in_channels=in_features_mask, dim_reduced=256, num_classes=num_classes
            )
            param_groups = resnet_fpn_backbone_param_groups(self.model.backbone)
        else:
            self.model = MaskRCNN(backbone, num_classes=num_classes, **kwargs)
            param_groups = param_groups or [backbone]

        self._param_groups = param_groups + [self.model.rpn, self.model.roi_heads]
        check_all_model_params_in_groups(self.model, self.param_groups)

    def forward(self, images, targets=None):
        return self.model(images, targets)

    @property
    def param_groups(self):
        return self._param_groups

    @staticmethod
    def build_training_sample(
        imageid: int,
        img: np.ndarray,
        label: List[int],
        bbox: List[BBox],
        mask: MaskArray,
        **kwargs,
    ):
        x, y = MantisFasterRCNN.build_training_sample(
            imageid=imageid, img=img, label=label, bbox=bbox,
        )
        y["masks"] = tensor(mask.data, dtype=torch.uint8)
        return x, y
