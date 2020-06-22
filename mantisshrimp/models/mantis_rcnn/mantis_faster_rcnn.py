__all__ = ["MantisFasterRCNN"]

from mantisshrimp.imports import *
from mantisshrimp.core import *
from mantisshrimp.models.mantis_rcnn.rcnn_param_groups import *
from mantisshrimp.models.mantis_rcnn.mantis_rcnn import *
from mantisshrimp.backbones import *


class MantisFasterRCNN(MantisRCNN):
    """
    Creates a flexible Faster RCNN implementation based on torchvision library.
    Args: 
    n_class (int) : number of classes. Do not have class_id "0" it is reserved as background. n_class = number of classes to label + 1 for background.
    """

    @delegates(FasterRCNN.__init__)
    def __init__(
        self,
        num_classes: int,
        backbone: nn.Module = None,
        param_groups: List[nn.Module] = None,
        metrics=None,
        **kwargs,
    ):
        super().__init__(metrics=metrics)
        self.num_classes = num_classes
        self.backbone = backbone
        if backbone is None:
            # Creates the default fasterrcnn as given in pytorch. Trained on COCO dataset
            self.model = fasterrcnn_resnet50_fpn(pretrained=True, **kwargs)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes
            )
            param_groups = resnet_fpn_backbone_param_groups(self.model.backbone)
        else:
            self.model = FasterRCNN(backbone, num_classes=num_classes, **kwargs)
            param_groups = param_groups or [backbone]

        self.param_groups = param_groups + [self.model.rpn, self.model.roi_heads]
        check_all_model_params_in_groups(self.model, self.param_groups)

    def forward(self, images, targets=None):
        return self.model(images, targets)

    # TODO: backwards compatability
    def model_splits(self):
        print("model_splits deprecated")
        return self.param_groups

    @staticmethod
    def build_training_sample(
        imageid: int, img: np.ndarray, label: List[int], bbox: List[BBox], **kwargs,
    ):
        x = im2tensor(img)
        # injected values when annotations are empty are disconsidered
        # because we mark label as 0 (background)
        _fake_box = [0, 1, 2, 3]
        y = {
            "image_id": tensor(imageid, dtype=torch.int64),
            "labels": tensor(label or [0], dtype=torch.int64),
            "boxes": tensor([o.xyxy for o in bbox] or [_fake_box], dtype=torch.float),
        }
        return x, y
