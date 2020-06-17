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
        self, n_class: int, backbone: nn.Module = None, metrics=None, **kwargs,
    ):
        super().__init__(metrics=metrics)
        self.n_class = n_class
        self.backbone = backbone
        if backbone is None:
            # Creates the default fasterrcnn as given in pytorch. Trained on COCO dataset
            self.m = fasterrcnn_resnet50_fpn(
                pretrained=False, num_classes=n_class, **kwargs,
            )
            in_features = self.m.roi_heads.box_predictor.cls_score.in_features
            self.m.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_class)
        else:
            self.m = FasterRCNN(backbone, num_classes=n_class, **kwargs)

    def forward(self, images, targets=None):
        return self.m(images, targets)

    def model_splits(self):
        return split_rcnn_model(self.m)

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
