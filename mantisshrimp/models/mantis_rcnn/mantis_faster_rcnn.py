__all__ = ["MantisFasterRCNN"]

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from mantisshrimp.imports import *
from mantisshrimp.core import *
from mantisshrimp.models.mantis_rcnn.rcnn_param_groups import *
from mantisshrimp.models.mantis_rcnn.mantis_rcnn import *


class MantisFasterRCNN(MantisRCNN):
    """
    Creates a flexible Faster RCNN implementation based on torchvision library.
    Args: 
    n_class (int) : number of classes. Do not have class_id "0" it is reserved as background.
                    n_class = number of classes to label + 1 for background.
    """

    @delegates(FasterRCNN.__init__)
    def __init__(
        self,
        num_classes: int,
        backbone: nn.Module = None,
        param_groups: List[nn.Module] = None,
        metrics=None,
        remove_internal_transforms=True,
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

        self._param_groups = param_groups + [self.model.rpn, self.model.roi_heads]
        check_all_model_params_in_groups(self.model, self._param_groups)

        if remove_internal_transforms:
            self._remove_transforms_from_model(self.model)

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def predict(self, images: List[np.ndarray], detection_threshold: float = 0.5):
        convert_raw_prediction = partial(
            self.convert_raw_prediction, detection_threshold=detection_threshold,
        )

        return self._predict(
            images=images, convert_raw_prediction=convert_raw_prediction
        )

    @property
    def param_groups(self):
        return self._param_groups

    @staticmethod
    def convert_raw_prediction(raw_pred: dict, detection_threshold: float):
        above_threshold = raw_pred["scores"] >= detection_threshold

        labels = raw_pred["labels"][above_threshold]
        labels = labels.detach().cpu().numpy()

        scores = raw_pred["scores"][above_threshold]
        scores = scores.detach().cpu().numpy()

        boxes = raw_pred["boxes"][above_threshold]
        bboxes = []
        for box_tensor in boxes:
            xyxy = box_tensor.cpu().tolist()
            bbox = BBox.from_xyxy(*xyxy)
            bboxes.append(bbox)

        return {
            "labels": labels,
            "scores": scores,
            "bboxes": bboxes,
            "above_threshold": above_threshold,
        }

    @staticmethod
    def build_training_sample(
        imageid: int, img: np.ndarray, labels: List[int], bboxes: List[BBox], **kwargs,
    ):
        x = im2tensor(img)
        # injected values when annotations are empty are disconsidered
        # because we mark label as 0 (background)
        _fake_box = [0, 1, 2, 3]
        y = {
            "image_id": tensor(imageid, dtype=torch.int64),
            "labels": tensor(labels or [0], dtype=torch.int64),
            "boxes": tensor([o.xyxy for o in bboxes] or [_fake_box], dtype=torch.float),
        }
        return x, y
