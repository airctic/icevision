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
        self, num_classes: int, backbone: nn.Module = None, metrics=None, **kwargs,
    ):
        super().__init__(metrics=metrics)
        self.num_classes = num_classes
        self.backbone = backbone
        # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # # get the number of input features for the classifier
        # in_features = model.roi_heads.box_predictor.cls_score.in_features
        # # replace the pre-trained head with a new one
        # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # # now get the number of input features for the mask classifier
        # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        # hidden_layer = 256
        # # and replace the mask predictor with a new one
        # model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
        #                                                    hidden_layer,
        #                                                    num_classes)

        if backbone is None:
            # Creates the default fasterrcnn as given in pytorch. Trained on COCO dataset
            self.m = maskrcnn_resnet50_fpn(
                pretrained=True, num_classes=num_classes, **kwargs,
            )
            in_features = self.m.roi_heads.box_predictor.cls_score.in_features

            self.m.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            in_features_mask = self.m.roi_heads.mask_predictor.conv5_mask.in_channels
            self.m.roi_heads.mask_predictor = MaskRCNNPredictor(
                in_features_mask, num_classes
            )

        else:
            self.m = MaskRCNN(backbone, num_classes=num_classes, **kwargs)

    def forward(self, images, targets=None):
        return self.m(images, targets)

    def model_splits(self):
        return split_rcnn_model(self.m)

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
