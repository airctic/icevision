__all__ = ["MantisRCNN"]

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN

from mantisshrimp.imports import *
from mantisshrimp.utils import *
from mantisshrimp.models.mantis_module import *
from mantisshrimp.backbones import *


class MantisRCNN(MantisModule, ABC):
    def __init__(self, metrics=None):
        super().__init__()
        self.metrics = metrics or []

    @staticmethod
    @abstractmethod
    def build_training_sample(*args, **kwargs):
        """
        Converts a record to a format understood by the model.
        """

    def _predict(
        self, images: List[np.ndarray], convert_raw_prediction,
    ):
        self.eval()
        tensor_images = [im2tensor(img).to(self.device) for img in images]
        raw_preds = self(tensor_images)

        return [convert_raw_prediction(raw_pred) for raw_pred in raw_preds]

    def _remove_transforms_from_model(self, model: GeneralizedRCNN):
        def noop_normalize(image):
            return image

        def noop_resize(image, target):
            return image, target

        model.transform.normalize = noop_normalize
        model.transform.resize = noop_resize

    @staticmethod
    def loss(preds, targs) -> Tensor:
        return sum(preds.values())

    @classmethod
    def collate_fn(cls, data):
        ts = [cls.build_training_sample(**o) for o in data]
        xb, yb = zip(*ts)
        return xb, list(yb)

    @classmethod
    def dataloader(cls, dataset, **kwargs) -> DataLoader:
        return DataLoader(dataset=dataset, collate_fn=cls.collate_fn, **kwargs)

    @staticmethod
    def get_backbone_by_name(
        name: str, fpn: bool = True, pretrained: bool = True
    ) -> nn.Module:
        """
        Args:
            backbone (str): If none creates a default resnet50_fpn model trained on MS COCO 2017
                Supported backones are: "resnet18", "resnet34","resnet50", "resnet101", "resnet152",
                 "resnext50_32x4d", "resnext101_32x8d", "wide_resnet50_2", "wide_resnet101_2", as resnets with fpn backbones.
                Without fpn backbones supported are: "resnet18", "resnet34", "resnet50","resnet101",
                 "resnet152", "resnext101_32x8d", "mobilenet", "vgg11", "vgg13", "vgg16", "vgg19",
            pretrained (bool): Creates a pretrained backbone with imagenet weights.
        """
        # Giving string as a backbone, which is either supported resnet or backbone
        if fpn:
            # Creates a torchvision resnet model with fpn added
            # It returns BackboneWithFPN model
            backbone = resnet_fpn_backbone(name, pretrained=pretrained)
        else:
            # This does not create fpn backbone, it is supported for all models
            raise NotImplementedError
            # backbone = create_torchvision_backbone(name, pretrained=pretrained)
        return backbone
