from icevision.models.torchvision.backbone_config import TorchvisionBackboneConfig


class TorchvisionMaskRCNNBackboneConfig(TorchvisionBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="mask_rcnn", **kwargs)
