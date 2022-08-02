from icevision.models.torchvision.backbone_config import TorchvisionBackboneConfig


class TorchvisionFasterRCNNBackboneConfig(TorchvisionBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="faster_rcnn", **kwargs)
