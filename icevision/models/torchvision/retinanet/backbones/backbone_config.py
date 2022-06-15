from icevision.models.torchvision.backbone_config import TorchvisionBackboneConfig


class TorchvisionRetinanetBackboneConfig(TorchvisionBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="retinanet", **kwargs)
