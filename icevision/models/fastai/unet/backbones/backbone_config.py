from icevision.models.torchvision.backbone_config import TorchvisionBackboneConfig


class TorchvisionUNetBackboneConfig(TorchvisionBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="unet", **kwargs)
