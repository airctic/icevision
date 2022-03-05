from icevision.models.torchvision.backbone_config import TorchvisionBackboneConfigV2


class RetinanetV2TorchVisionBackboneConfig(TorchvisionBackboneConfigV2):
    def __init__(self, backbone, **kwargs):
        self.backbone = backbone
        super().__init__(model_name="retinanet", **kwargs)
