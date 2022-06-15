from icevision.models.torchvision.backbone_config import TorchvisionBackboneConfig


class TorchvisionKeypointRCNNBackboneConfig(TorchvisionBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="keypoint_rcnn", **kwargs)
