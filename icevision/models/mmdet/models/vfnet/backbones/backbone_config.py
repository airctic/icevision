from icevision.models.mmdet.utils import *


class MMDetVFNETBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="vfnet", **kwargs)
