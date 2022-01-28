from icevision.models.mmdet.utils import *


class MMDetRetinanetBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="retinanet", **kwargs)
