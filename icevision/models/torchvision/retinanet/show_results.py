__all__ = ["show_results", "interp"]

from icevision.models.torchvision.faster_rcnn.show_results import *
from icevision.models.interpretation import Interpretation
from icevision.models.torchvision.retinanet.dataloaders import (
    valid_dl,
    infer_dl,
)
from icevision.models.torchvision.retinanet.prediction import (
    predict_from_dl,
)

_LOSSES_DICT = {
    "classification": [],
    "bbox_regression": [],
    "loss_total": [],
}

interp = Interpretation(
    losses_dict=_LOSSES_DICT,
    valid_dl=valid_dl,
    infer_dl=infer_dl,
    predict_from_dl=predict_from_dl,
)
