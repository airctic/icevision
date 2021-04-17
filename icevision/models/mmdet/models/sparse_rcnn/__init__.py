from icevision.models.mmdet.models.sparse_rcnn import backbones
from icevision.models.mmdet.common.bbox.two_stage import *
from icevision.models.interpretation import Interpretation, _move_to_device
from icevision.models.mmdet.common.interpretation_utils import (
    sum_losses_mmdet,
    loop_mmdet,
)

_LOSSES_DICT = {
    "loss_cls": [],
    "loss_bbox": [],
    "loss_total": [],
}

interp = Interpretation(
    losses_dict=_LOSSES_DICT,
    valid_dl=valid_dl,
    infer_dl=infer_dl,
    predict_from_dl=predict_from_dl,
)

interp._loop = loop_mmdet
