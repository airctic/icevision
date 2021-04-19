from icevision.models.mmdet.models.mask_rcnn import backbones
from icevision.models.mmdet.common.mask.two_stage import *
from icevision.models.interpretation import Interpretation, _move_to_device
from icevision.models.mmdet.common.interpretation_utils import (
    sum_losses_mmdet,
    loop_mmdet,
)

_LOSSES_DICT = {
    "loss_rpn_cls": [],
    "loss_rpn_bbox": [],
    "loss_cls": [],
    "loss_bbox": [],
    "loss_mask": [],
    "loss_total": [],
}

interp = Interpretation(
    losses_dict=_LOSSES_DICT,
    valid_dl=valid_dl,
    infer_dl=infer_dl,
    predict_from_dl=predict_from_dl,
)

interp._loop = loop_mmdet
