from icevision.models.mmseg.models.deeplabv3plus import backbones
from icevision.models.interpretation import Interpretation, _move_to_device
from icevision.models.mmseg.common.dataloaders import *
from icevision.models.mmseg.common.prediction import *
from icevision.models.mmseg.common.show_results import *
from icevision.models.mmseg.common.show_batch import *

from icevision.models.mmseg.common.interpretation_utils import (
    sum_losses_mmseg,
    loop_mmseg,
)

from icevision.models.mmseg.common.segmentors.encoder_decoder import *


_LOSSES_DICT = {
    "decode.loss_seg": [],
    "aux.loss_seg": [],
    "loss_total": [],
}

interp = Interpretation(
    losses_dict=_LOSSES_DICT,
    valid_dl=valid_dl,
    infer_dl=infer_dl,
    predict_from_dl=predict_from_dl,
)

interp._loop = loop_mmseg
