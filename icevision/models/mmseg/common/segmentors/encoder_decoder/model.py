__all__ = ["model"]

from icevision.imports import *
from icevision.models.mmseg.utils import *
from icevision.utils.download_utils import *
from icevision.models.mmseg.common.utils import *


def model(
    backbone: MMSegBackboneConfig,
    num_classes: int,
    checkpoints_path: Optional[Union[str, Path]] = "checkpoints",
    force_download=False,
    cfg_options=None,
    pre_training_dataset: str = None,
    lr_schd: int = None,
    crop_size: tuple = None,
) -> nn.Module:

    return build_model(
        model_type="EncoderDecoder",
        backbone=backbone,
        num_classes=num_classes,
        pretrained=backbone.pretrained,
        checkpoints_path=checkpoints_path,
        force_download=force_download,
        cfg_options=cfg_options,
        pre_training_dataset=pre_training_dataset,
        lr_schd=lr_schd,
        crop_size=crop_size,
    )
