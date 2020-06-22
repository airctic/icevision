import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "detr"))

from mantisshrimp import *
from mantisshrimp.hub.detr.detr_pretrained_checkpoint_base import *
from mantisshrimp.hub.detr.detr_dataset import *
from mantisshrimp.hub.detr.detr_parser import *
from mantisshrimp.hub.detr.detr.datasets.coco import (
    make_coco_transforms as detr_transform,
)
from mantisshrimp.hub.detr.detr.main import get_args_parser as detr_args_parser
from mantisshrimp.hub.detr.detr.main import main as run_detr
from mantisshrimp.hub.detr.detr.util import plot_utils as detr_plot_utils
