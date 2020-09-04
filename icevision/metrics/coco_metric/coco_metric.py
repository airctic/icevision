__all__ = ["COCOMetric", "COCOMetricType"]

from icevision.imports import *
from icevision.utils import *
from icevision.data import *
from icevision.metrics.metric import *


class COCOMetricType(Enum):
    """Available options for `COCOMetric`."""

    bbox = "bbox"
    mask = "segm"
    keypoint = "keypoints"


class COCOMetric(Metric):
    """Wrapper around [cocoapi evaluator](https://github.com/cocodataset/cocoapi)

    Calculates average precision.

    # Arguments
        metric_type: Dependent on the task you're solving.
        print_summary: If `True`, prints a table with statistics.
    """

    def __init__(
        self,
        metric_type: COCOMetricType = COCOMetricType.bbox,
        print_summary: bool = False,
    ):
        self.metric_type = metric_type
        self.print_summary = print_summary
        self._records, self._preds = [], []

    def _reset(self):
        self._records.clear()
        self._preds.clear()

    def accumulate(self, records, preds):
        self._records.extend(records)
        self._preds.extend(preds)

    def finalize(self) -> Dict[str, float]:
        with CaptureStdout():
            coco_eval = create_coco_eval(
                records=self._records,
                preds=self._preds,
                metric_type=self.metric_type.value,
            )
            coco_eval.evaluate()
            coco_eval.accumulate()

        with CaptureStdout(propagate_stdout=self.print_summary):
            coco_eval.summarize()
        # TODO: all results
        mAP = coco_eval.stats[0]
        logs = {"mAP": mAP}

        self._reset()
        return logs
