import pytest
from icevision.all import *
from .test_coco_metric import records, preds


def test_confusion_matrix(records, preds):
    confusion_matrix = SimpleConfusionMatrix()
    confusion_matrix.accumulate(records, preds)
    dummy_result = confusion_matrix.finalize()
    cm_result = confusion_matrix.confusion_matrix
    expected_result = np.diagflat([0, 0, 2, 1])
    assert dummy_result["dummy_value_for_fastai"] == -1
    assert np.equal(cm_result, expected_result).all()
