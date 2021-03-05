import pytest
from icevision.all import *


def test_confusion_matrix(records, preds):
    confusion_matrix = SimpleConfusionMatrix()
