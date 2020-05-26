import pytest
from mantisshrimp import *


@pytest.fixture
def model():
    return test_utils.SimpleModel()


def test_params_splits(model):
    assert len(list(model.params_splits())) == 3


def test_freeze_to(model):
    model.freeze_to(n=-2)
    assert len(list(model.trainable_params_splits())) == 2
    assert len(list(filter_params(model, only_trainable=True))) == 4
    model.freeze_to(n=0)
    assert len(list(model.trainable_params_splits())) == 3
    assert len(list(filter_params(model, only_trainable=True))) == 6
    model.freeze_to(n=None)
    assert len(list(model.trainable_params_splits())) == 0
    assert len(list(filter_params(model, only_trainable=True))) == 0
