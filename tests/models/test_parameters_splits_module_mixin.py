import pytest
from mantisshrimp import *
from mantisshrimp.imports import *


@pytest.fixture
def model():
    class SimpleModel(MantisModule):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(8, 6)
            self.l2 = nn.Linear(6, 6)
            self.l3 = nn.Linear(6, 2)

        def forward(self, x):
            return self.l3(self.l2(self.l1(x)))

        def dataloader(cls, **kwargs) -> DataLoader:
            return DataLoader(**kwargs)

        def load_state_dict(
            self, state_dict: Dict[str, Tensor], strict: bool = True,
        ):
            pass

    return SimpleModel()


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
