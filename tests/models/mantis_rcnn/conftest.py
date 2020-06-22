import pytest
from mantisshrimp.imports import nn


@pytest.fixture(scope="module")
def simple_backbone():
    class SimpleBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, 2)
            self.c2 = nn.Conv2d(8, 16, 3, 2)
            self.out_channels = 16

        def forward(self, x):
            return self.c2(self.c1(x))

    return SimpleBackbone()
