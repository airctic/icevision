__all__ = ["SimpleModel"]

from mantisshrimp.all import *


class SimpleModel(MantisModule):
    def __init__(self):
        super().__init__()
        self.ls = [nn.Linear(8, 6), nn.Linear(6, 6), nn.Linear(6, 2)]
        self.model = nn.Sequential(*self.ls)

    def model_splits(self):
        return self.ls

    def forward(self, x):
        return self.model(x)

    def dataloader(cls, **kwargs) -> DataLoader:
        return DataLoader(**kwargs)
