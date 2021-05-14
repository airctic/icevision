__all__ = ["MMDetTimmBase"]

import torch.nn as nn


class MMDetTimmBase(nn.Module):
    def __init__(self, pretrained=True, out_indices=(0, 1, 2, 3, 4), **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.out_indices = out_indices

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):  # should return a tuple
        return self.model(x)
