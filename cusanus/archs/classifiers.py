import torch
from torch import nn
import torch.nn.functional as F
from ensembles.pytypes import *

class FCClassifier(nn.Module):
    def __init__(self, e_size:int = 128, n:int = 10):
        """
        Inputs:
            e_size - Size of granularity embedding
            n - number of granularity levels
        """
        super().__init__()
        self.inner_layers = Sequential(
            Linear(e_size, 128),
            TanH(),
            Linear(128, 64),
            TanH()
        )
        self.softmax_layer = Sequential(
            Linear(64, n),
        )

    def inner(self, x: Tensor) -> Tensor:
        return self.inner_layers(x)

    def forward(self, x: Tensor) -> Tensor:
        x = self.inner(x)
        return self.softmax_layer(x)
