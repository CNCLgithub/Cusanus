import torch
from abc import ABC
from torch import nn
from ensembles.pytypes import *
from ensembles.flows import InvertibleModule

class Transform(InvertibleModule):
    def __init__(self):
        super().__init__()

class ShiftTransform(Transform):

    def forward(self, ld: Tensor, x: Tensor, sigma: Tensor) -> Tensor :
        y = x + sigma
        return y

    def inverse(self, ld: Tensor, y: Tensor, sigma: Tensor) -> Tensor :
        x = y - sigma
        return x

class ExpAffine(Transform):

    def forward(self, ld: Tensor, x: Tensor, sigma: Tensor) -> Tensor :
        shift, scale = torch.chunk(sigma, 2)
        y = x * torch.exp(scale) + shift
        ld += torch.sum(scale, dim=list(range(1, shift.dim())))
        return y

    def inverse(self, ld: Tensor, y: Tensor, sigma: Tensor) -> Tensor:
        shift, scale = torch.chunk(sigma, 2)
        x = (y - shift) * torch.exp(-scale)
        ld += -torch.sum(scale, dim=list(range(1, shift.dim())))
        return x
