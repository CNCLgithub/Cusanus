import math
import torch
from torch import nn
from functorch import vmap
from functools import partial

from cusanus.pytypes import *
from cusanus.archs import ImplicitNeuralModule


def qspline(a: Tensor, b:Tensor, c: Tensor, t: Tensor):
        return a * (t**2) + b * t + c


def eval_spline_lpdf(loc:Tensor, sigma:Tensor, value:Tensor):
    var = (sigma ** 2)
    log_scale = torch.log(sigma)
    return -((value - loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

class QSplineModule(nn.Module):

    def __init__(self,
                 kdim:int=16,
                 hidden:int = 64,
                 depth:int = 3):

        super().__init__()
        # Hidden layers
        layers = []
        for l in range(depth):
            layer = nn.Sequential(
                nn.Linear(kdim if l == 0 else hidden,
                          hidden),
                nn.ReLU())
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

        # output ABCs for qspline
        self.X = nn.Linear(hidden, 3)
        self.Y = nn.Linear(hidden, 3)
        self.Z = nn.Linear(hidden, 3)

    def forward(self, kmod):
        hidden = self.layers(kmod)
        xa,xb,xc = self.X(hidden)
        ya,yb,yc = self.Y(hidden)
        za,zb,zc = self.Z(hidden)
        yt = partial(qspline, ya,yb,yc)
        xt = partial(qspline, xa,xb,xc)
        zt = partial(qspline, za,zb,zc)
        return (xt,yt,zt)

class PQSplineModule(nn.Module):

    def __init__(self,
                 kdim:int,
                 qspline_params:dict,
                 sigma_params:dict):

        super().__init__()
        # qspline for mean
        self.qspline = QSplineModule(kdim = kdim,
                                     **qspline_params)

        # variance INR
        # TODO: ensure non-zero output
        self.sigma = ImplicitNeuralModule(q_in = 1,
                                          out = 3,
                                          mod = kdim,
                                          sigmoid = False,
                                          **sigma_params)

    def forward(self, qs:Tensor, mod):
        # b x 1
        t = qs[:, 0].unsqueeze(1)
        # b x 3
        xyz = qs[:, 1:]

        xt, yt, zt = self.qspline(mod)
        # b x 3
        loc = torch.cat([xt(t), yt(t), zt(t)], axis = 1)
        # b x 3
        sigma = self.sigma(t, mod) + 0.01

        # lpdfs = vmap(eval_spline_lpdf)(loc, sigma, xyz)
        lpdfs = eval_spline_lpdf(loc, sigma, xyz)

        # print(t.shape)
        # print(loc)
        # print(loc.shape)
        # print(sigma)
        # print(sigma.shape)
        # print(lpdfs)
        # raise ValueError()
        # REVIEW: could also return spline partials
        return lpdfs, loc, sigma
