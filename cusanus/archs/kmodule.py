import torch
from torch import nn

from functools import partial
from functorch import vmap

from cusanus.pytypes import *



def qspline(a: Tensor, b:Tensor, c: Tensor, t: Tensor):
        return a * (t**2) + b * t + c


def eval_spline_lpdf(loc:Tensor, sigma:Tensor, ys:Tensor):
    mv = MultivariateNormal(loc, sigma)
    return mv.log_pdf(ys)

class QSplineModule(nn.Module):

    def __init__(self,
                 kdim:int=16,
                 hidden:int = 64,
                 depth:int = 3):

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
        hidden = self.layers(mod)

        xa,xb,xc = self.X(hidden).T
        ya,yb,yc = self.Y(hidden).T
        za,zb,zc = self.Z(hidden).T
        yt = partial(qspline, ya,yb,yc)
        xt = partial(qspline, xa,xb,xc)
        zt = partial(qspline, za,zb,zc)
        return (xt,yt,zt)

class PQSplineModule(nn.Module):

    def __init__(self,
                 kdim:int=16,
                 hidden:int = 64,
                 depth:int = 3):

        # qspline for mean
        self.qspline = QSplineModule(**qspline_params)

        # variance INR
        self.sigma = ImplicitNeuralModule(q_in = 4,
                                          out = 9,
                                          mod = kdim,
                                          sigmoid = False,
                                          **sigma_params)

    def forward(self, qs:Tensor, mod):

        # b x 1
        t = qs[:, 0].unsqueeze(1)
        # b x 3
        xys = qs[:, 1:]

        xt, yt, zt = self.qspline(mod)
        # b x 3
        loc = torch.cat([xt(t), yt(t), zt(t)], axis = 1)
        # b x 3 x 3
        sigma = self.sigma(t, mod).reshape((-1, 3,3))

        lpdfs = vmap(eval_spline_lpdf)(loc, sigma, xyz)

        # REVIEW: could also return spline partials
        return -lpdfs
