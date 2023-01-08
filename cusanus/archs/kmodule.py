import math
import torch
from torch import nn
from functorch import vmap
from functools import partial

from cusanus.pytypes import *
from cusanus.archs import ImplicitNeuralModule, SirenNet


def qspline(a: Tensor, b:Tensor, c: Tensor, t: Tensor):
        return a * (t**2) + b * t + c


def eval_spline_lpdf(loc:Tensor, sigma:Tensor, value:Tensor):
    var = (sigma ** 2)
    log_scale = torch.log(sigma)
    return -((value - loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

def _fc_layers(indim:int, outdim:int, hidden:int, depth:int):
        layers = []
        for l in range(depth-1):
            layer = nn.Sequential(
                nn.Linear(indim if l == 0 else hidden,
                          hidden),
                nn.ReLU())
            layers.append(layer)
        layers.append(nn.Linear(hidden, outdim))
        return nn.Sequential(*layers)

class QSplineModule(nn.Module):

    def __init__(self,
                 kdim:int,
                 hidden:int,
                 qdepth:int,
                 siren_params:dict):

        super().__init__()
        # Hidden layers
        # layers = []
        # for l in range(depth):
        #     layer = nn.Sequential(
        #         nn.Linear(kdim if l == 0 else hidden,
        #                   hidden),
        #         nn.ReLU())
        #     layers.append(layer)
        # self.layers = nn.Sequential(*layers)
        self.mod = kdim
        self.layers = SirenNet(theta_in = kdim,
                               theta_hidden = hidden,
                               theta_out = hidden,
                               final_activation = nn.ReLU,
                               **siren_params)

        # output ABCs for qspline
        self.X = _fc_layers(hidden, 3, hidden, qdepth)
        self.Y = _fc_layers(hidden, 3, hidden, qdepth)
        self.Z = _fc_layers(hidden, 3, hidden, qdepth)

    def forward(self, kmod):
        hidden = self.layers(kmod)
        xa,xb,xc = self.X(hidden)
        ya,yb,yc = self.Y(hidden)
        za,zb,zc = self.Z(hidden)
        xt = partial(qspline, xa,xb,xc)
        yt = partial(qspline, ya,yb,yc)
        zt = partial(qspline, za,zb,zc)
        return (xt,yt,zt)

class PQSplineModule(nn.Module):

    def __init__(self,
                 kdim:int,
                 hidden:int,
                 qspline_params:dict,
                 sigma_params:dict):

        super().__init__()
        # qspline for mean
        self.qspline = QSplineModule(kdim = kdim,
                                     hidden = hidden,
                                     **qspline_params)

        # variance INR
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
        sigma = self.sigma(t, mod)
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        ys = eps * std + loc
        # lpdfs = vmap(eval_spline_lpdf)(loc, sigma, xyz)
        # lpdfs = eval_spline_lpdf(loc, sigma, xyz)
        # REVIEW: could also return spline partials
        return ys, loc, std
