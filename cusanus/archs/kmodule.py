import torch
from torch import nn
from functools import partial

from cusanus.pytypes import *
from cusanus.archs import ImplicitNeuralModule, SirenNet


def qspline(a: Tensor, b:Tensor, c: Tensor, t: Tensor):
    return (a * (t**2)) + (b * t) + c


class QSplineModule(nn.Module):

    def __init__(self,
                 kdim:int,
                 **kwargs):

        super().__init__()
        self.mod = kdim
        self.abc = SirenNet(theta_in = kdim,
                            theta_out = 9,
                            final_activation = nn.Identity,
                            **kwargs)


    def forward(self, m:Tensor):
        a,b,c = self.abc(m).reshape(3,3)
        xa, ya, za = a
        xb, yb, zb = b
        xc, yc, zc = c
        xt = partial(qspline, xa,xb,xc)
        yt = partial(qspline, ya,yb,yc)
        zt = partial(qspline, za,zb,zc)
        return (xt,yt,zt)

class PQSplineModule(nn.Module):

    def __init__(self,
                 kdim:int,
                 qspline_params:dict,
                 sigma_params:dict):

        super().__init__()
        self.mod = kdim
        # qspline for mean
        # self.qspline = QSplineModule(kdim = kdim,
        #                              **qspline_params)
        self.mu = ImplicitNeuralModule(q_in = 1,
                                       out = 3,
                                       mod = kdim,
                                       # Identity act
                                       sigmoid = False,
                                       **sigma_params)

        # variance INR
        self.sigma = ImplicitNeuralModule(q_in = 1,
                                          out = 3,
                                          mod = kdim,
                                          # Identity act
                                          sigmoid = False,
                                          **sigma_params)
    def forward(self, qs:Tensor, m:Tensor):
        # b x 3
        loc = self.mu(qs, m)
        # b x 3
        sigma = self.sigma(qs, m)
        std = torch.exp(0.5 * sigma)
        # eps = torch.randn_like(std)
        # ys = eps * std + loc
        # REVIEW: could also return spline partials
        return loc, loc, std

    # def forward(self, qs:Tensor, m:Tensor):
    #     # b x 1
    #     xt, yt, zt = self.qspline(m)
    #     # b x 3
    #     loc = torch.cat([xt(qs), yt(qs), zt(qs)],
    #                     axis = 1)
    #     # b x 3
    #     sigma = self.sigma(qs, m)
    #     std = torch.exp(0.5 * sigma)
    #     # eps = torch.randn_like(std)
    #     # ys = eps * std + loc
    #     # REVIEW: could also return spline partials
    #     return loc, loc, std
