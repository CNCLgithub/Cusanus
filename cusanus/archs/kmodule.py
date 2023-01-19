import torch
from torch import nn
from functools import partial
from functorch import vmap

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
        assert kdim % 2 == 0, 'kdim {kdim} not even'
        self.mod = kdim
        mdim = int(kdim / 2)
        # qspline for mean
        # self.qspline = QSplineModule(kdim = mdim,
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
                                          mod = mdim,
                                          # Identity act
                                          sigmoid = False,
                                          **sigma_params)
    def forward(self, qs:Tensor, m:Tensor):
        m1,m2 = torch.chunk(m, 2)
        # b x 3
        # xt, yt, zt = self.qspline(m1)
        # # b x 3
        # loc = torch.cat([xt(qs), yt(qs), zt(qs)],
        #                 axis = 1)
        loc = self.mu(qs, m)
        # b x 3
        sigma = self.sigma(qs, m2)
        std = torch.exp(0.5 * sigma)
        # eps = torch.randn_like(std)
        # ys = eps * std + loc
        ys = torch.cat([loc, std], axis = 1)
        # REVIEW: could also return spline partials
        return ys, loc, std

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

class KModule(nn.Module):

    def __init__(self,
                 mdim:int,
                 pdim:int,
                 pf_params:dict,
                 mf_params:dict):

        super().__init__()
        self.mod = mdim
        self.pos_field = ImplicitNeuralModule(q_in = 3,
                                              out = 1,
                                              mod = pdim,
                                              sigmoid = False,
                                              **pf_params)
        self.motion_field = ImplicitNeuralModule(q_in = 1,
                                                 out = pdim,
                                                 mod = mdim,
                                                 # Identity act
                                                 sigmoid = False,
                                                 **mf_params)
        self.act = nn.Softplus()

    def forward(self, qs:Tensor, m:Tensor):
        b, _ = qs.shape
        t = qs[:, 0].unsqueeze(1)    # b x 1
        xyz = qs[:, 1:] # b x 3
        # pmod <- motion_field(t | motion_code)
        pmods = self.motion_field(t, m) # b x pdim
        ys = vmap(self.pos_field)(xyz, pmods) # b x 1
        ys = self.act(ys)
        return ys

class EModule(nn.Module):

    def __init__(self,
                 mdim:int,
                 pdim:int,
                 edim:int,
                 inr_params:dict):

        super().__init__()
        self.mod = mdim + pdim
        self.inr = ImplicitNeuralModule(q_in = mdim + pdim,
                                        out = mdim,
                                        mod = edim,
                                        # Identity act
                                        sigmoid = False,
                                        **inr_params)

    def forward(self, k0:Tensor, m:Tensor):
        return self.inr(k0, m)
