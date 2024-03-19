import torch
from torch import nn
from functorch import vmap

from cusanus.pytypes import *
from cusanus.archs import ImplicitNeuralModule

class KModule(nn.Module):

    def __init__(self,
                 mdim:int,
                 pdim:int,
                 pf_params:dict,
                 mf_params:dict):

        super().__init__()
        self.mod = mdim
        self.pos_field = ImplicitNeuralModule(q_in = 2,
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
        x = qs[:, 1:] # b x 2
        # pmod <- motion_field(t | motion_code)
        pmods = self.motion_field(t, m) # b x pdim
        ys = vmap(self.pos_field)(x, pmods) # b x 1
        ys = self.act(ys)
        return ys

# TODO: move to own file
class EModule(nn.Module):

    def __init__(self,
                 mdim:int,
                 pdim:int,
                 edim:int,
                 inr_params:dict):

        super().__init__()
        self.mod = edim
        self.inr = ImplicitNeuralModule(q_in = mdim + pdim,
                                        out = mdim,
                                        mod = edim,
                                        # Identity act
                                        sigmoid = False,
                                        **inr_params)

    def forward(self, k0:Tensor, m:Tensor):
        k1 = self.inr(k0, m)
        return k1
