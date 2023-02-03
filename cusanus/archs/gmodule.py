import torch
from torch import nn
from functorch import vmap

from cusanus.pytypes import *
from cusanus.archs import ImplicitNeuralModule, Sine, SirenNet

class GModule(nn.Module):

    def __init__(self,
                 sdim:int,
                 odim:int,
                 sf_params:dict,
                 of_params:dict):

        super().__init__()
        self.sdim = sdim
        self.odim = odim
        self.mod = sdim + odim
        # self.scale_field = ImplicitNeuralModule(q_in = 2,
        #                                         out = 2,
        #                                         mod = sdim,
        #                                         sigmoid = False,
        #                                         **sf_params)
        self.scale_field = SirenNet(theta_in = sdim,
                                    theta_out = 2,
                                    final_activation=nn.Softplus,
                                    **sf_params)
        # self.scale_act = Sine()
        # self.scale_act = nn.Softplus()
        self.occ_field = ImplicitNeuralModule(q_in = 2,
                                              out = 1,
                                              mod = odim,
                                              # Identity act
                                              sigmoid = True,
                                              **of_params)

    # def scale_qs(self, qs:Tensor, sm:Tensor):
    #     s = self.scale_field(qs, sm)
    #     s = self.scale_act(s)
    #     return qs * s
    def scale_qs(self, qs:Tensor, sm:Tensor):
        s = self.scale_field(sm)
        return qs * s

    def forward(self, qs:Tensor, m:Tensor):
        sm = m[:self.sdim]
        om = m[self.sdim:]
        qs = self.scale_qs(qs, sm)
        # query occupacy field
        ys = self.occ_field(qs, om)
        return ys
