# Several chunks cribbed from
# https://github.com/lucidrains/siren-pytorch/blob/master/siren_pytorch/siren_pytorch.py
import math
import torch
from torch import nn
import torch.nn.functional as F
# from einops import rearrange
# from siren_pytorch import SirenNet, SirenWrapper

from cusanus.pytypes import *

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class Siren(nn.Module):
    def __init__(self, dim_in:int, dim_out:int,
                 w0:float = 1., w_std:float = 1.,
                 bias = True, activation = True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias = bias)
        nn.init.uniform_(self.linear.weight, a = -w_std, b = w_std)
        nn.init.uniform_(self.linear.bias, a = -w_std, b = w_std)
        self.activation = Sine(w0) if activation else nn.Identity()

    def forward(self, x:Tensor):
        x =  self.linear(x)
        return self.activation(x)


class SirenNet(nn.Module):
    def __init__(self,
                 theta_in:int,
                 theta_hidden:int,
                 theta_out:int,
                 depth:int,
                 w0 = 1.0,
                 w0_initial = 5.0,
                 c = 6.0,
                 use_bias = True,
                 final_activation = nn.Sigmoid):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([])
        w_std = math.sqrt(c / theta_hidden) / w0
        for l in range(depth - 1):
            layer = Siren(
                dim_in = theta_in if l == 0 else theta_hidden,
                dim_out = theta_hidden,
                w0 = w0_initial if l == 0 else w0,
                w_std = w_std, # 1.0 / theta_in if l == 0 else w_std,
                bias = use_bias,
                activation = True,
            )
            self.layers.append(layer)
        self.last_layer = nn.Sequential(Siren(dim_in = theta_hidden,
                                              dim_out = theta_out, w0 = w0,
                                              bias = use_bias, activation = False),
                                        final_activation())

    def forward(self, x:Tensor, phi:Tensor):
        for l in range(self.depth - 1):
            x = self.layers[l](x) + phi[l]
        x = self.last_layer(x)
        return x

class LatentModulation(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.latent_code = nn.Parameter(data = torch.zeros(dim))

    def forward(self):
        return self.latent_code

class Modulator(nn.Module):
    def __init__(self, dim_in: int, dim_hidden:int, depth: int):
        super().__init__()
        self.depth = depth
        self.hidden = dim_hidden
        self.layers = nn.ModuleList([])
        for ind in range(depth):
            is_first = ind == 0
            # for skip connection
            dim = dim_in if is_first else (dim_hidden + dim_in)
            layer = nn.Sequential(nn.Linear(dim, dim_hidden),
                                  nn.ReLU())
            self.layers.append(layer)

    def forward(self, m):
        x = m # set as first input
        hiddens = x.new_empty((self.depth, self.hidden))
        for l in range(self.depth - 1):
            # pass through next layer in modulator
            x = self.layers[l](x)
            # save layer output
            hiddens[l] = x
            # concat with latent code for next step
            x = torch.cat((x, m), dim = -1)
        #
        hiddens[-1] = self.layers[-1](x)
        return hiddens

class ImplicitNeuralModule(nn.Module):
    """ Implicit Neural Module

    Arguments:
        q_in: int, query dimension
        out: int, output dimensions
        hidden: int = 256, hidden dimensions (for theta and psi)
        depth: int = 5, depth of theta, modulator is `depth` - 1
    """

    def __init__(self,
                 q_in: int = 1,
                 out: int = 1,
                 hidden: int = 256,
                 mod: int = 24,
                 depth: int = 5) -> None:
        super().__init__()
        self.hidden = hidden
        self.mod = mod
        # Siren Network - weights refered to as `theta`
        # optimized during outer loop
        self.theta = SirenNet(q_in, hidden, out, depth)
        # Modulation FC network - refered to as psi
        # psi is initialize with default weights
        # and is not optimized
        self.psi = Modulator(mod, hidden, depth - 1)

    def forward(self, qs:Tensor, m) -> Tensor:
        phi = self.psi(m) # shift modulations
        return self.theta(qs, phi)
