# inspired by https://github.com/VincentStimper/normalizing-flows/blob/3edff5e7398b3ca4d035da1f8f50612569d78a6a/normflow/flows/affine_coupling.py

import torch
from torch import nn
from ensembles.pytypes import *
from ensembles.flows import IvertibleModule, Transform



class PiecewiseCoupling(IvertibleModule):
    """
    TODO
    """

    def __init__(self, param_map: nn.Module, t: Transform):
        """
        Constructor
        TODO
        """
        super().__init__()
        self.add_module('param_map', param_map)
        self.add_module('transform', t)

    def forward(self, ld: Tensor, a: TTensor, b: TTensor) -> Tuple[TTensor, TTensor]:
        """
        """
        xa, ra = a
        xb, rb = b # xb -> yb
        sigma = self.param_map(xa)
        yb = self.transform(ld, xb, sigma)
        return ((xa, ra), (yb, rb))

    def inverse(self, ld: Tensor, a: TTensor, b: TTensor) -> Tuple[TTensor, TTensor]:
        xa, ra = a
        yb, rb = b # yb -> xb
        sigma = self.param_map(xa)
        xb = self.transform.inverse(ld, yb, sigma)
        return ((xa, ra), (xb, rb))

class PiecewiseJunction(PiecewiseCoupling):

    def forward(self, ld: Tensor, a: TTensor, b: TTensor, c: TTensor):
        xa, ra = a
        xb, rb = b # xb -> yb
        xc, rc = c
        sigma = self.param_map(torch.cat((xa, xc), dim = 1))
        yb = self.transform(ld, xb, sigma)
        return (xa, ra), (yb, rb), (xc, rc)

    def inverse(self, ld: Tensor, a: TTensor, b: TTensor, c: TTensor):
        xa, ra = a
        yb, rb = b # yb -> xb
        xc, rc = c
        sigma = self.param_map(torch.cat((xa, xc), dim = 1))
        xb = self.transform.inverse(ld, yb, sigma)
        return (xa, ra), (xb, rb), (xc, rc)

class PiecewiseBlock(InvertibleModule):
    """
    TODO
    """
    def __init__(self,
                 sigma: nn.Module,
                 c: Channel,
                 t: Transform):
        """
        Constructor
             1. a b -> xa ra xb rb
             2.     -> xa yb ra rb
             3.     -> a c
        TODO
        """
        super().__init__()
        # `Broadcast` caches the inverse of `c` so no need
        # to explicitly initialize split an merge modules
        self.steps = nn.ModuleList([Broadcast(c),
                                    PiecewiseCoupling(sigma, t),
                                    Broadcast(invert(c))])

    def forward(self, ld: Tensor, a: Tensor, b: Tensor):
        for step in self.steps:
            a, b = step(ld, a, b)
        return (a, b)

    def inverse(self, ld: Tesnor, a: Tensor, b: Tensor):
        for step in self.flows:
            a, b = step.inverse(ld, a, b)
        return (a, b)

class JunctionBlock(InvertibleModule):

    def __init__(self,
                 sigma: nn.Module,
                 ec: Channel,
                 kc: Channel,
                 t:  Transform):
        super().__init__()
        self.split_k = Broadcast(kc)
        self.split_e = ec
        # merges instantiated for performance
        self.merge_k = Broadcast(invert(kc))
        self.merge_e = invert(ec)
        self.junction = PiecewiseJunction(sigma, t)

    def _split(self, ld: Tensor, a: Tensor, b: Tensor, c: Tensor):
        (a, b) = self.split_k(ld, a, b)
        c = self.split_e(ld, c)
        return (a, b, c)

    def _merge(self, ld: Tensor, a: TTensor, b: TTensor, c: TTensor):
        (a, b) = self.merge_k(ld, a, b)
        c = self.merge_e(ld, c)
        return (a, b, c)

    def forward(self, ld: Tensor, a: Tensor, b: Tensor, c: Tensor):
        (a, b, c) = self._split(ld, a, b, c)
        (a, b, c) = self.junction(ld, a, b, c)
        (a, b, c) = self._merge(ld, a, b, c)
        return (a, b, c)

    def inverse(self, ld: Tensor, a: Tensor, b: Tensor, c: Tensor):
        (a, b, c) = self._split(ld, a, b, c)
        (a, b, c) = self.junction.inverse(ld, a, b, c)
        (a, b, c) = self._merge(ld, a, b, c)
        return (a, b, c)


class MergeBlock(InvertibleModule):

    def __init__(self,
                 esig: nn.Module,
                 emask: BoolTensor,
                 ksig: nn.Module,
                 kmask: BoolTensor):
        super().__init__()
        esplice = Splice(mask)
        at = ExpAffine()
        self.e_block = PiecewiseBlock(esig,
                                      esplice,
                                      at)
        self.k_block = JunctionBlock(ksig,
                                     esplice,
                                     Splice(kmask),
                                     at)

    def forward(self, ld: Tensor, a: Tensor, b: Tensor) -> TTensor:
        (a, c) = self.e_block(ld, a, b)
        (a, c) = self.k_block(ld, a, c, c)
        return (a, c)

    def inverse(self, ld: Tensor, a: Tensor, c: Tensor) -> TTensor:
        (a, c) = self.k_block.inverse(ld, a, c, c)
        (a, b) = self.e_block.inverse(ld, a, b)
        return (a, b)
