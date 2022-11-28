import torch
from abc import ABC
from torch import nn
from ensembles.pytypes import *

class Channel(InvertibleModule, ABC):
    def __init__(self):
       super().__init__()

    @abstractmethod
    def invert(self) -> Channel:
        pass

    def inverse(self, x):
        self.invert().forward(x)

def invert(c: Channel) -> Channel:
    return c.invert()


class Broadcast(Channel):

    def __init__(self, c: Channel):
        self.c = c
        self.ic = invert(c)

    def forward(self, ld: Tensor, *args):
        results = []
        for x in args:
            y = self.c(ld, x)
            results.append(y)
        return tuple(results)

    def inverse(self, ld: Tensor, *args):
        results = []
        for x in args:
            y = self.ic(ld, x)
            results.append(y)
        return tuple(results)

    def invert(self):
        return Broadcast(self.ic)

class Splice(Channel):

    def __init__(self, mask: BoolTensor):
        super().__init__()
        self.register_buffer('mask', mask)
        self.register_buffer('neg_mask',
                             torch.logical_not(mask))

    def forward(self, ld:Tensor, x: Tensor) -> TTensor:
        xm = x[self.mask]
        xr = x[self.neg_mask]
        return (xm, xr)

    def invert(self):
        return UnSplice(self.mask)

    def complement(self):
        return Splice(torch.logical_not(self.mask))

class UnSplice(Splice):

    # TODO: check performance
    def forward(self, ld: Tensor, xs: TTensor) -> TTensor:
        x,y = xs
        z = torch.empty_like(self.mask,
                             dtype = x.dtype)
        z[self.mask] = x
        self[self.neg_mask] = y
        return z

    def invert(self):
        return Splice(self.mask)

    def complement(self):
        return UnSplice(self.neg_mask)

class DimSplit(Channel):
    def __init__(self, d: int = 1):
        super().__init__()
        self.d = d

    def forward(self, z: Tensor) -> Tuple:
        z1, z2 = z.chunk(2, dim=self.d)
        log_det = 0
        return (z1, z2), log_det

    # def inverse(self, z1z2: Tuple) -> Tuple:
    #     return self.invert().forward(z1z2)

    def invert(self):
        return DimMerge(d = self.d)

# TODO: add `ld` to the following

class DimMerge(Channel):
    def __init__(self, d: int = 1):
        super().__init__()
        self.d = d

    def forward(self, z1z2:Tuple[Tensor, Tensor]) -> Tuple:
        z = torch.cat(z1z2, 1)
        log_det = 0
        return z, log_det

    # def inverse(self, z:Tensor) -> Tuple:
    #     return self.invert().forward(z1z2)

    def invert(self):
        return DimSplit()

class CheckerboardSplit(Channel):

    def forward(self, z: Tensor):
        n_dims = z.dim()
        cb0 = 0
        cb1 = 1
        for i in range(1, n_dims):
            cb0_ = cb0
            cb1_ = cb1
            cb0 = [cb0_ if j % 2 == 0 else cb1_ for j in range(z.size(n_dims - i))]
            cb1 = [cb1_ if j % 2 == 0 else cb0_ for j in range(z.size(n_dims - i))]
        cb = cb1 if 'inv' in self.mode else cb0
        cb = torch.tensor(cb)[None].repeat(len(z), *((n_dims - 1) * [1]))
        cb = cb.to(z.device)
        z_size = z.size()
        z1 = z.reshape(-1)[torch.nonzero(cb.view(-1), as_tuple=False)].view(*z_size[:-1], -1)
        z2 = z.reshape(-1)[torch.nonzero((1 - cb).view(-1), as_tuple=False)].view(*z_size[:-1], -1)
        log_det = 0
        return (z1, z2), log_det

    def invert(self):
        return CheckboardMerge()


class CheckerboardMerge(Channel):
    def forward(self, z1z2: Tuple[Tensor, Tensor]):
        z1, z2 = z1z2
        n_dims = z1.dim()
        z_size = list(z1.size())
        z_size[-1] *= 2
        cb0 = 0
        cb1 = 1
        for i in range(1, n_dims):
            cb0_ = cb0
            cb1_ = cb1
            cb0 = [cb0_ if j % 2 == 0 else cb1_ for j in range(z_size[n_dims - i])]
            cb1 = [cb1_ if j % 2 == 0 else cb0_ for j in range(z_size[n_dims - i])]
        cb = cb1 if 'inv' in self.mode else cb0
        cb = torch.tensor(cb)[None].repeat(z_size[0], *((n_dims - 1) * [1]))
        cb = cb.to(z1.device)
        z1 = z1[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
        z2 = z2[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
        z = cb * z1 + (1 - cb) * z2
        log_det = 0
        return z, log_det

    def invert(self):
        return CheckboardMerge()
