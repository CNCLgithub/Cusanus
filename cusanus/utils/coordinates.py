import torch
from cusanus.pytypes import *

def grids_along_depth(ny : int, nxz : int):
    y = torch.linspace(-1, 1, steps = ny)
    xz = torch.linspace(-1, 1, steps = nxz)
    cube = torch.cartesian_prod(y, xz, xz)
    # reorder y axis
    cube = cube.roll(1).reshape(-1,3)
    return cube
