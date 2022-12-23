import torch
from cusanus.pytypes import *

def grids_along_depth(ny : int, nxz : int,
                      delta:float = 1.0):
    y = torch.linspace(-delta, delta, steps = ny)
    xz = torch.linspace(-delta, delta, steps = nxz)
    cube = torch.cartesian_prod(y, xz, xz)
    # reorder y axis
    cube = cube.roll(1).reshape(-1,3)
    return cube

def grids_along_axis(naxis : int, nres : int,
                     lead:int = 0, delta:float = 1.0):

    axis = torch.linspace(-delta, delta, steps = naxis)
    rest = torch.linspace(-delta, delta, steps = nres)
    cube = torch.cartesian_prod(axis, rest, rest)
    # reorder axis
    cube = cube.roll(lead).reshape(-1,3)
    return cube

def motion_grids(nt : int, nxyz : int, delta:float = 1.0):
    time = torch.linspace(0., 1.0, steps = nt)
    rest = torch.linspace(-delta, delta, steps = nxyz)
    tcube = torch.cartesian_prod(time, rest, rest, rest)
    # reorder axis
    tcube = tcube.reshape(-1,4)
    return tcube
