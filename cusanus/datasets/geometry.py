import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (Convert, NormalizeImage, ToTensor,
    ToDevice)

import trimesh

from cusanus.pytypes import *
from cusanus.utils import grids_along_depth
from cusanus.utils.meshes import sheered_rect_prism, ramp_mesh

_pipe = [NDArrayDecoder(),
         ToTensor(),
         Convert(torch.float32)]
_pipelines = {'qs': _pipe, 'ys': _pipe}

def write_ffcv(d:Dataset, k:int, path:str):
    qshape = (k, 3)
    yshape = (k, 1)
    fields = {
        'qs': NDArrayField(dtype = np.dtype('float32'),
                            shape = qshape),
        'ys': NDArrayField(dtype = np.dtype('float32'),
                            shape = yshape),
    }
    writer = DatasetWriter(path, fields)
    writer.from_indexed_dataset(d)

def load_ffcv(p:str, device, **kwargs):
    ps = {}
    for k in ['qs', 'ys']:
        ps[k] = deepcopy(_pipe)
        if not device is None:
            ps[k].append(ToDevice(device))
    return Loader(p, pipelines = ps, order = OrderOption(3),
                  **kwargs)

class OccupancyFieldDataset(Dataset):
    ffcv_pipelines = _pipelines

    def write_ffcv(self, path:str):
        write_ffcv(self,self.k_queries,path)


class SphericalGeometryDataset(OccupancyFieldDataset):

    # inheriting pytorch dataset; return vector of object and gstate
    def __init__(self, n_shapes:int = 1000, k_queries:int = 100,
                 r_min:float = 0.1, r_max:float = 0.8,
                 sigma:float=3.0) -> None:
        self.n_shapes = n_shapes
        self.k_queries = k_queries
        self.r_min = r_min
        self.r_max = r_max
        self.sigma = sigma

    def __len__(self):
        return self.n_shapes

    def __getitem__(self, idx):
        radius = np.random.uniform(low = self.r_min,
                                   high = self.r_max)
        qs = np.random.normal(scale = radius * self.sigma,
                              size = (self.k_queries, 3))
        qs = qs.astype(np.float32)
        ys = spherical_occupancy_field(radius, qs)
        return (qs, ys)


def spherical_occupancy_field(r: float, qs: np.ndarray):
    # return np.linalg.norm(qs, axis = 1) - r
    bs = r >= np.linalg.norm(qs, axis = 1)
    ys = bs.astype(np.float32)
    return ys

class MeshGeometryDataset(OccupancyFieldDataset):

    # inheriting pytorch dataset; return vector of object and gstate
    def __init__(self, n_shapes:int = 1000, k_queries:int = 100,
                 delta_y:float = 0.1, delta_size:float = 0.1,
                 qsigma:float = 1.0,
                 obs_extents:List[float] = [1.5,1.5,2.5],
                 ramp_extents:List[float]= [4.0, 1.5, .1]) -> None:
        self.n_shapes = n_shapes
        self.k_queries = k_queries
        self.delta_y = delta_y
        self.delta_size = delta_size
        self.qsigma = qsigma
        self.axis = np.array([0., 1., 0.])
        self.obs_extents = obs_extents
        self.ramp_extents = ramp_extents

    def __len__(self):
        return self.n_shapes

    def sample_obstacle(self):
        return sample_obstacle(self.obs_extents, 0.1,
                               np.pi * 0.5)

    def sample_ramp(self):
        return sample_ramp(self.ramp_extents, 0.1,
                           np.pi * 0.4)


    def __getitem__(self, idx):

        # obstacle
        if np.random.rand() > 0.5:
            mesh = self.sample_obstacle()
        else:
            mesh = self.sample_ramp()

        # sample random rotation along y axis (radians)
        ytheta = np.random.uniform(low = -self.delta_y,
                                   high = self.delta_y)
        rotm = trimesh.transformations.rotation_matrix(ytheta,
                                                       self.axis)
        # sample random size scale
        scale = np.random.uniform(low = 1.0 - self.delta_size,
                                  high = 1.0 + self.delta_size)
        sclm = trimesh.transformations.scale_matrix(scale)
        # apply transforms
        mesh.apply_transform(sclm)
        mesh.apply_transform(rotm)

        # sample qs
        bounds = np.max(np.abs(mesh.bounds))
        qs = np.random.normal(scale = bounds * self.qsigma,
                              size = (self.k_queries, 3))
        qs = qs.astype(np.float32)
        # comput occupancy outputs
        ys = mesh_occupancy_field(mesh, qs)
        return (qs, ys)


def mesh_occupancy_field(mesh, qs: np.ndarray):
    #NOTE: Points OUTSIDE the mesh will have NEGATIVE distance
    # Points within tol.merge of the surface will have POSITIVE distance
    # Points INSIDE the mesh will have POSITIVE distance
    ds = trimesh.proximity.signed_distance(mesh, qs)
    bs = np.array(ds) >= 0
    return bs.astype(np.float32)
