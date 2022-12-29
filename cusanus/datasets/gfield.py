import torch
import trimesh
import numpy as np
from abc import ABC

from cusanus.pytypes import *
from cusanus.datasets import FieldDataset
from cusanus.utils import grids_along_depth
from cusanus.utils.meshes import sheered_rect_prism, ramp_mesh

class GFieldDataset(FieldDataset, ABC):
    @property
    def qsize(self):
        return 3
    @property
    def ysize(self):
        return 1

class SphericalGFieldDataset(GFieldDataset):

    # inheriting pytorch dataset; return vector of object and gstate
    def __init__(self, n_shapes:int = 1000, k_queries:int = 100,
                 r_min:float = 0.1, r_max:float = 0.8,
                 sigma:float=3.0) -> None:
        self.n_shapes = n_shapes
        self._k_queries = k_queries
        self.r_min = r_min
        self.r_max = r_max
        self.sigma = sigma

    @property
    def k_queries(self):
        return self._k_queries

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

class MeshGFieldDataset(GFieldDataset):

    # inheriting pytorch dataset; return vector of object and gstate
    def __init__(self, n_shapes:int = 1000, k_queries:int = 100,
                 delta_y:float = 0.1, delta_size:float = 0.1,
                 qsigma:float = 1.0,
                 obs_extents:List[float] = [1.5,1.5,2.5],
                 ramp_extents:List[float]= [4.0, 1.5, .1]) -> None:
        self.n_shapes = n_shapes
        self._k_queries = k_queries
        self.delta_y = delta_y
        self.delta_size = delta_size
        self.qsigma = qsigma
        self.axis = np.array([0., 1., 0.])
        self.obs_extents = obs_extents
        self.ramp_extents = ramp_extents

    @property
    def k_queries(self):
        return self._k_queries

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
    bs = mesh.contains(qs)
    return bs.astype(np.float32)
