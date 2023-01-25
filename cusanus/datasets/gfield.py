import torch
import trimesh
import numpy as np
from abc import ABC
from functools import partial
from torch.utils.data import Dataset

from cusanus.pytypes import *
from cusanus.datasets import FieldDataset, SceneDataset
from cusanus.utils import grids_along_depth
from cusanus.utils.meshes import center_mesh

class ShapeDataset(Dataset):

    def __init__(self,
                 n_shapes:int = 100,
                 rect_x_rng:List[float] = [0.1, 4.0],
                 rect_y_rng:List[float] = [0.05, 0.1],
                 rect_z:float = 0.1,
                 sphere_prob:float = 0.5,
                 sphere_radius_rng:List[float] = [0.01, 1.0],
                 ):
        self.n_shapes = n_shapes
        self.rect_x_rng = rect_x_rng
        self.rect_y_rng = rect_y_rng
        self.rect_z = rect_z
        self.sphere_prob = sphere_prob
        self.sphere_radius_rng = sphere_radius_rng


    def __len__(self):
        return self.n_shapes

    def sample_ocf(self):
        if np.random.rand() > (1-self.sphere_prob):
            obj = np.random.uniform(*self.sphere_radius_rng)
            ocf = partial(spherical_occupancy_field, obj)
        else:
            extents = [np.random.uniform(*self.rect_x_rng),
                       np.random.uniform(*self.rect_y_rng),
                       self.rect_z]
            obj = trimesh.primitives.Box(extents = extents)
            # apply random rotation along z-axis
            theta = np.random.uniform(0, np.pi)
            rm = trimesh.transformations.rotation_matrix(theta, [0., 0., 1])
            obj.apply_transform(rm)
            ocf = partial(mesh_occupancy_field, obj)

        return (obj, ocf)



    def __getitem__(self, idx):
        return self.sample_ocf()



class GFieldDataset(FieldDataset):

    def __init__(self, shapes:ShapeDataset,
                 k_inside:int = 100,
                 k_other:int = 100,
                 k_outside:int = 100,
                 qsigma:float = 3.0,
                 qmean:np.ndarray=np.zeros(2),
                 qstd:np.ndarray=np.ones(2),
                 test:bool = False,
                 ) -> None:
        self.shapes = shapes
        self.k_inside = k_inside
        self.k_other = k_other
        self.k_outside = k_outside
        self._k_queries = k_inside + k_outside
        self._k_queries = k_inside + k_other + k_outside
        self.qsigma = qsigma
        self.qmean = qmean
        self.scene_bounds = [qsigma, qsigma]
        self.qstd = qstd
        self.test = test

    @property
    def qsize(self):
        return 2

    @property
    def ysize(self):
        return 1

    @property
    def k_queries(self):
        return self._k_queries

    def __len__(self):
        return len(self.shapes)

    def __getitem__(self, idx):

        # sample target object
        obj, ocf = self.shapes.sample_ocf()

        # sample another object
        other, _ = self.shapes.sample_ocf()

        # sample qs
        qs = np.zeros((self.k_queries, 3), dtype = np.float32)
        # inside object
        qs_in = query_inside(obj, self.k_inside)
        qs[:self.k_inside, :2] = qs_in
        # other
        qs_other = query_around(obj, self.k_other)
        qs[self.k_inside:-self.k_outside, :2] = qs_other
        # outside object
        # qs[-self.k_outside:, :2] = np.random.normal(scale=self.qsigma,
        #                                             size=(self.k_outside,2))
        qs[-self.k_outside:, :2] = sample_inside_bounds(self.scene_bounds,
                                                        self.k_outside)
        # comput occupancy outputs
        ys = ocf(qs).astype(np.float32)
        # only xy points
        qs = (qs[:, :2]).astype(np.float32)
        return (qs, ys)

def sample_inside_bounds(bounds, k, s:float = 2.0):
    dx,dy = bounds
    return np.random.uniform([-s*dx, -s*dy],
                             [ s*dx,  s*dy],
                             size = (k, 2))


def query_inside(obj, k:int):
    if isinstance(obj, float):
        radius = obj
        bounds = [radius, radius]
        qs = sample_inside_bounds(bounds, k, s = 1.0)
    else:
        mesh = center_mesh(obj)
        qs = mesh.sample_volume(k)[:, :2]

    return qs

def query_around(obj, k:int):
    if isinstance(obj, float):
        radius = obj
        bounds = [radius, radius]
    else:
        mesh = center_mesh(obj)
        bounds = 0.5 * mesh.extents[:2]

    kq = np.floor(np.sqrt(k)).astype(int)
    xbounds = (-2*bounds[0], 2*bounds[0])
    ybounds = (-2*bounds[1], 2*bounds[1])
    xs = np.linspace(*xbounds, kq)
    ys = np.linspace(*ybounds, kq)
    qx,qy = np.meshgrid(xs,ys)
    qs = np.concatenate([qx.reshape(-1, 1), qy.reshape(-1, 1)], axis = 1)
    qs += np.random.normal(scale = 0.1, size=(k, 2))
    # qs = sample_inside_bounds(bounds, k, s = 2.15)
    return qs

def spherical_occupancy_field(r: float, qs: np.ndarray):
    bs = (np.linalg.norm(qs, axis = 1) - r) <= 0
    return bs


def mesh_occupancy_field(mesh, qs: np.ndarray):
    return mesh.contains(qs)

# def spherical_occupancy_field(r: float, qs: np.ndarray):
#     return np.linalg.norm(qs, axis = 1) - r


# def mesh_occupancy_field(mesh, qs: np.ndarray):
#     return -1 * trimesh.proximity.signed_distance(mesh, qs)
