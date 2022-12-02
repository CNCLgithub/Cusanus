import torch
import numpy as np
from torch.utils.data import Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import (Convert, NormalizeImage, ToTensor,
    ToDevice)

from ensembles.pytypes import *

_pipe = [NDArrayDecoder(),
         ToTensor(),
         Convert(torch.float32)]
_pipelines = {'qs': pipe, 'ys': pipe}

class SphericalGeometryDataset(Dataset):

    ffcv_pipelines = _pipelines

    # inheriting pytorch dataset; return vector of object and gstate
    def __init__(self, n_shapes:int = 1000, k_queries:int = 1000,
                 r_min:float = 0.1, r_max:float = 0.9) -> None:
        self.n_shapes = n_shapes
        self.k_queries = k_queries
        self.r_min = r_min
        self.r_max = rmax

    def __len__(self):
        return self.n_shapes

    def __getitem__(self, idx):
        radius = np.random.uniform(low = self.r_min,
                                   high = self.r_max)
        qs = np.random.uniform(low = -1.0,
                               high = 1.0,
                               size = (self.k_queries, 3))
        qs = qs.astype(np.float32)
        ys = spherical_occupancy_field(r, qs)
        return (qs, ys)

    def write_ffcv(self, path:str) -> None:
        kwargs = {
            'qs': NDArrayField(dtype = np.dtype('float32'),
                               shape = (self.k_queries, 3)),
            'ys': NDArrayField(dtype = np.dtype('float32'),
                               shape = (self.k_queries, 1)),
        }
        writer = DatasetWriter(path, **kwargs)
        writer.from_indexed_dataset(self)

    def ffcv_pipelines(self):
        pipe = [NDArrayDecoder(),
                ToTensor(),
                Convert(torch.float32)]
        pipelines = {'qs': pipe, 'ys': pipe}
        return pipelines

def spherical_occupancy_field(r: float, qs: np.ndarray):
    bs = r <= np.linalg.norm(qs, axis = 1)
    ys = bs.astype(np.float32)
    return ys
