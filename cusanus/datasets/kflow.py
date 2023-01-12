import torch
from torch.utils.data import Dataset
import transforms3d
import numpy as np
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (Convert, ToTensor,
                             ToDevice)

from cusanus.pytypes import *
from cusanus.datasets import SimDataset

class KFlowDataset(Dataset):

    def __init__(self,
                 sim_dataset:SimDataset,
                 segment_dur:float = 2000.0,
                 t_scale:float = 60.0,
                 mean:np.ndarray = np.zeros(12),
                 std:np.ndarray = np.ones(12),
                 ):
        self.simulations = sim_dataset
        segment_steps = np.floor(segment_dur / (1000/240)).astype(int)
        dur_per_frame = 1000.0 / 60.0
        steps_per_frame = np.floor(dur_per_frame * 240/1000).astype(int)
        nframes = np.ceil(segment_dur / dur_per_frame).astype(int)
        self.segment_dur = segment_dur
        self.segment_steps = segment_steps
        self.segment_frames = nframes
        self.steps_per_frame = steps_per_frame
        self._k_queries = nframes
        self.t_scale = t_scale
        self.mean = mean
        self.std = std

    @property
    def qsize(self):
        return 1

    @property
    def ysize(self):
        return 3

    @property
    def k_queries(self):
        return self._k_queries

    def __len__(self):
        return len(self.simulations)


    def __getitem__(self, idx):
        # sample random initial scene and simulate
        _, position = self.simulations[idx]
        steps = position.shape[0]
        # sample time range
        start = np.random.randint(0, steps - self.segment_steps)
        stop = start + self.segment_steps
        xyz = position[start:stop:self.steps_per_frame]

        ts = np.arange(self.segment_frames) / self.t_scale
        #  x y z
        # a
        # b
        # c
        coefs = np.polyfit(ts, xyz, 2).astype(np.float32)
        # a b c
        norms = np.linalg.norm(coefs, axis = 0)
        # Tait-Bryan euler-angles:
        #   x  y  z
        # a
        # b
        # c
        angles = np.empty((3,3), dtype = np.float32)
        for coef in range(3):
            nv = coefs[coef] / norms[coef]
            m = rotm_to_xaxis(nv)
            angles[coef] = transforms3d.taitbryan.mat2euler(m)

        # compute residuals
        resds = np.empty_like(xyz, dtype = np.float32)
        for d in range(3):
            resds[:, d] = np.abs(xyz[:, d] - \
                                 np.polyval(coefs[:, d], ts))


        x = np.empty(12, dtype = np.float32)
        x[:3] = norms
        x[3:] = angles.reshape(9)
        x = (x - self.mean) / self.std

        return (x, resds)

    def write_ffcv(self, path:str, **writer_kwargs):
        fields = {
            'x': NDArrayField(dtype = np.dtype('float32'),
                              shape = (12,)),
            'resids': NDArrayField(dtype = np.dtype('float32'),
                                shape = (self.k_queries, 3)),
        }
        writer = DatasetWriter(path, fields, **writer_kwargs)
        writer.from_indexed_dataset(self)

    @staticmethod
    def load_ffcv_flow(p:str, device, **kwargs):
        ps = {
            'x': [NDArrayDecoder(),
                  ToTensor(),
                  Convert(torch.float32)],
            'resids': None
            }
        if not device is None:
            ps['x'].append(ToDevice(device))
        return Loader(p, pipelines = ps, order = OrderOption(2),
                      **kwargs)

# trimesh.geometry.align_vectors
# trimesh.geometry.vector_angle(pairs)
#
#transforms3d.taitbryan.mat2euler

# from https://stackoverflow.com/a/59204638
def rotm_to_xaxis(b):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param b: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a = np.zeros(3)
    a[0] = 1
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
