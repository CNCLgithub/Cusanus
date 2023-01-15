import torch
import scipy
import trimesh
import numpy as np
import pybullet as p

from cusanus.pytypes import *
from cusanus.datasets import FieldDataset, SimDataset

class KFieldDataset(FieldDataset):

    def __init__(self,
                 sim_dataset:SimDataset,
                 segment_dur:float = 2000.0,
                 t_scale:float = 60.0,
                 y_mean:np.ndarray = np.zeros(3),
                 y_std:np.ndarray = np.ones(3),
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
        self.y_mean = y_mean
        self.y_std = y_std

    @property
    def qsize(self):
        return 1

    @property
    def ysize(self):
        return 6

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

        qs = np.arange(self.segment_frames,
                       dtype = np.float32) / self.segment_steps

        xyz = position[start:stop:self.steps_per_frame]
        xyz = (xyz - self.y_mean) / self.y_std

        coefs = np.polyfit(qs, xyz, 2).astype(np.float32)
        resds = np.empty_like(xyz, dtype = np.float32)
        for d in range(3):
            resds[:, d] = np.abs(xyz[:, d] - \
                                 np.polyval(coefs[:, d], qs))

        ys = np.concatenate([xyz, resds], axis = 1).astype(np.float32)
        return qs, ys
