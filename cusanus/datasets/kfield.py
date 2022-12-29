import time
import torch
import trimesh
import numpy as np
import pybullet as p

from cusanus.pytypes import *
from cusanus.datasets import FieldDataset
from cusanus.utils.meshes import sample_ramp, sample_obstacle
from cusanus.utils.physics import mesh_to_bullet, sphere_to_bullet

class KFieldDataset(FieldDataset):

    def __init__(self,
                 sim_dataset:SimDataset,
                 k_per_frame:int = 5,
                 segment_dur:float = 2000.0,
                 ):
        self.simulations = sim_dataset
        self.k_per_frame = k_per_frame
        segment_steps = np.floor(segment_dur / (1000/240)).astype(int)
        steps_per_frame = np.floor(dur_per_frame * 240 / 1000).astype(int)
        nframes = np.ceil(segment_dur / dur_per_frame).astype(int)
        self.segment_dur = segment_dur
        self.segment_steps = segment_steps
        self.segment_frames = nframes
        self.steps_per_frame = steps_per_frame
        self._k_queries = nframes * k_per_frame

    @property
    def qsize(self):
        return 4

    @property
    def ysize(self):
        return 1

    @property
    def k_queries(self):
        return self._k_queries

    def __len__(self):
        return len(self.simulations)


    def __getitem__(self, idx):
        # sample random initial scene and simulate
        _, position = self.simulations[idx]

        # sample time range
        start = np.random.randint(0, steps - self.segment_steps)
        stop = start + self.segment_steps
        gt_kinematics = position[start:stop:self.steps_per_frame]

        qs = np.empty((self.segment_frames, self.k_per_frame, self.qsize),
                      dtype = np.float32)
        ys = np.empty((self.segment_frames, self.k_per_frame, self.ysize),
                      dtype = np.float32)
        for t in range(self.segment_frames):
            tn = t / self.segment_frames
            ks = gt_kinematics[t]
            noise = np.random.normal(0.0, 10.0,
                                   size = (self.k_per_frame,
                                           ks.shape[0]))
            qs[t, :, 0] = tn
            qs[t, :, 1:] = ks + noise
            val = np.linalg.norm(noise, axis = 1, keepdims=True)
            ys[t] = val

        qs = qs.reshape((-1, self.qsize))
        ys = ys.reshape((-1, self.ysize))

        return qs, ys
