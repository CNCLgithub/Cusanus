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
                 k_per_frame:int = 5,
                 segment_dur:float = 2000.0,
                 t_scale:float = 30.0,
                 mean:np.ndarray = np.zeros(3),
                 std:np.ndarray = np.ones(3),
                 add_noise:bool = False,
                 ):
        self.simulations = sim_dataset
        self.k_per_frame = k_per_frame
        segment_steps = np.floor(segment_dur / (1000/240)).astype(int)
        dur_per_frame = 1000.0 / 60.0
        steps_per_frame = np.floor(dur_per_frame * 240/1000).astype(int)
        nframes = np.ceil(segment_dur / dur_per_frame).astype(int)
        self.segment_dur = segment_dur
        self.segment_steps = segment_steps
        self.segment_frames = nframes
        self.steps_per_frame = steps_per_frame
        self._k_queries = nframes * k_per_frame
        self.t_scale = t_scale
        self.mean = mean
        self.std = std
        self.small_noise_rv = scipy.stats.norm(loc = np.zeros(3),
                                               scale = np.ones(3))
        self.large_noise_rv = scipy.stats.uniform(loc = -1*np.ones(3),
                                                  scale = 2*np.ones(3))
        self.add_noise = add_noise

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
        steps = position.shape[0]
        # sample time range
        start = np.random.randint(0, steps - self.segment_steps)
        stop = start + self.segment_steps
        xyz = position[start:stop:self.steps_per_frame]

        qs = np.empty((self.segment_frames, self.k_per_frame, self.qsize),
                      dtype = np.float32)
        ys = np.empty((self.segment_frames, self.k_per_frame, self.ysize),
                      dtype = np.float32)
        for t in range(self.segment_frames):
            tn = t / self.t_scale # 30fps
            # normalized to mu 0, std 1
            ks = (xyz[t] - self.mean) / self.std
            for i in range(self.k_per_frame):
                if self.add_noise:
                    # add small or large amounts of noise
                    if np.random.rand() > 0.5:
                        noise = 0.001 * self.small_noise_rv.rvs()
                    else:
                        noise = 3.0 * self.large_noise_rv.rvs()
                else:
                    noise = np.zeros(3)
                qs[t, i, 0] = tn
                qs[t, i, 1:] = ks + noise
                ys[t, i] = np.linalg.norm(noise)

        qs = qs.reshape((-1, self.qsize))
        ys = ys.reshape((-1, self.ysize))

        return qs,ys
