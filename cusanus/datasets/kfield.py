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
                 nframes:int = 30,
                 mean:np.ndarray = np.zeros(2),
                 std:np.ndarray = np.ones(2),
                 add_noise:bool = False,
                 ):
        self.simulations = sim_dataset
        self.k_per_frame = k_per_frame
        self.segment_frames = nframes
        self._k_queries = nframes * k_per_frame
        self.mean = mean
        self.std = std
        self.small_noise_rv = scipy.stats.norm(loc = np.zeros(2),
                                               scale = np.ones(2))
        self.large_noise_rv = scipy.stats.uniform(loc = -1*np.ones(2),
                                                  scale = 2*np.ones(2))
        self.add_noise = add_noise

    @property
    def qsize(self):
        return 3

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
        _, registry, state = self.simulations[idx]
        target_id = registry['target']
        # position of the target across time
        # only want xy points
        position = state['position'][:, target_id, :2]
        steps = position.shape[0]
        # sample time scale
        # fps = [15, 30, 60, 120]
        fps = 15.0 * 2**np.random.randint(0, 3)
        # physics steps per frame
        spf = int(240 / fps)
        segment_steps = self.segment_frames * spf
        # sample time range
        start = np.random.randint(0, steps - segment_steps)
        stop = start + segment_steps
        position = position[start:stop]

        # construct queries and outputs
        qs = np.empty((self.segment_frames, self.k_per_frame, self.qsize),
                      dtype = np.float32)
        ys = np.empty((self.segment_frames, self.k_per_frame, self.ysize),
                      dtype = np.float32)
        for t in range(self.segment_frames):
            tn = t / fps
            # normalized to mu 0, std 1
            x = (position[t*spf] - self.mean) / self.std
            for i in range(self.k_per_frame):
                if self.add_noise:
                    # add small or large amounts of noise
                    if np.random.rand() > 0.5:
                        noise = 0.01 * self.small_noise_rv.rvs()
                    else:
                        noise = 3.0 * self.large_noise_rv.rvs()
                else:
                    noise = np.zeros(2)
                qs[t, i, 0] = tn
                qs[t, i, 1:] = x + noise
                ys[t, i] = np.linalg.norm(noise)

        qs = qs.reshape((-1, self.qsize))
        ys = ys.reshape((-1, self.ysize))

        return qs,ys
