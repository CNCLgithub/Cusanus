import torch
import scipy
import trimesh
import numpy as np
import pybullet as p

from cusanus.pytypes import *
from cusanus.datasets import FieldDataset, SimDataset
from cusanus.tasks import KField

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

class KCodesDataset(FieldDataset):

    def __init__(self,
                 sim:SimDataset,
                 kfield:KField,
                 segment_frames:int=30,
                 mean:np.ndarray=np.zeros(2),
                 std:np.ndarray=np.zeros(2),
                 ):

        self.sim = sim
        self.kfield = kfield
        self.kdim = kfield.module.motion_field.mod
        self.pkdim = (kfield.module.pos_field.mod +
                      self.kdim)
        self.segment_frames = segment_frames
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.sim)

    @property
    def qsize(self):
        return self.pkdim

    @property
    def ysize(self):
        return self.kdim

    @property
    def k_queries(self):
        return 1


    def trial_from_sequence(self, x, t0, t1, spf):
        ts = torch.linspace(0, (t1-t0)/240, self.segment_frames,
                            device = self.kfield.device,
                            dtype = torch.float32,
                            requires_grad=True).unsqueeze(1)
        x = torch.tensor(x[t0:t1:spf],
                         device=self.kfield.device,
                         dtype = torch.float32,
                         requires_grad=True)
        noise = 0.1 * torch.rand_like(x)
        qs = torch.cat([ts, x + noise], axis = 1)
        ys = torch.linalg.vector_norm(noise, dim = 1,
                                      keepdim=True)
        return qs, ys

    def __getitem__(self, idx):
        # sample random initial scene and simulate
        _, registry, state = self.sim[idx]
        target_id = registry['target']
        # position of the target across time
        # only want xy points
        x = state['position'][:, target_id, :2]
        steps = x.shape[0]
        # sample time scale
        # fps = [15, 30, 60, 120]
        fps = 15.0 * 2**np.random.randint(0, 3)
        # physics steps per frame
        spf = int(240 / fps)
        segment_steps = self.segment_frames * spf
        # sample time range
        # (double what was used from training kmodule)
        t0 = np.random.randint(0, steps - segment_steps * 2)
        t1 = t0 + segment_steps
        qsA,ysA = self.trial_from_sequence(x, t0, t1, spf)
        kfunc, kparams = self.kfield.fit_modulation(qsA, ysA)
        mA = kfunc(kparams)
        t = torch.tensor([(t1-t0)/240],
                         dtype=torch.float32,
                         device=self.kfield.device).unsqueeze(1)
        pA = self.kfield.module.motion_field(t, mA).squeeze()
        kA = torch.cat([mA, pA], 0).detach().cpu().numpy()

        # pick second segment
        t2 = t1 + segment_steps
        qsB,ysB = self.trial_from_sequence(x, t1, t2, spf)
        kfunc, kparams = self.kfield.fit_modulation(qsB, ysB)
        kB = kfunc(kparams).detach().cpu().numpy()

        return kA, kB
