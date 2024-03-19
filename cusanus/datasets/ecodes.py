import torch
import numpy as np
from torch.utils.data import Dataset

from cusanus.pytypes import *
from cusanus.datasets import FieldDataset, SimDataset
from cusanus.tasks import KField

class EFieldDataset(FieldDataset):

    def __init__(self,
                 sim:SimDataset,
                 kfield:KField,
                 gfield:GField,
                 segment_frames:int=30,
                 mean:np.ndarray=np.zeros(2),
                 std:np.ndarray=np.zeros(2),
                 ):

        self.sim = sim
        self.segment_frames = segment_frames
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.sim)

    @property
    def qsize(self):
        return 3

    @property
    def ysize(self):
        return 3

    @property
    def k_queries(self):
        return self.segment_frames


    def trial_from_sequence(self, x, t0, t1, spf):
        qs = np.empty((self.segment_frames, self.ysize),
                      dtype = np.float32)
        qs[:, 0]  = np.linspace(0, (t1-t0)/240,
                                self.segment_frames)
        qs[:, 1:] = x[t0:t1:spf]
        return qs

    def fit_gcodes(self, scene:dict):
        gcodes = {}
        for (k,d) in scene.items():
            qs, ys = sample_from_geo(d['geometry'])
            gfunc, gparams = gfield.fit_modulation(qs, ys)
            gcodes[k] = gfunc(gparams).detach().cpu().numpy()

        return gcodes

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
        qsA = self.trial_from_sequence(x, t0, t1, spf)

        # pick second segment
        t2 = t1 + segment_steps
        qsB = self.trial_from_sequence(x, t1, t2, spf)

        return qsA, qsB
