import torch
import numpy as np
from torch.utils.data import Dataset

from cusanus.pytypes import *

class ECodesDataset(FieldDataset):

    def __init__(self,
                 kf_dataset:KFieldDataset,
                 module: KField,
                 mean,
                 std,
                 uthresh:float):

        self.kf_dataset = kf_dataset
        self.module = module
        self.device = module.device
        self.mean = mean
        self.std = std
        self.uthresh = uthresh

    def __len__(self):
        return len(self.kf_dataset)

    def trial_from_sequence(self, xyz, t0, t1):

# Mean: [6.79334173e-01 4.53180075e-17 2.42131324e+00], Std. Dev.: [1.77139619e+00 7.34710679e-15 2.04555669e+00]
    def __getitem__(self, idx):

        # sample random initial scene and simulate
        world, position = self.simulations[idx]
        xyz = (position - self.mean) / self.std

        steps = position.shape[0]
        # sample time range
        # (double what was used from training)
        t0 = np.random.randint(0, steps - segment_steps * 3)
        t1 = t0 + segment_steps
        qsA, ysA = self.trial_from_sequence(xyz, t0, t1) # TODO
        m = self.module.fit_modulation(qsA, ysA)

        # pick second segment
        t2 = t1 + segment_steps
        t3 = t2 + segment_steps
        qs, _ = self.trial_from_sequence(xyz, t2, t3)

        return kcode, qs
