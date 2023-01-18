import torch
import numpy as np
from torch.utils.data import Dataset

from cusanus.pytypes import *

class KCodeDataset(Dataset):

    def __init__(self,
                 kf_dataset:KFieldDataset,
                 module: KField):

        self.kf_dataset = kf_dataset
        self.module = module
        self.device = module.device

    def __len__(self):
        return len(self.kf_dataset)

    def __getitem__(self, idx):

        qs, ys = self.kf_dataset[idx]
        qs = torch.tensor(qs, device = self.device)
        ys = torch.tensor(ys, device = self.device)

        mfunc, mparams = self.module.fit_modulation(qs, ys)
        kcode = mfunc(mparams).detach().cpu().numpy()
        qs = qs.detach().cpu().numpy()
        ys = ys.detach().cpu().numpy()

        return qs, ys, kcode

class KTransitionDataset(FieldDataset):

    def __init__(self,
                 kf_dataset:KFieldDataset,
                 module: KField):

        self.kf_dataset = kf_dataset
        self.module = module
        self.device = module.device

    def __len__(self):
        return len(self.kf_dataset)
# Mean: [6.79334173e-01 4.53180075e-17 2.42131324e+00], Std. Dev.: [1.77139619e+00 7.34710679e-15 2.04555669e+00]
    def __getitem__(self, idx):

        # sample random initial scene and simulate
        world, position = self.simulations[idx]
        position = (position - self.mean) / self.std

        steps = position.shape[0]
        # sample time range
        # (double what was used from training)
        t0 = np.random.randint(0, steps - self.segment_steps * 4)
        t1 = t0 + self.segment_steps
        tend = t0 + (4 * self.segment_steps)
        seqA = position[t0:t1:self.steps_per_frame]
        qsA, ysA = self.trial_from_sequence(seqA, t0 = 0) # TODO
        m = self.module.fit_modulation(qsAB, ysAB)

        # Evaluate fitted motion code on second sequence
        # compute non-parametric estimate of uncertaintity
        seqB = position[t1:tend:self.steps_per_frame]
        qsB = self.trial_from_sequence(seqB, t0 = t1 - t0)
        uqsB = self.add_noise(qsB) # TODO
        uysB = self.module.eval_modulation(uqsB, m).detach().numpy()
        uB = self.estimate_uncertainty(uysB) # TODO

        #
        above_thresh = np.nonzero(uB > self.y_thresh)

        # if kmod fits well then split in half
        if len(above_thresh) == 0:
            t1 = t0 + self.segment_steps
        else:
            t1 = t0 + above_thresh[0]

        qsB, _ = self.trial_from_sequence(seqB)

        return qs, ys, kcode
