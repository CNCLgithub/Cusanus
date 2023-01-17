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

class KCodeDataset(FieldDataset):

    def __init__(self,
                 kf_dataset:KFieldDataset,
                 module: KField):

        self.kf_dataset = kf_dataset
        self.module = module
        self.device = module.device

    def __len__(self):
        return len(self.kf_dataset)

    def __getitem__(self, idx):

        # sample random initial scene and simulate
        world, position = self.simulations[idx]

        steps = position.shape[0]
        # sample time range
        # (double what was used from training)
        start = np.random.randint(0, steps - self.segment_steps * 2)
        stop = start + self.segment_steps
        xyz = position[start:stop:self.steps_per_frame]

        qs, ys = self.kf_dataset[idx]
        qs = torch.tensor(qs, device = self.device)
        ys = torch.tensor(ys, device = self.device)

        mfunc, mparams = self.module.fit_modulation(qs, ys)
        kcode = mfunc(mparams)



        kcode.detach().cpu().numpy()
        qs = qs.detach().cpu().numpy()
        ys = ys.detach().cpu().numpy()

        return qs, ys, kcode
