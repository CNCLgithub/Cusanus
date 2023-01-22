import os
import h5py
from torch.utils.data import Dataset


def write_to_hdf5(d:SizedDataset, path:str):
    with h5py.File(path, 'w') as f:
        # class attributes
        f.attrs['len'] = len(d)

        # initialize hdf5 datasets
        # one dataset per part
        parts = {}
        for p in d.parts:
            eshape = (1, *d.enum_shape[p])
            shape = (n, *d.enum_shape[p])
            etype = d.dtype[p]
            # chunk at the trial level
            parts[p] = f.create_dataset(p,
                                       shape=shape,
                                       chunks=eshape,
                                       dtype=etype)
        # populate from dataset
        for i in range(len(d)):
            trial = d[i]
            for p,d in zip(d.parts, trial):
                parts[p][i] = d
    return None



class H5Dataset(Dataset):

    def __init__(self, d:Type[SizedDataset], src:str):
        self.d = d
        self.src = h5py.File(src, 'r')

    def __len__(self):
        return src.attrs['len']

    def __getitem__(self, idx):
        trial = []
        for p in self.d.parts:
            part = self.src[p][idx]
            trial.append(part)
        return trial
