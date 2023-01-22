import torch
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (Convert, NormalizeImage, ToTensor,
    ToDevice)

from cusanus.pytypes import *

def write_ffcv(d:Dataset, path:str, **writer_kwargs):
    fields = {
        'qs': NDArrayField(dtype = np.dtype('float32'),
                            shape = d.qshape()),
        'ys': NDArrayField(dtype = np.dtype('float32'),
                            shape = d.yshape()),
    }
    writer = DatasetWriter(path, fields, **writer_kwargs)
    writer.from_indexed_dataset(d)

def load_ffcv(p:str, device, **kwargs):
    ps = {}
    for k in ['qs', 'ys']:
        ps[k] = [NDArrayDecoder(),
                 ToTensor(),
                 Convert(torch.float32)]
        if not device is None:
            ps[k].append(ToDevice(device))
    return Loader(p, pipelines = ps, order = OrderOption(2),
                  **kwargs)


class FieldDataset(Dataset, ABC):

    @classmethod
    @property
    def dtype(cls):
        return {'qs' : np.float32, 'ys' : np.float32}

    @classmethod
    @property
    def parts(cls):
        return ['qs', 'ys']

    @property
    @abstractmethod
    def qsize(self) -> int:
        pass

    @property
    @abstractmethod
    def ysize(self) -> int:
        pass

    @property
    @abstractmethod
    def k_queries(self) -> int:
        pass

    def qshape(self) -> Tuple[int, int]:
        return (self.k_queries, self.qsize)

    def yshape(self) -> Tuple[int, int]:
        return (self.k_queries, self.ysize)

    def enum_shape(self):
        return {'qs' : self.qshape, 'ys' : self.yshape}



    def write_ffcv(self, path:str, **kwargs) -> None:
        write_ffcv(self, path, **kwargs)
