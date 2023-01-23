from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Type
from cusanus.pytypes import *

from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (Convert, NormalizeImage, ToTensor,
    ToDevice)

class SizedDataset(Dataset, ABC):

    @classmethod
    @property
    @abstractmethod
    def parts(cls) -> List[str]:
        pass

    @classmethod
    @property
    @abstractmethod
    def dtype(cls) -> dict:
        pass

    @property
    @abstractmethod
    def enum_shape(self) -> dict:
        pass

    def write_ffcv(self, path:str, **kwargs) -> None:
        return write_ffcv(self, path, **kwargs)

    @classmethod
    def load_ffcv(cls, path:str, device, **kwargs) -> Loader:
        return load_ffcv(cls, path, device, **kwargs)

def write_ffcv(d:SizedDataset, path:str, **writer_kwargs):
    fields = {}
    for part in d.parts:
        fields[part] = NDArrayField(dtype = d.dtype[part],
                                    shape = d.enum_shape[part])
    writer = DatasetWriter(path, fields, **writer_kwargs)
    writer.from_indexed_dataset(d)

def load_ffcv(cls:Type[SizedDataset], p:str, device, **kwargs):
    pipes = {}
    for part in cls.parts:
        pipe = [NDArrayDecoder(),
                ToTensor(),
                Convert(torch.float32)]
        if not device is None:
            pipe.append(ToDevice(device))
        pipes[part] = pipe

    return Loader(p, pipelines = pipes, order = OrderOption(2),
                  **kwargs)
