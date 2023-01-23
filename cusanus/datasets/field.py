import numpy as np
from abc import ABC, abstractmethod

from cusanus.pytypes import *
from cusanus.datasets import SizedDataset

class FieldDataset(SizedDataset, ABC):

    @classmethod
    @property
    def dtype(cls):
        return {'qs' : np.dtype('float32'),
                'ys' : np.dtype('float32')}

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

    @property
    def qshape(self) -> Tuple[int, int]:
        return (self.k_queries, self.qsize)

    @property
    def yshape(self) -> Tuple[int, int]:
        return (self.k_queries, self.ysize)

    @property
    def enum_shape(self):
        return {'qs' : self.qshape, 'ys' : self.yshape}

