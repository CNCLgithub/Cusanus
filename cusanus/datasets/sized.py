from abc import ABC
from torch.utils.data import Dataset

class SizedDataset(Dataset, ABC):

    @classmethod
    @property
    @abstractmethod
    def parts(cls) -> List[str]:
        pass

    @property
    @abstractmethod
    def enum_shape(self) -> dict:
        pass

    @classmethod
    @property
    @abstractmethod
    def dtype(cls) -> dict:
        pass
