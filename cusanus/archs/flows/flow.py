from abc import ABC
from torch import nn

class InvertibleModule(nn.Module, ABC):
    """
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def inverse(self):
        pass

