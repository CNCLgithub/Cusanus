#cribbed from https://github.com/AntixK/PyTorch-VAE/blob/master/models/types_.py

from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor
from torch import BoolTensor
Tensor = TypeVar('torch.tensor')
TTensor = Tuple[Tensor, Tensor]
