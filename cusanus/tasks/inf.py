import pytorch_lightning as pl
from ensembles.pytypes import *

class ImplicitNeuralDistribution(pl.LightingModule, ABC):

    def __init__(self):
        super().__init__()

    @abstractproperty
    def query(self):
        pass

    def forward(self):
        pass
