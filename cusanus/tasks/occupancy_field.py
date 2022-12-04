from cusanus.pytypes import *
from cusanus.tasks import ImplicitNeuralField
from torch.nn.functional import mse_loss

class OccupancyField(ImplicitNeuralField):
    """Implements an occupancy field distribution """
    def inner_loss(self, pred_ys: Tensor, ys: Tensor):
        return mse_loss(pred_ys, ys)
