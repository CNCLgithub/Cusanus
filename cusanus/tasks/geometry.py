from cusanus.pytypes import *
from cusanus.tasks import ImplicitNeuralField

class OccupancyField(ImplicitNeuralField):
    """Implements an occupancy field distribution """
    def inner_loss(self, pred_ys: Tensor, ys: Tensor):
        return F.mse_loss(pred_ys, ys)

