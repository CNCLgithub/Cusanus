import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
from functorch import make_functional
from torch.nn.functional import mse_loss, l1_loss

from cusanus.pytypes import *
from cusanus.archs import PQSplineModule, LatentModulation
from cusanus.tasks import ImplicitNeuralField

from cusanus.tasks.inf import fit_and_eval

class KSplineField(ImplicitNeuralField):
    """Implements kinematic spline fields

    Arguments:
        inr: ImplicitNeuralModule, INR architecture
        lr: float = 0.001, learning rate
        weight_decay: float = 0.001
        sched_gamma: float = 0.8
    """

    def __init__(self,
                 module: PQSplineModule,
                 inner_steps:int = 5,
                 lr:float = 0.001,
                 lr_inner:float = 0.001,
                 weight_decay:float = 0.001,
                 sched_gamma:float = 0.8) -> None:
        super(ImplicitNeuralField, self).__init__()
        self.save_hyperparameters(ignore = 'module')
        self.module = module

    def pred_loss(self, qs: Tensor, ys: Tensor, pred):
        pred_ys = pred
        loss = mse_loss(ys, pred_ys)
        return loss

    @torch.enable_grad()
    @torch.inference_mode(False)
    def test_step(self, batch, batch_idx):
        (qs, ys) = batch
        qs = qs[0]
        ys = ys[0]
        m = self.fit_modulation(qs, ys)
        pred = self.eval_modulation(m, qs)
        pred_loss = self.pred_loss(qs, ys, pred).detach().cpu()
        self.log('test_loss', pred_loss)
        return {'loss' : pred_loss,
                'mod' : m,
                'pred':pred.detach().cpu()}

    @torch.enable_grad()
    @torch.inference_mode(False)
    def validation_step(self, batch, batch_idx):
        (qs, ys) = batch
        qs = qs[0]
        ys = ys[0]
        m = self.fit_modulation(qs, ys)
        pred = self.eval_modulation(m, qs)
        pred_loss = self.pred_loss(qs, ys, pred).detach().cpu()
        self.log('val_loss', pred_loss)
        return {'loss' : pred_loss,
                'mod' : m,
                'pred':pred.detach().cpu()}



    def configure_optimizers(self):

        params = [
            {'params': self.module.pos_field.theta.parameters()},
            {'params': self.module.motion_field.theta.parameters()},
        ]
        optimizer = optim.Adam(params,
                               lr=self.hparams.lr,
                               weight_decay=self.hparams.weight_decay)
        gamma = self.hparams.sched_gamma
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma = gamma)
        return [optimizer], [scheduler]
