import torch
from torch import optim

from cusanus.pytypes import *
from cusanus.tasks import ImplicitNeuralField

class GField(ImplicitNeuralField):
    """Implements occupancy fields

    Arguments:
        inr: ImplicitNeuralModule, INR architecture
        lr: float = 0.001, learning rate
        weight_decay: float = 0.001
        sched_gamma: float = 0.8
    """

    @torch.enable_grad()
    @torch.inference_mode(False)
    def test_step(self, batch, batch_idx):
        (qs, ys) = batch
        qs = qs[0]
        ys = ys[0]
        m = self.fit_modulation(qs, ys)
        pred = self.eval_modulation(m, qs)
        pred_diff = ys - pred
        pred_loss = self.pred_loss(qs, ys, pred).detach().cpu()
        self.log('test_loss', pred_loss)
        return {'loss' : pred_loss,
                'mod'  : m,
                'pred_diff' : pred_diff.detach().cpu()}

    @torch.enable_grad()
    @torch.inference_mode(False)
    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def configure_optimizers(self):

        params = [
            {'params': self.module.scale_field.parameters()},
            {'params': self.module.occ_field.theta.parameters()},
        ]
        optimizer = optim.Adam(params,
                               lr=self.hparams.lr,
                               weight_decay=self.hparams.weight_decay)
        gamma = self.hparams.sched_gamma
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma = gamma)
        return [optimizer], [scheduler]
