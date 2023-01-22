import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
from functools import partial
from functorch import make_functional, vmap
from torch.nn.functional import mse_loss, l1_loss

from cusanus.pytypes import *
from cusanus.archs import LatentModulation, ImplicitNeuralModule
from cusanus.tasks import ImplicitNeuralField, KField

from cusanus.tasks.inf import inner_modulation_loop

class EField(ImplicitNeuralField):
    """Implements kinematic event fields

    Arguments:
        inr: ImplicitNeuralModule, INR architecture
        lr: float = 0.001, learning rate
        weight_decay: float = 0.001
        sched_gamma: float = 0.8
    """

    def __init__(self,
                 module: ImplicitNeuralModule,
                 kfield: KField,
                 inner_steps:int = 5,
                 lr:float = 0.001,
                 lr_inner:float = 0.001,
                 weight_decay:float = 0.001,
                 sched_gamma:float = 0.8) -> None:
        super(ImplicitNeuralField, self).__init__()
        self.save_hyperparameters(ignore = ['module',
                                            'kfield'])
        self.module = module
        self.kfield = kfield

    """
    Arguments:
        qs:Tensor - position queries to eval
        k1:Tensor - predicted motion code
    """
    def pred_loss(self, k0: Tensor, qs: Tensor, k1:Tensor):
        ys = self.kfield.module(qs, k1)
        loss = torch.mean(ys**2) + torch.mean(k1**2)
        return loss


    def fit_kmod(self, qs: Tensor):
        k,_ = qs.shape
        ys = torch.zeros((k,1), device=qs.device,
                           dtype=qs.dtype,
                           requires_grad=qs.requires_grad)
        mfunc, mparams = self.kfield.fit_modulation(qs, ys)
        kmod = mfunc(mparams)
        return kmod

    def pos_code(self, qs:Tensor, kmod: Tensor):
        t = torch.max(qs[:, 0]).unsqueeze(0)
        pmod = self.kfield.module.motion_field(t, kmod)
        return pmod


    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        # qs for kmod_a , ys for kmod_b
        qs, ys = batch

        # motion codes
        vkmods = vmap(self.fit_kmod)(qs)
        # also need pos code
        vpmods = vmap(self.pos_code)(qs, vkmods)
        vmods = torch.cat((vkmods, vpmods),
                          axis = -1)
        vmods = vmods.detach().requires_grad_()

        # Fitting modulations for current generation
        # In parallel, trains one mod per task.
        vloss = vmap(partial(inner_modulation_loop, self))
        # fit modulations on batch - returns averaged loss
        # Compute the maml loss by summing together the returned losses.
        mod_losses = torch.mean(vloss(vmods, ys))
        self.log('loss', mod_losses.item())
        return mod_losses # overriding `backward`. See above

    @torch.enable_grad()
    @torch.inference_mode(False)
    def test_step(self, batch, batch_idx):
        (qs, ys) = batch
        qs = qs[0]
        ys = ys[0]
        kmod = self.fit_kmod(qs)
        pmod = self.pos_code(qs, kmod)
        k = torch.cat((kmod, pmod), axis = -1)
        m = self.fit_modulation(k, ys)
        k2 = self.eval_modulation(m, k)
        pred_loss = self.pred_loss(qs, ys, k2).detach().cpu()
        pred_ys = self.kfield.module(qs, k2).detach().cpu()
        self.log('test_loss', pred_loss)
        return {'loss' : pred_loss,
                'mod' : m,
                'kmod': k2,
                'pred':pred_ys}

    @torch.enable_grad()
    @torch.inference_mode(False)
    def validation_step(self, batch, batch_idx):
        (qs, ys) = batch
        qs = qs[0]
        ys = ys[0]
        kmod = self.fit_kmod(qs)
        pmod = self.pos_code(qs, kmod)
        k = torch.cat((kmod, pmod), axis = -1)
        m = self.fit_modulation(k, ys)
        k2 = self.eval_modulation(m, k)
        pred_loss = self.pred_loss(qs, ys, k2).detach().cpu()
        pred_ys = self.kfield.module(qs, k2).detach().cpu()
        self.log('val_loss', pred_loss)
        return {'loss' : pred_loss,
                'mod' : m,
                'pred':pred_ys}

    def configure_optimizers(self):

        params = [
            {'params': self.module.inr.theta.parameters()},
        ]
        optimizer = optim.Adam(params,
                               lr=self.hparams.lr,
                               weight_decay=self.hparams.weight_decay)
        gamma = self.hparams.sched_gamma
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma = gamma)
        return [optimizer], [scheduler]
