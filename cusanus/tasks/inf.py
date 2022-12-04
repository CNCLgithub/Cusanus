import os
import torch
import torchopt
from torch import optim
from torch.nn.functional import mse_loss
import pytorch_lightning as pl
# import torchvision.utils as vutils
from functorch import vmap, make_functional, grad_and_value

from cusanus.pytypes import *
from cusanus.archs import ImplicitNeuralModule, LatentModulation

class ImplicitNeuralField(pl.LightningModule):
    """Implements a generic task implicit neural fields

    Arguments:
        inr: ImplicitNeuralModule, INR architecture
        lr: float = 0.001, learning rate
        weight_decay: float = 0.001
        sched_gamma: float = 0.8
    """

    def __init__(self,
                 inr: ImplicitNeuralModule,
                 inner_steps:int = 5,
                 lr:float = 0.001,
                 lr_inner:float = 0.001,
                 weight_decay:float = 0.001,
                 sched_gamma:float = 0.8) -> None:
        super().__init__()
        self.save_hyperparameters(ignore = 'inr')
        self.inr = inr
        # (tfunc, tparams) = make_functional(inr.theta,
        #                                    disable_autograd_tracking = True)
        # self.tfunc = tfunc
        # self.tparms = tparams
        # self.lr = lr
        # self.weight_decay = weight_decay
        # self.sched_gamma = sched_gamma

    def initialize_modulation(self):
        m = LatentModulation(self.inr.hidden)
        m.to(self.device)
        m.train()
        return make_functional(m)

    def initialize_inner_opt(self, mparams):
        optim = torchopt.sgd(self.hparams.lr_inner)
        opt_state = optim.init(mparams)
        return (optim, opt_state)

    def inner_loss(self, pred_ys : Tensor, ys : Tensor):
    # def inner_loss(self, mparams, qs: Tensor, ys: Tensor):
        # pred_ys = self.inr(qs, mparams)
        return mse_loss(pred_ys, ys)

    def inner_loop(self, qs : Tensor, ys : Tensor):
        m, mparams = self.initialize_modulation()
        inner_opt, inner_state = self.initialize_inner_opt(m)
        for _ in range(self.hparams.inner_steps):
            pred_ys = self.inr(qs, m(mparams))
            loss = mse_loss(pred_ys, ys)
            # grad, loss = grad_and_value(self.inner_loss)(pred_ys, ys)
            print(ys)
            print(loss)
            print(mparams)
            grad = torch.autograd.grad(loss, mparams)
            updates, inner_state = inner_opt.update(grad, inner_state)
            mparams = torchopt.apply_updates(mparams, updates)

        return (mparams, loss)

    def outer_loop(self, batch):
        # each trial in the batch is a group of queries and outputs
        qs, ys = batch
        # fitting modulations for current generation
        ms, ls = vmap(self.inner_loop)(qs, ys)
        loss = ls.mean() # average across batch
        return ms, loss

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        # fit modulations on batch - returns averaged loss
        _, loss = self.outer_loop(batch)
        # Update theta
        self.log_dict({'loss' : loss.item()}, sync_dist=True)
        return loss
        # return None  # skip the default optimizer steps by PyTorch Lightning

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        ms, loss = self.outer_loop(batch)
        self.log_dict({'val_loss' : loss.item()}, sync_dist = True)
        # TODO: add visualization
        #
        # val_loss = self.decoder.loss_function(pred_gs,
        #                                       real_gs,
        #                                       optimizer_idx=optimizer_idx,
        #                                       batch_idx = batch_idx)
        #     results = pred_gs.unsqueeze(1)
        # vutils.save_image(results.data,
        #                   os.path.join(self.logger.log_dir ,
        #                                "reconstructions",
        #                                f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
        #                   normalize=False,
        #                   nrow=6)
        # vutils.save_image(real_gs.unsqueeze(1).data,
        #                   os.path.join(self.logger.log_dir ,
        #                                "reconstructions",
        #                                f"gt_{self.logger.name}_Epoch_{self.current_epoch}.png"),
        #                   normalize=False,
        #                   nrow=6)
        # self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        return loss


    def configure_optimizers(self):

        optimizer = optim.Adam(self.inr.theta.parameters(),
                               lr=self.hparams.lr,
                               weight_decay=self.hparams.weight_decay)
        gamma = self.hparams.sched_gamma
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma = gamma)
        return [optimizer], [scheduler]
