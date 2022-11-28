import os
import torch
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from ensembles.pytypes import *
from ensembles.archs import BetaVAE

class SymEmbedding(pl.LightningModule):
    """Task of embedding symbolic object into z-space"""

    def __init__(self,
                 vae_model: BetaVAE,
                 lr:float = 1E-4,
                 weight_decay:float = 0.0,
                 sched_gamma:float = 0.9) -> None:
        super().__init__()
        self.model = vae_model
        self.lr = lr
        self.weight_decay = weight_decay
        self.sched_gamma = sched_gamma

    def forward(self, dk: Tensor) -> Tensor:
        return self.model(dk)

    def loss_function(self,
                      recons:Tensor, gt:Tensor,
                      mu:Tensor, log_var:Tensor) -> dict:
        #recons_loss =F.mse_loss(recons_dk, dk)/(dk.shape[-1]) + F.mse_loss(recons_ogs, ogs)/(ogs.shape[-1] * ogs.shape[-2])
        recons_loss =F.mse_loss(recons, gt)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + self.model.beta * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        x = batch[0]
        results = self.forward(x)
        train_loss = self.loss_function(*results)
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        x = batch[0]
        results = self.forward(x)
        loss = self.loss_function(*results)
        self.log_dict({f"val_{key}": val.item() for key, val in loss.items()}, sync_dist=True)
        return loss['loss']

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.lr,
                               weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma = self.sched_gamma)
        return [optimizer], [scheduler]
