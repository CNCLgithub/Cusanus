import os
import torch
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.utils as vutils

from ensembles.pytypes import *
from ensembles.archs import ImplicitNeuralModule

class GraphicsINR(pl.LightningModule):
    """Task of embedding symbolic object into z-space"""

    def __init__(self,
                 inr: ImplicitNeuralModule,
                 lr:float = 0.001,
                 weight_decay:float = 0.0,
                 sched_gamma:float = 0.8,
                 ih:int = 128,
                 iw:int = 128) -> None:
        super().__init__()
        self.inr = inr
        self.lr = lr
        self.weight_decay = weight_decay
        self.sched_gamma = sched_gamma
        self.grid_size = ih * iw
        tensors = [torch.linspace(-1, 1, steps = ih),
                   torch.linspace(-1, 1, steps = iw)]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing = 'ij'), dim=-1)
        mgrid = torch.reshape(mgrid, (-1, 2))
        print(mgrid.shape)
        # mgrid = rearrange(mgrid, 'h w c -> (h w) c')
        self.register_buffer('grid', mgrid)

    def forward(self, qs: Tensor, xs: Tensor) -> Tensor:
        return self.inr(qs, xs)

    def loss_function(self, pred_ys: Tensor, ys: Tensor):
        return F.mse_loss(pred_ys, ys)

    def package_batch(self, batch):
        (xs, ys) = batch
        bs = xs.shape[0]
        # (h w) 2 -> (b h w) 2
        qs = self.grid.tile((bs, 1))
        # b x -> (b h w) x
        xs = xs.tile((self.grid_size, 1))
        # b h w -> (b h w) 1
        ys = ys.reshape((-1, 1))
        return qs, xs, ys


    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        qs, xs, ys = self.package_batch(batch)
        pred_ys = self.forward(qs, xs)
        train_loss = self.loss_function(pred_ys, ys)
        self.log_dict({'loss' : train_loss.item()},
                      sync_dist=True)
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        qs, xs, ys = self.package_batch(batch)
        pred_ys = self.forward(qs, xs)
        val_loss = self.loss_function(pred_ys, ys)
        self.log_dict({'val_loss' : val_loss.item()},
                      sync_dist=True)
        # vutils.save_image(pred_ys.unsqueeze(1).data,
        #                   os.path.join(self.logger.log_dir ,
        #                                "reconstructions",
        #                                f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
        #                   normalize=False,
        #                   nrow=6)
        # vutils.save_image(ys.unsqueeze(1).data,
        #                   os.path.join(self.logger.log_dir ,
        #                                "reconstructions",
        #                                f"gt_{self.logger.name}_Epoch_{self.current_epoch}.png"),
        #                   normalize=False,
        #                   nrow=6)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.inr.parameters(),
                               lr=self.lr,
                               weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma = self.sched_gamma)
        return [optimizer], [scheduler]
