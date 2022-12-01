import os
from torch import optim
import pytorch_lightning as pl
import torchvision.utils as vutils

from ensembles.pytypes import *
from ensembles.archs.inm import ImplicitNeuralModule

class OccupancyField(pl.LightningModule):
    """Implements an occupancy field distribution

    Arguments:
        inr: ImplicitNeuralModule, INR architecture
        lr: float = 0.001, learning rate
        weight_decay: float = 0.001
        sched_gamma: float = 0.8
    """

    def __init__(self,
                 inr: ImplicitNeuralModule,
                 lr:float = 0.001,
                 weight_decay:float = 0.001,
                 sched_gamma:float = 0.8) -> None:
        super().__init__()
        self.inr = inr
        self.lr = lr
        self.weight_decay = weight_decay
        self.sched_gamma = sched_gamma

    # TODO
    def batch_forward(self, qs: Tensor, zs: Tensor) -> Tensor:
        """ Applies each z in zs to each q in qs """
        # (nz) -> (b nz)
        bqs = qs.tile((zs.shape[0], 1))
        bzs = zs.tile((qs.shape[0], 1))
        # Need to decibe whether to broadcast all calls in a single batch
        # or iterate across zs
        pass

    def run_model(self, qs: Tensor, z: Tensor) -> Tensor:
        return self.inr(qs, z)

    def inner_loop(self, qs: Tensor, ys: Tensor):
        m = self.initialize_modulation()
        m.train()
        inner_opt = self.initiliaze_inner_opt(m)
        for _ in range(self.hparams.num_inner_steps):
            pred_ys = self.run_model(qs, m)
            loss = self.inner_loss(ys, pred_ys)
            # Update modulator via SGD
            loss.backward()
            inner_opt.step()
            inner_opt.zero_grad()

        return m, loss

    def outer_loop(self, batch, mode="train"):
        accuracies = []
        losses = []

        # each trial in the batch is a group of queries
        qs, ys = batch
        # fitting modulations for current generation
        ms, ls = vmap(inner_loop)(qs, ys)
        loss = ls.mean() # average across batch
        # Update theta
        self.outer_opt.zero_grad()
        loss.backward()
        self.outer_optim.step()
        self.outer_optim.zero_grad()
        return ms, loss

    def inner_loss(self, pred_ys: Tensor, ys: Tensor):
        return F.mse_loss(pred_ys, ys)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        _, train_loss = self.outer_loop(batch, mode = "train")
        self.log_dict({'loss' : train_loss.item()}, sync_dist=True)
        return train_loss

    # TODO
    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
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
        pass


    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.decoder.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        if self.params['scheduler_gamma'] is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                         gamma = self.params['scheduler_gamma'])
            scheds.append(scheduler)

        return optims, scheds
