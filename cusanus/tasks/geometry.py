import os
from torch import optim
import pytorch_lightning as pl
import torchvision.utils as vutils

from ensembles.pytypes import *
from ensembles.archs.inm import ImplicitNeuralModule

class Geometry(ImplicitNeuralDistribution):
    """Task of occupancy probability"""

    def __init__(self,
               module: ImplicitNeuralModule) -> None:
        super(Geometry, self).__init__()
        self.module = module

    def forward(self, qs: Tensor, zg: Tensor) -> Tensor:
        return self.module(qs, zg)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        sym_vector, real_gs = batch
        pred_gs = self.forward(sym_vector)
        train_loss = self.decoder.loss_function(pred_gs,
                                                real_gs,
                                                optimizer_idx=optimizer_idx,
                                                batch_idx = batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()},
                      sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        sym_vector, real_gs = batch
        pred_gs = self.forward(sym_vector)
        # print(f"pimport torch
        # print(f"ground truth shape {real_og.shape}")
        # print(f"prediction max {pred_og.max()}")
        # print(f"ground truth max {real_og.max()}")
        val_loss = self.decoder.loss_function(pred_gs,
                                              real_gs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)
       	results = pred_gs.unsqueeze(1)
        vutils.save_image(results.data,
                          os.path.join(self.logger.log_dir ,
                                       "reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=False,
                          nrow=6)
        vutils.save_image(real_gs.unsqueeze(1).data,
                          os.path.join(self.logger.log_dir ,
                                       "reconstructions",
                                       f"gt_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=False,
                          nrow=6)
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        self.sample_gs(sym_vector.device)


    def sample_gs(self, device):
        samples = self.decoder.sample(25,
                                    device).unsqueeze(1)
        sdata = samples.cpu().data
        vutils.save_image(sdata ,
                        os.path.join(self.logger.log_dir ,
                                        "samples",
                                        f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                        normalize=False,
                        nrow=5)


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
