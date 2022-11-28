import os
from torch import optim
import pytorch_lightning as pl
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from ensembles.pytypes import *
from ensembles.archs.vae import BaseVAE
from ensembles.archs.decoder import Decoder
from ensembles.tasks.sym_embedding import SymEmbedding

class Graphics(pl.LightningModule):
    """Task of embedding symbolic object into z-space"""

    def __init__(self,
                 vae_model: SymEmbedding,
                 dec_model: Decoder,
                 params: dict) -> None:
        super(Graphics, self).__init__()

        vae_model.eval()
        vae_model.freeze()

        self.vae = vae_model
        self.decoder = dec_model
        self.params = params

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.vae.model.encode(input)
        return self.decoder(mu)

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
