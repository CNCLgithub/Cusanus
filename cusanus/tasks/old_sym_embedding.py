import os
from torch import optim
import pytorch_lightning as pl
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from ensembles.pytypes import *
from ensembles.archs.vae import BaseVAE

class SymEmbedding(pl.LightningModule):
    """Task of embedding symbolic object into z-space"""

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(SymEmbedding, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, dk: Tensor) -> Tensor:
        return self.model(dk)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        dk= batch[0]
        results = self.forward(dk)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        #dk, ogs = batch
        dk = batch[0]
        print("validation step")
        print(type(self.forward(dk)))
        results = self.forward(dk)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)


        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)



       	results = results[1].unsqueeze(1)
        vutils.save_image(results.data,
                          os.path.join(self.logger.log_dir ,
                                       "reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=False,
                          nrow=6)
        vutils.save_image(ogs.unsqueeze(1).data,
                          os.path.join(self.logger.log_dir ,
                                       "reconstructions",
                                       f"gt_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=False,
                          nrow=6)
        #self.sample_ogs(real_img.device)


    # def sample_images(self, device):
    #     samples = self.model.sample(25,
    #                                 device)
    #
    #     vutils.save_image(samples.cpu().data,
    #                     os.path.join(self.logger.log_dir ,
    #                                     "samples",
    #                                     f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
    #                     normalize=True,
    #                     nrow=12)

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        if self.params['scheduler_garesultmma'] is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                         gamma = self.params['scheduler_gamma'])
            scheds.append(scheduler)

        return optims, scheds
