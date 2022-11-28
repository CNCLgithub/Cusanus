import os
from torch import optim
import pytorch_lightning as pl
import torchvision.utils as vutils

from ensembles.pytypes import *
from ensembles.archs.vae import BaseVAE


# based off
# https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/09-normalizing-flows.html

def create_mask(n:int, xstart:int, xstop:int) -> Tensor:
    m = torch.zeros((1, n), dtype = torch.Bool)
    mid = int(n / 2)
    m[:, 1:mid] = 1
    m[:, xstart:xstop] = 1
    return m

class MergeSplit(pl.LightningModule):
    """Task of Merge or Split over z-space"""

    def __init__(self,
                 rep_dims: int = 256,
                 chunks: int = 8,
                 import_samples:int =8,
                 lr: float = 1E-4,
                 sched_gamma: float = 0.8):
        """
        Args:
            TODO
        """
        super().__init__()
        self.init_flows(rep_dims, chunks)
        self.import_samples = import_samples
        # Create prior distribution for final latent space
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def init_flows(rep_dims:int, chunks: int) -> Nothing:
        mask_dims = rep_dims / chunks / 2 # = 16
        latent_dims = rep_dims / 2
        self.blocks = nn.ModuleList([])
        for i in range(chunks):
            x_start = i * mask_dims
            x_stop = x_start + mask_dims
            esig = FC(latent_dims, latent_dims)
            ksig = FC(latent_dims, latent_dims)
            emask = create_mask(rep_dims, x_start, x_stop).to(self.device)
            # kmask is shifted
            kmask = create_mask(rep_dims, x_start + latent_dims,
                                x_stop + latent_dims).to(self.device)
            self.blocks.append(MergeBlock(esig, emask, ksig, kmask))

        

    def forward(self, a: Tensor, b: Tensor):
        # TODO: optimize ldj calculation
        ldj = torch.zeros(a.shape[0], device=self.device)
        for block in self.blocks:
            (a, b), ld = block(a, b)
            ldj += ld
        return (a, b), ldj

    def logpdf(self, a: Tensor, b: Tensor):
        """Given a batch of object pairs, return the likelihood of those.
        """
        z, ldj = self(xs)
        z = torch.cat(z, dim = 1)
        log_pz = self.prior.log_prob(z).sum(dim=list(range(z.dim())))
        return ldj + log_pz


    # TODO
    @torch.no_grad()
    def sample(self, batch_size = 1, constraint=None):
        pass
    #     """Sample a batch of images from the flow."""
    #     device = self.device
    #     # Sample latent representation from prior
    #     z_shape = [batch_size, *self.z_shape]
    #     if z_init is None:
    #         z = self.prior.sample(sample_shape=z_shape).to(device)
    #     # or condition on constraint (for full prior)
    #     else:
    #         z = z_init.to(device)

    #     # Transform z to x by inverting the flows
    #     ldj = torch.zeros(batch_size, device=device)
    #     for flow in reversed(self.flows):
    #         z, ldj = flow(z, ldj, reverse=True)
    #     return z

    def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=1e-3)
        optimizer = optim.Adam(self.parameters(),
                               lr=self.lr)
    #     scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma = self.sched_gamma)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # Normalizing flows are trained by maximum likelihood => return bpd
        xs, ys = batch
        loss = self.logpdf(xs, ys)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.bpd(batch[0])
        xs, ys = batch
        loss = self.logpdf(xs, ys)
        self.log("val_loss", loss)
        return loss

    # def test_step(self, batch, batch_idx):
    #     # Perform importance sampling during testing => estimate likelihood M times for each image
    #     samples = []
    #     for _ in range(self.import_samples):
    #         img_ll = self.logpdf(batch[0])
    #         samples.append(img_ll)
    #     img_ll = torch.stack(samples, dim=-1)

    #     # To average the probabilities, we need to go from log-space to exp, and back to log.
    #     # Logsumexp provides us a stable implementation for this
    #     img_ll = torch.logsumexp(img_ll, dim=-1) - np.log(self.import_samples)

    #     # Calculate final bpd
    #     bpd = -img_ll * np.log2(np.exp(1)) / np.prod(batch[0].shape[1:])
    #     bpd = bpd.mean()

    #     self.log("test_bpd", bpd)
