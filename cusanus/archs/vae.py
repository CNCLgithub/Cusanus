import torch
from torch import nn
import torch.nn.functional as F
from abc import abstractmethod

from ensembles.pytypes import *

class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    # @abstractmethod
    # def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
    #     pass

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x

class BetaVAE(BaseVAE):

    def __init__(self,
                 latent_dim: int = 256,
                 sym_dim:int = 20,
                 beta: float = 1E-4) -> None:
        super(BetaVAE, self).__init__()
        self.beta = beta
        # encoder
        self.encoder = nn.Sequential(
                nn.Linear(sym_dim, latent_dim),
                nn.LeakyReLU(),
                nn.Linear(latent_dim ,latent_dim),
                nn.LeakyReLU()
        )
        # z layer
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)
        # decoder
        self.decoder = nn.Sequential(
                nn.Linear(latent_dim,latent_dim),
                nn.LeakyReLU(),
                nn.Linear(latent_dim, sym_dim),
                nn.Tanh()
        )


    def encode(self, x: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x S]
        :return: (Tensor) List of latent codes
        """
        z = self.encoder(x)
        mu = self.fc_mu(z)
        log_var = self.fc_var(z)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        x = self.decoder(z)
        return x

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: Tensor) -> Tensor:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), x, mu, log_var]

    def sample(self,
               num_samples:int,
               current_device: int) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]
