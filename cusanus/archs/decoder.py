import torch
from torch import nn
import torch.nn.functional as F
from abc import abstractmethod

from ensembles.pytypes import *
from ensembles.archs.vae import PrintLayer

class Decoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        # Build Decoder
        hidden_dims = [16, 32, 64, 64, 64, 64]
        in_channels = 1
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 16)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride = 2,
                                       padding=1),
                    PrintLayer(),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
                           # nn.Conv2d(hidden_dims[-1], out_channels= 4,
                           #           kernel_size= 3, padding= 1),
                           # #nn.LeakyReLU()
                           # nn.Sigmoid(),
                           # PrintLayer()
                           #           )

        self.final_layer = nn.Sequential(
                            nn.Conv2d(hidden_dims[-1], out_channels= 1,
                                      kernel_size= 3, padding= 1),
                            # PrintLayer(),
                            # nn.LeakyReLU(),
                            # nn.Conv2d(hidden_dims[-1], out_channels= 1,
                            #           kernel_size= 3, padding= 1),
                            # PrintLayer(),
                            nn.Sigmoid())



        # self.final_layer = nn.Sequential(
        #                     nn.ConvTranspose2d(hidden_dims[-1],
        #                                        hidden_dims[-1],
        #                                        kernel_size=3,
        #                                        stride=2,
        #                                        padding=1,
        #                                        output_padding=1),
        #                     nn.BatchNorm2d(hidden_dims[-1]),
        #                     nn.LeakyReLU(),
        #                     nn.Conv2d(hidden_dims[-1], out_channels= 3,
        #                               kernel_size= 3, padding= 1),
        #                     nn.Tanh())

    def decode(self, z: Tensor) -> Tensor:
        #result = self.decoder_input(z)
        # result=z.unsqueeze(1)
        # print(result.shape)
        # result = self.decoder(result)
        # result = result.view(z.shape[0],128,128)
        result = self.decoder_input(z)
        result = result.view(z.shape[0], -1, 4, 4)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result.squeeze()


        #result = self.final_layer(result)
        return result


    def forward(self, mu: Tensor, **kwargs) -> Tensor:
        d = self.decode(mu)
        return d

    def loss_function(self,
                      pred_og,
                      real_og,
                      **kwargs) -> dict:
        loss = F.mse_loss(pred_og, real_og)
        return {'loss': loss}

    def generate(self, z: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(z)[0]

    def sample(self,num_samples:int, current_device: int, **kwargs) -> Tensor:
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
