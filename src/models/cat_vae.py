import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import Tensor


class CategoricalVAE(nn.Module):

    def __init__(self,
                 action_dim: int,
                 context_dim: int,
                 categorical_dim: int = 4,  # Num classes
                 hidden_dims: List = None,
                 temperature: float = 0.5,
                 anneal_rate: float = 3e-5,
                 anneal_interval: int = 100,  # every 100 batches
                 alpha: float = 30.,
                 **kwargs) -> None:
        super(CategoricalVAE, self).__init__()

        self.action_dim = action_dim
        self.context_dim = context_dim
        self.categorical_dim = categorical_dim
        self.temp = temperature
        self.min_temp = temperature
        self.anneal_rate = anneal_rate
        self.anneal_interval = anneal_interval
        self.alpha = alpha

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        tmp_dim = self.action_dim + self.context_dim

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(tmp_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                )
            )
            tmp_dim = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_z = nn.Linear(hidden_dims[-1],
                              self.categorical_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(
            self.categorical_dim + self.context_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], self.action_dim))

        self.sampling_dist = torch.distributions.OneHotCategorical(
            1. / categorical_dim * torch.ones((self.categorical_dim, 1)))

    def encode(self, input: Tensor, context: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B x (A+C)]
        :param 
        :return: (Tensor) Latent code [B x Q]
        """
        result = self.encoder(torch.column_stack((input, context)))

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        z = self.fc_z(result)
        z = z.view(-1, self.categorical_dim)
        return [z]

    def decode(self, z: Tensor, context: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x Q]
        :return: (Tensor) [B x A]
        """
        result = self.decoder_input(torch.column_stack((z, context)))
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, z: Tensor, eps: float = 1e-7) -> Tensor:
        """
        Gumbel-softmax trick to sample from Categorical Distribution
        :param z: (Tensor) Latent Codes [B x Q]
        :return: (Tensor) [B x D]
        """
        # Sample from Gumbel
        u = torch.rand_like(z)
        g = - torch.log(- torch.log(u + eps) + eps)

        # Gumbel-Softmax sample
        s = F.softmax((z + g) / self.temp, dim=-1)
        s = s.view(-1, self.categorical_dim)
        return s

    def forward(self, action: Tensor, context: Tensor, **kwargs) -> List[Tensor]:
        q = self.encode(action, context)[0]
        z = self.reparameterize(q)
        return [self.decode(z, context), action, q]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        q = args[2]

        # Convert the categorical codes into probabilities
        q_p = F.softmax(q, dim=-1)

        # Account for the minibatch samples from the dataset
        kld_weight = kwargs['kld_weight']
        batch_idx = kwargs['batch_idx']

        # Anneal the temperature at regular intervals
        if batch_idx % self.anneal_interval == 0 and self.training:
            self.temp = np.maximum(self.temp * np.exp(- self.anneal_rate * batch_idx),
                                   self.min_temp)

        recons_loss = F.mse_loss(recons, input, reduction='mean')

        # KL divergence between gumbel-softmax distribution
        eps = 1e-7

        # Entropy of the logits
        h1 = q_p * torch.log(q_p + eps)

        # Cross entropy with the categorical distribution
        h2 = q_p * np.log(1. / self.categorical_dim + eps)
        kld_loss = torch.mean(torch.sum(h1 - h2, dim=-1), dim=0)

        # kld_weight = 1.2
        loss = self.alpha * recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def generate(self, action: Tensor, context: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x (A + C)]
        :return: (Tensor) [B x A]
        """

        return self.forward(action, context)[0]
