# models/bernoulli_vae.py
import torch
import torch.nn.functional as F
from .base_vae import BaseVAE

class BernoulliVAE(BaseVAE):
    def decode(self, z):
        h = self.decoder(z)
        return torch.sigmoid(h)

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.input_dim), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD, BCE, KLD
