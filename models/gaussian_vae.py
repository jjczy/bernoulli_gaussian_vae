# models/gaussian_vae.py
import torch
import torch.nn as nn
from .base_vae import BaseVAE

class GaussianVAE(BaseVAE):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=40):
        super(GaussianVAE, self).__init__(input_dim, hidden_dim, latent_dim)
        self.log_sigma = nn.Parameter(torch.zeros(1))

    def decode(self, z):
        return self.decoder(z)

    def loss_function(self, recon_x, x, mu, logvar):
        sigma = torch.exp(self.log_sigma)
        MSE = 0.5 * torch.sum((recon_x - x.view(-1, self.input_dim)).pow(2)) / (sigma**2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        log_sigma_term = self.input_dim * torch.log(sigma)
        return MSE + KLD + log_sigma_term, MSE, KLD
