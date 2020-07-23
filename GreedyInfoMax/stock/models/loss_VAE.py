import torch.nn as nn
import torch
import numpy as np

from GreedyInfoMax.stock.models import loss
from GreedyInfoMax.stock.models import decoder


class VAE_Loss(loss.Loss):
    def __init__(self, opt, hidden_dim, encoder, calc_accuracy):
        super(VAE_Loss, self).__init__()

        self.opt = opt
        self.hidden_dim = hidden_dim
        self.enc_hidden = encoder.hidden
        self.calc_accuracy = calc_accuracy

        # layers for latent to mean conversion
        self.mu = nn.Linear(hidden_dim, self.enc_hidden)
        self.logvar = nn.Linear(hidden_dim, self.enc_hidden)

        # create a decoder from the encoder
        self.decoder = decoder.Decoder(self.enc_hidden, encoder)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_loss(self, x, z, c, filename=None, start_idx=None):
        assert (x == x).all(), "NaN in x!"
        assert (z == z).all(), "NaN in z!"
        assert (c == c).all(), "NaN in c!"

        # convert c to the mu and logvar
        mu = self.mu(c)
        logvar = self.logvar(c)

        assert (mu == mu).all(), "NaN in mu"
        assert (logvar == logvar).all(), "NaN in log variance"

        # use the reparameterization trick
        latent_sample = self.reparameterize(mu, logvar)

        assert (latent_sample == latent_sample).all(), "NaN in latent sample"

        # then, decode the newly constructed latent_sample
        recon_x = self.decoder(latent_sample.permute(0, 2, 1))

        # empty accuracies, not calculated
        cm = np.zeros((2, 2))

        # return the loss and accuracies
        return self.loss_function(recon_x, x, mu, logvar), cm

    def loss_function(self, recon_x, x, mu, logvar):
        assert (recon_x == recon_x).all(), "NaN in reconstruction!"

        # first calculate the reconstruction loss wrt x per sample
        RL = torch.pow(x - recon_x, 2).mean(axis=1).mean(axis=1)

        assert (RL == RL).all(), "NaN in reconstruction loss!"

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(axis=1).sum(axis=1)

        assert (KLD == KLD).all(), "NaN in KLD!"

        # add reconstruction loss and divergence together
        return RL + KLD
