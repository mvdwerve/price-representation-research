import torch
import torch.nn as nn

from GreedyInfoMax.stock.models import (
    autoregressor,
    encoder,
    loss_InfoNCE,
    loss_VAE,
    loss_supervised_fn,
)


class IndependentModule(nn.Module):
    def __init__(
        self,
        opt,
        enc_kernel_sizes,
        enc_strides,
        enc_padding,
        enc_hidden,
        reg_hidden,
        enc_input=18,
        use_autoregressive=True,
        use_encoder=True,
        calc_accuracy=False,
    ):
        super(IndependentModule, self).__init__()

        self.opt = opt
        self.use_autoregressive = use_autoregressive
        self.use_encoder = use_encoder
        self.calc_accuracy = calc_accuracy

        assert (
            self.use_autoregressive or self.use_encoder
        ), "No parts of the model left!"

        # encoder
        if self.use_encoder:
            self.encoder = encoder.Encoder(
                input_dim=enc_input,
                hidden=enc_hidden,
                kernel_sizes=enc_kernel_sizes,
                strides=enc_strides,
                padding=enc_padding,
            )
        self.enc_hidden = enc_hidden

        # autoregressor
        if self.use_autoregressive:
            self.hidden_dim = reg_hidden
            self.autoregressor = autoregressor.Autoregressor(
                opt=opt, input_size=self.enc_hidden, hidden_dim=self.hidden_dim
            )
        else:
            self.hidden_dim = enc_hidden

        # different loss functions
        if self.opt.loss == "InfoNCE":
            self.loss = loss_InfoNCE.InfoNCE_Loss(
                opt, self.hidden_dim, self.enc_hidden, calc_accuracy
            )
        elif self.opt.loss == "BCE-Movement":
            self.loss = loss_supervised_fn.Supervised_Loss(
                opt,
                self.hidden_dim,
                calc_accuracy,
                fn=loss_supervised_fn.target_movement,
            )
        elif self.opt.loss == "BCE-Up-Movement":
            self.loss = loss_supervised_fn.Supervised_Loss(
                opt,
                self.hidden_dim,
                calc_accuracy,
                fn=loss_supervised_fn.target_up_movement,
            )
        elif self.opt.loss == "BCE-Anomaly":
            self.loss = loss_supervised_fn.Supervised_Loss(
                opt,
                self.hidden_dim,
                calc_accuracy,
                fn=loss_supervised_fn.target_anomaly,
            )
        elif self.opt.loss == "BCE-Future-Anomaly":
            self.loss = loss_supervised_fn.Supervised_Loss(
                opt,
                self.hidden_dim,
                calc_accuracy,
                fn=loss_supervised_fn.target_anomaly_offset,
            )
        elif self.opt.loss == "VAE":
            self.loss = loss_VAE.VAE_Loss(
                opt, self.hidden_dim, self.encoder, calc_accuracy
            )
        else:
            raise Exception("Invalid option passed for opt.loss")

    def get_latent_seq_len(self, input_size):
        r"""
        Returns the size of the latent representation based on an input of size input_size
        :return: size of the corresponding latent representation c, where c is the output of the autoregressor if
        use_autoregressor=True, or the output of the encoder otherwise,
        latent representation size: C and latent sequence length L
        """
        model_input = torch.zeros(input_size, device=self.opt.device)
        c, z = self.get_latents(model_input)
        return (
            c.size(2),
            c.size(1),
        )

    def get_latents(self, x, calc_autoregressive=True):
        """Calculate the latent representation of the input (using both the
        encoder and the autoregressive model)

        :param x: batch with sampled audios (dimensions: B x C x L)
        :return: c - latent representation of the input (either the output of the autoregressor,
                if use_autoregressor=True, or the output of the encoder otherwise)
                z - latent representation generated by the encoder (or x if self.use_encoder=False)
                both of dimensions: B x L x C
        """
        # encoder in and out: B x C x L, permute to be  B x L x C
        if self.use_encoder:
            z = self.encoder(x)
        else:
            z = x

        z = z.permute(0, 2, 1)

        if self.use_autoregressive and calc_autoregressive:
            c = self.autoregressor(z)
        else:
            c = z

        return c, z

    def forward(self, x, filename=None, start_idx=None):
        """combines all the operations necessary for calculating the loss and
        accuracy of the network given the input.

        :param x: batch with sampled audios (dimensions: B x C x L)
        :return: total_loss - average loss over all samples, timesteps and prediction steps in the batch
                accuracies - average accuracies over all samples, timesteps and predictions steps in the batch
                c - latent representation of the input (either the output of the autoregressor,
                if use_autoregressor=True, or the output of the encoder otherwise)
        """

        c, z = self.get_latents(x)  # B x L x C
        loss, cm = self.loss.get_loss(x, z, c, filename, start_idx)

        # for multi-GPU training
        # loss = loss.unsqueeze(0)
        # accuracies = accuracies.unsqueeze(0)

        return loss, cm, c, z
