import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

from GreedyInfoMax.stock.models import loss
from GreedyInfoMax.utils import utils


class InfoNCE_Loss(loss.Loss):
    def __init__(self, opt, hidden_dim, enc_hidden, calc_accuracy):
        super(InfoNCE_Loss, self).__init__()

        self.opt = opt
        self.hidden_dim = hidden_dim
        self.enc_hidden = enc_hidden
        self.neg_samples = self.opt.negative_samples
        self.calc_accuracy = calc_accuracy

        # predict self.opt.prediction_step timesteps into the future
        self.predictor = nn.Linear(
            self.hidden_dim, self.enc_hidden * self.opt.prediction_step, bias=False
        )

        self.loss = nn.LogSoftmax(dim=1)

    def get_loss(self, x, z, c, filename=None, start_idx=None):
        Wc = self.predictor(c)
        loss, accuracies = self.calc_InfoNCE_loss(Wc, z)
        return loss, accuracies

    def broadcast_batch_length(self, input_tensor):
        """broadcasts the given tensor in a consistent way, such that it can be
        applied to different inputs and keep their indexing compatible.

        :param input_tensor: tensor to be broadcasted, generally of shape B x L x C
        :return: reshaped tensor of shape (B*L) x C
        """
        assert input_tensor.size(0) == self.opt.batch_size
        assert len(input_tensor.size()) == 3

        return input_tensor.reshape(-1, input_tensor.size(2))

    def get_pos_sample_f(self, Wc_k, z_k):
        """calculate the output of the log-bilinear model for the positive
        samples, i.e. where z_k is the actual encoded future that had to be
        predicted.

        :param Wc_k: prediction of the network for the encoded future at time-step t+k (dimensions: (B*L) x C)
        :param z_k: encoded future at time-step t+k (dimensions: (B*L) x C)
        :return: f_k, output of the log-bilinear model (without exp, as this is part of the log-softmax function)
        """
        Wc_k = Wc_k.unsqueeze(1)
        z_k = z_k.unsqueeze(2)
        f_k = torch.squeeze(torch.matmul(Wc_k, z_k), 1)
        return f_k

    def get_neg_z(self, z, cur_device):
        """scramble z to retrieve negative samples, i.e. z values that should
        not be predicted by the model.

        :param z: unshuffled z as output by the model
        :return: z_neg - shuffled z to be used for negative sampling
                shuffling params rand_neg_idx, rand_offset for testing this function
        """

        if self.opt.sampling_method == 0:
            """carefully selecting negative samples, such that they never
            include positive samples; done individually for every time-step -->
            very slow."""
            offset = 1
            # generate uncorrelated negative samples by using an individual random
            # offset for every index
            rand_neg_idx = torch.arange(z.size(0), device=cur_device)

            rand_offset = (
                torch.multinomial(
                    torch.ones(z.size(0) - offset),
                    self.neg_samples * z.size(0),
                    replacement=True,
                )
                + offset
            )
            rand_offset = rand_offset.reshape(self.neg_samples, -1).to(cur_device)

            z_neg = torch.stack(
                [
                    torch.index_select(
                        z, 0, (rand_neg_idx + rand_offset[i]) % z.size(0)
                    )
                    for i in range(self.neg_samples)
                ],
                2,
            )
        elif self.opt.sampling_method == 1:
            """randomly selecting from all z values.

            can cause positive samples to be selected as negative
            samples as well (but probability is <0.1% in our
            experiments) done once for all time-steps, much faster.
            """
            z = self.broadcast_batch_length(z)
            z_neg = torch.stack(
                [
                    torch.index_select(
                        z, 0, torch.randperm(z.size(0), device=cur_device)
                    )
                    for i in range(self.neg_samples)
                ],
                2,
            )
            rand_neg_idx = None
            rand_offset = None

        elif self.opt.sampling_method == 2:
            """randomly selecting from z values within the same sequence.

            can cause positive samples to be selected as negative
            samples as well done once for all time-steps, much faster.
            """
            z_neg = []
            channel = z.size(-1)
            batch_dim = z.size(0)
            seq_len = z.size(1)

            for _ in range(self.neg_samples):
                rand_perm_index = torch.randperm(
                    batch_dim * seq_len, device=cur_device
                ).remainder_(seq_len)
                rand_perm_index = rand_perm_index.reshape(batch_dim, seq_len)
                batch_index_offset = (
                    torch.arange(0, batch_dim, device=cur_device) * seq_len
                )
                rand_perm_index += batch_index_offset[:, None]

                z_neg.append(
                    z.reshape(-1, channel)[rand_perm_index.view(-1)].reshape(
                        batch_dim, seq_len, channel
                    )
                )

            z_neg = torch.stack(z_neg, 3)

            rand_neg_idx = None
            rand_offset = None

        else:
            raise Exception("Invalid sampling_method option")

        return z_neg, rand_neg_idx, rand_offset

    def get_neg_samples_f(self, Wc_k, z_k, device, z_neg=None, k=None):
        """calculate the output of the log-bilinear model for the negative
        samples. For this, we get z_k_neg from z_k by randomly shuffling the
        indices.

        :param Wc_k: prediction of the network for the encoded future at time-step t+k (dimensions: (B*L) x C)
        :param z_k: encoded future at time-step t+k (dimensions: (B*L) x C)
        :return: f_k, output of the log-bilinear model (without exp, as this is part of the log-softmax function)
        """
        Wc_k = Wc_k.unsqueeze(1)

        if self.opt.sampling_method == 0:
            z_k_neg, _, _ = self.get_neg_z(z_k, device)

        elif self.opt.sampling_method == 1:
            """by shortening z_neg from the front, we get different negative
            samples for every prediction-step without having to re-sample; this
            might cause some correlation between the losses within a batch
            (e.g. negative samples for projecting from z_t to z_(t+k+1) and
            from z_(t+1) to z_(t+k) are the same)"""
            z_k_neg = z_neg[z_neg.size(0) - Wc_k.size(0) :, :, :]

        elif self.opt.sampling_method == 2:
            z_k_neg = z_neg[:, k:, :]
            z_k_neg = z_k_neg.reshape(-1, z_neg.size(2), z_neg.size(3))

        else:
            raise Exception("Invalid sampling_method option")

        f_k = torch.squeeze(torch.matmul(Wc_k, z_k_neg), 1)

        return f_k

    def calc_InfoNCE_loss(self, Wc, z):
        """calculate the loss based on the model outputs Wc (the prediction)
        and z (the encoded future)

        :param Wc: output of the predictor, where W are the weights for the different timesteps and
        c the latent representation (either from the autoregressor, if use_autoregressor=True,
        or from the encoder otherwise) - dimensions: (B, L, C*self.opt.prediction_step)
        :param z: encoded future - output of the encoder - dimensions: (B, L, C)
        :return: loss - average loss per sample, timesteps and prediction steps in the batch
                    accuracies - average accuracies over all samples, timesteps and predictions steps in the batch
        """
        seq_len = z.size(1)

        cur_device = utils.get_device(self.opt, Wc)

        assert (Wc == Wc).all(), "InfoNCE: NaN in Wc"
        assert (z == z).all(), "InfoNCE: NaN in z"

        cm = np.zeros((2, 2))

        if self.opt.sampling_method == 1 or self.opt.sampling_method == 2:
            z_neg, _, _ = self.get_neg_z(z, cur_device)
        else:
            z_neg = None

        # the common loss
        loss_per_sample = torch.zeros((self.opt.batch_size,), device=cur_device)

        for k in range(1, self.opt.prediction_step + 1):
            z_k = z[:, k:, :]
            Wc_k = Wc[:, :-k, (k - 1) * self.enc_hidden : k * self.enc_hidden]

            z_k = self.broadcast_batch_length(z_k)
            Wc_k = self.broadcast_batch_length(Wc_k)

            pos_samples = self.get_pos_sample_f(Wc_k, z_k)
            neg_samples = self.get_neg_samples_f(Wc_k, z_k, cur_device, z_neg, k)

            # concatenate positive and negative samples
            results = torch.cat((pos_samples, neg_samples), 1)
            loss = self.loss(results)[:, 0]
            total_samples = (seq_len - k) * self.opt.batch_size

            # normalized, both to the loss over all timesteps and over all prediction steps K
            loss_per_sample += (
                -loss.reshape(self.opt.batch_size, -1).mean(axis=1)
                / self.opt.prediction_step
            )

            # calculate accuracy
            if self.calc_accuracy:
                predicted = torch.argmax(results, 1) == 0
                cm += confusion_matrix(
                    np.ones((total_samples)), predicted.cpu().numpy(), labels=[0, 1]
                )

        return loss_per_sample, cm
