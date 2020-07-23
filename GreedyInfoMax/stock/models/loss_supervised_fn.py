import torch.nn as nn
import torch

# from GreedyInfoMax.stock.data import phone_dict
from GreedyInfoMax.stock.models import loss

import pandas as pd
import numpy as np
import logging

from sklearn.metrics import confusion_matrix


def target_movement(df, opt, col="close", T=0.005):
    # get all the values for the column
    values = df[col].values

    # get the marks
    marks = values[opt.prediction_step :] / values[: -opt.prediction_step] - 1

    # now append np.nan to the marks (missing values)
    return np.hstack([np.abs(marks) > T, [np.nan] * opt.prediction_step])


def target_up_movement(df, opt, col="close", T=0.005):
    # get all the values for the column
    values = df[col].values

    # get the marks
    marks = values[opt.prediction_step :] / values[: -opt.prediction_step] - 1

    # now append np.nan to the marks (missing values)
    return np.hstack([marks > T, [np.nan] * opt.prediction_step])


def target_anomaly(df, opt, col="close", T=2):
    # get all the values for the column
    x = df[col].values

    # we take the full prediction step
    w = opt.prediction_step * 2 + 1
    k = opt.prediction_step

    # first calculate all the point means
    Ex = np.convolve(x, np.ones(w), "valid") / w
    Ex2 = np.convolve(x ** 2, np.ones(w), "valid") / w

    # calculate the standard deviation + epsilon (prevents sqrt of negative or zero)
    s = np.mean(np.sqrt(Ex2 - Ex ** 2 + 1e-6))

    # now append np.nan to the marks (missing values)
    return np.hstack([[np.nan] * k, (x[k:-k] - Ex) / s > T, [np.nan] * k])


# @todo SEQ LEN OFFSET@!


def target_anomaly_offset(df, opt, col="close", T=2, k=12):
    # simply calculate anomaly as usual, but shift it by k places
    return np.hstack([target_anomaly(df, opt, col=col, T=T)[k:], [np.nan] * k])


def target_shift(array, shift):
    return np.hstack([array[shift:], [np.nan] * shift])


class Supervised_Loss(loss.Loss):
    def __init__(self, opt, hidden_dim, calc_accuracy, fn=target_movement):
        super(Supervised_Loss, self).__init__()

        self.opt = opt
        self.fn = fn
        self.hidden_dim = hidden_dim
        self.calc_accuracy = calc_accuracy

        # the data cache
        self._targets = {}

        # create linear classifier
        self.linear_classifier = nn.Sequential(nn.Linear(self.hidden_dim, 2)).to(
            self.opt.device
        )  # 2 different states to differentiate between (either true or false)

        # the actual loss
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([0.01, 0.99]))

        # we only want a single label per timestep
        self.label_num = 1

    def log_info(self):
        total = 0
        num = 0

        # iterate over all target files
        for v in self._targets.values():
            total += (v == 1).sum()
            num += len(v)

        logging.info(
            "Supervised Loss target %s for %s bars has positive %.3f"
            % (self.opt.loss, self.opt.bar_type, float(total) / num)
        )

    def targets(self, fname):
        # if already in there, return that
        if fname in self._targets:
            return self._targets[fname]

        # load the df
        df = pd.read_csv(fname)

        # calculate the function on the targets, and shift everything left by 120
        # places (that's the perceptive window)
        self._targets[fname] = target_shift(self.fn(df, self.opt), 120)

        assert (
            len(self._targets[fname]) == df.shape[0]
        ), "mismatch in mapping function output shape"

        # return the targets
        return self._targets[fname]

    def get_targets(self, filename, start_idx):
        # simply convert all filenames and start indices to their respective targets
        return np.array(
            [self.targets(name)[idx] for name, idx in zip(filename, start_idx)]
        )

    def get_loss(self, x, z, c, filename, start_idx):
        # forward pass
        c = c.permute(0, 2, 1)

        # average it over all timesteps
        pooled_c = nn.functional.adaptive_avg_pool1d(c, self.label_num)
        pooled_c = pooled_c.permute(0, 2, 1).reshape(-1, self.hidden_dim)

        # predict first
        Wc = self.linear_classifier(pooled_c)

        # calculate targets
        targets = self.get_targets(filename, start_idx)

        # find the 'usable' targets (not containing NaN)
        usable = ~np.isnan(targets)

        # wrap in a tensor
        valid_Wc = Wc[usable, :]
        valid_targets = torch.LongTensor(targets[usable]).to(device=self.opt.device)

        # initialize accuracies
        cm = torch.zeros(2, 2)

        # calculate accuracy
        if self.calc_accuracy:
            _, predicted = torch.max(valid_Wc, 1)
            cm = confusion_matrix(
                valid_targets.cpu().numpy(), predicted.cpu().numpy(), labels=[0, 1]
            )

        # calculate the loss
        return self.loss_fn(valid_Wc, valid_targets), cm
