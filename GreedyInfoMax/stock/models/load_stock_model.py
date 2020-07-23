import torch

from GreedyInfoMax.stock.models import independent_module
from GreedyInfoMax.utils import model_utils

import logging


def load_model_and_optimizer(
    opt, reload_model=False, calc_accuracy=False, num_GPU=None
):

    # original dimensions given in CPC paper (Oord et al.)
    kernel_sizes = [10, 8, 4, 4, 4]
    strides = [1, 1, 1, 1, 1]  # [5, 4, 2, 2, 2] was removed, consumed too much
    padding = [2, 2, 2, 2, 1]
    enc_hidden = opt.enc_hidden
    reg_hidden = opt.reg_hidden

    # initialize model
    model = independent_module.IndependentModule(
        opt,
        enc_kernel_sizes=kernel_sizes,
        enc_strides=strides,
        enc_padding=padding,
        enc_hidden=enc_hidden,
        reg_hidden=reg_hidden,
        calc_accuracy=calc_accuracy,
    )

    # run on only one GPU for supervised losses
    if opt.loss == 2 or opt.loss == 1:
        num_GPU = 1

    model, num_GPU = model_utils.distribute_over_GPUs(opt, model, num_GPU=num_GPU)

    """ initialize optimizers
    We need to have a separate optimizer for every individually trained part of the network
    as calling optimizer.step() would otherwise cause all parts of the network to be updated
    even when their respective gradients are zero (due to momentum)
    """
    optimizer = []
    optimizer.append(torch.optim.Adam(model.parameters(), lr=opt.learning_rate))

    model, optimizer = model_utils.reload_weights(opt, model, optimizer, reload_model)

    model.train()
    logging.info(str(model))

    return model, optimizer
