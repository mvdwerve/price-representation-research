import torch
import numpy as np

from GreedyInfoMax.utils import utils


def val_by_latent_profits(opt, dataset, model, epoch, step, stage="train", clf=None):
    """Validate the training process by plotting the t-SNE representation of
    the latent space for different speakers."""
    big_feature_space = []
    max_profits = 5
    batch_size = 100

    # @todo don't hardcode this
    input_size = (opt.batch_size, 20, 120)

    model.eval()
    latent_rep_size, latent_rep_length = model.get_latent_seq_len(input_size)
    big_feature_space.append(
        np.zeros((max_profits, batch_size, latent_rep_size * latent_rep_length))
    )
    input_size = (opt.batch_size, model.enc_hidden, latent_rep_length)

    profit_labels = np.zeros((1, max_profits, batch_size))
    counter = 0

    with torch.no_grad():
        for p in range(max_profits):

            audio = dataset.get_data_by_profit(p, batch_size=batch_size)

            if audio.size(0) != batch_size:
                print("NOT ENOUGH DATA FOR PROFIT %d" % p)
                continue

            model_input = audio.to(opt.device)

            context, z = model.get_latents(model_input)
            model_input = z.permute(0, 2, 1)

            latent_rep = context.permute(0, 2, 1).cpu().numpy()
            big_feature_space[0][counter, :, :] = np.reshape(
                latent_rep, (batch_size, -1)
            )
            profit_labels[0, counter, :] = counter

            counter += 1

    if clf is None:
        clf = {}

    utils.fit_TSNE_and_plot(
        opt,
        big_feature_space[0],
        profit_labels[0],
        "tsne_{}_{}_{}_model_{}".format(stage, epoch, step, 0),
    )

    clf[0] = utils.fit_PCA_and_plot(
        opt,
        big_feature_space[0],
        profit_labels[0],
        "pca_{}_{}_{}_model_{}".format(stage, epoch, step, 0),
        clf=(clf[0] if 0 in clf else None),
    )

    model.train()

    return clf
