import torch
import time

import logging


def val_by_InfoNCELoss(opt, model, test_loader):
    total_step = len(test_loader)

    loss_epoch = 0
    acc_total = 0.0
    starttime = time.time()

    with torch.no_grad():
        for step, (audio, filename, profit, start_idx) in enumerate(test_loader):

            model_input = audio.to(opt.device)

            loss, acc, c, z = model(model_input, filename, start_idx)
            loss = loss.mean()

            loss_epoch += loss.data.cpu().numpy()
            acc_total += acc.item()

        validation_loss = loss_epoch / total_step

        logging.info(
            "Validation Loss Model {}: Time (s): {:.1f} --- {:.4f} --- (Acc {:.4f})".format(
                0, time.time() - starttime, validation_loss, acc_total / total_step
            )
        )

    return validation_loss
