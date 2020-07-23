import torch
import time
import numpy as np
import logging


def cm_accuracy(cm):
    tn, fp, fn, tp = cm.ravel()
    return (tp + tn) / cm.sum()


def cm_precision(cm):
    tn, fp, fn, tp = cm.ravel()
    return (tp) / (tp + fp) if (tp + fp) > 0 else 0


def cm_recall(cm):
    tn, fp, fn, tp = cm.ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def val_by_Loss(opt, model, test_loader):
    total_step = len(test_loader)

    loss_epoch = 0
    cm_epoch = np.zeros((2, 2))
    starttime = time.time()

    with torch.no_grad():
        for step, (audio, filename, profit, start_idx) in enumerate(test_loader):

            model_input = audio.to(opt.device)

            loss, cm, c, z = model(model_input, filename, start_idx)
            loss = loss.mean()

            loss_epoch += loss.data.cpu().numpy()
            cm_epoch += cm

        validation_loss = loss_epoch / total_step

        logging.info(
            "Validation Loss Model {}: Time (s): {:.1f} --- {:.4f} --- (Acc {:.4f} Prec {:.4f} Rec {:.4f})".format(
                0,
                time.time() - starttime,
                loss_epoch / total_step,
                cm_accuracy(cm_epoch),
                cm_precision(cm_epoch),
                cm_recall(cm_epoch),
            )
        )

    return validation_loss, cm_epoch
