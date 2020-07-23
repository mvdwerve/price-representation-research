import cmf
import torch
import time
import numpy as np
import dataset

import wandb
import os

from GreedyInfoMax.utils import logger
from GreedyInfoMax.stock.arg_parser import arg_parser
from GreedyInfoMax.stock.models import load_stock_model
from GreedyInfoMax.stock.validation import val_by_Loss
from GreedyInfoMax.stock.validation import val_by_latent_profits

import logging

import json

import sys

os.environ["WANDB_API_KEY"] = "55bfb66f97aa0be0e1b0f571ffa2c2817ce1c7ac"
# os.environ["WANDB_MODE"] = "dryrun"

LOG_FORMAT = "[%(asctime)s] [%(levelname)-8s] %(filename)24s:%(lineno)-4d | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def train(opt, model):
    total_step = len(train_loader)

    # how often to output training values
    print_idx = 10
    # how often to validate training process by plotting latent representations of various speakers
    latent_val_idx = 1000

    starttime = time.time()

    logged_info = False
    early_stop = 0
    min_avg_val = 1000

    for epoch in range(opt.start_epoch, opt.num_epochs + opt.start_epoch):

        loss_epoch = 0
        cm_epoch = np.zeros((2, 2))

        for step, (audio, filename, profit, start_idx) in enumerate(train_loader):
            # validate training progress by plotting latent representation of various speakers
            if step % latent_val_idx == 0 and False:
                clf = val_by_latent_profits.val_by_latent_profits(
                    opt, train_dataset, model, epoch, step
                )
                if opt.validate:
                    val_by_latent_profits.val_by_latent_profits(
                        opt, test_dataset, model, epoch, step, stage="valid", clf=clf
                    )
            if step % print_idx == 0:
                logging.info(
                    "Epoch [{}/{}], Step [{}/{}], Time (s): {:.1f}".format(
                        epoch + 1,
                        opt.num_epochs + opt.start_epoch,
                        step,
                        total_step,
                        time.time() - starttime,
                    )
                )

            starttime = time.time()

            assert (audio == audio).all(), "NaN in audio!"

            model_input = audio.to(opt.device)

            loss, cm, c, z = model(model_input, filename, start_idx)

            if loss.mean() > 1000:
                # indices of disproportionally high losses
                bad_loss = loss.squeeze() > 1000
                loss = loss * ~bad_loss

            assert (loss == loss).all(), "NaN in loss!"

            loss = (
                loss.mean()
            )  # average over the losses from different GPUs and samples

            model.zero_grad()
            loss.backward()

            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1.0 / 2)

            if total_norm > 5:
                logging.warning("CLIPPED GRADIENT NORM!!! WAS %.3f", total_norm)

            torch.nn.utils.clip_grad_value_(model.parameters(), 5)

            optimizer[0].step()

            print_loss = loss.item()
            if step % print_idx == 0:
                logging.info("\t \t Loss:   {:.4f}".format(print_loss))
                logging.info("\t \t Acc:    {:.4f}".format(cmf.accuracy(cm)))
                logging.info("\t \t Prec:   {:.4f}".format(cmf.precision(cm)))
                logging.info("\t \t Recall: {:.4f}".format(cmf.recall(cm)))

            loss_epoch += print_loss
            cm_epoch += cm

        logs.append_train_loss(loss_epoch / total_step)

        logging.info(
            "Training Loss Model {}: Time (s): {:.1f} --- {:.4f} --- (Acc {:.4f} Prec {:.4f} Rec {:.4f})".format(
                0,
                time.time() - starttime,
                loss_epoch / total_step,
                cmf.accuracy(cm_epoch),
                cmf.precision(cm_epoch),
                cmf.recall(cm_epoch),
            )
        )

        tn, fp, fn, tp = cm_epoch.ravel() / cm_epoch.sum()

        message = {
            "Training Loss": loss_epoch / total_step,
            "Training Accuracy": cmf.accuracy(cm_epoch),
            "Training Precision": cmf.precision(cm_epoch),
            "Training Recall": cmf.recall(cm_epoch),
            "Training TP": tp,
            "Training TN": tn,
            "Training FP": fp,
            "Training FN": fn,
            "Training samples": cm_epoch.sum(),
        }

        # validate by testing the CPC performance on the validation set
        if opt.validate:
            validation_loss, validation_cm = val_by_Loss.val_by_Loss(
                opt, model, test_loader
            )

            # note that we might have to stop (validation loss is not decreasing any more)
            if len(logs.val_loss) > 10:
                min_avg_val = min(np.mean(logs.val_loss[-10:]), min_avg_val)
                if min_avg_val - validation_loss < 1e-3:
                    logging.info(
                        "Validation Loss did not decrease enough - %.4f vs %.4f"
                        % (validation_loss, min_avg_val)
                    )
                    early_stop += 1
                else:
                    early_stop = 0

            tn, fp, fn, tp = (
                validation_cm.ravel() / validation_cm.sum()
                if validation_cm.sum() > 0
                else 0
            )
            logs.append_val_loss(validation_loss)
            starttime = time.time()
            message.update(
                {
                    "Validation Loss": validation_loss,
                    "Validation Accuracy": cmf.accuracy(validation_cm),
                    "Validation Precision": cmf.precision(validation_cm),
                    "Validation Recall": cmf.recall(validation_cm),
                    "Validation TP": tp,
                    "Validation TN": tn,
                    "Validation FP": fp,
                    "Validation FN": fn,
                    "Validation samples": validation_cm.sum(),
                }
            )

        # send over the message
        wandb.log(message)

        # if we did not log info yet, do so now
        if not logged_info:
            model.loss.log_info()
            logged_info = True

        logs.create_log(model, epoch=epoch, optimizer=optimizer)

        # stop it
        if early_stop >= 10:
            logging.info(
                "Early stopping hit the 1e-3 threshold in the last 10 iterations, stopping now (early)."
            )
            return


if __name__ == "__main__":
    # init the file
    wandb.init(project="price-representations", entity="mvdwerve")

    opt = arg_parser.parse_args()
    arg_parser.create_log_path(opt)

    logging.info("Finished processing CLI arguments, running %s" % " ".join(sys.argv))

    # set random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    wandb.config.update({key: str(val) for key, val in vars(opt).items()})

    # load model
    model, optimizer = load_stock_model.load_model_and_optimizer(
        opt, calc_accuracy=True
    )

    # dump the config file
    with open(os.path.join(wandb.run.dir, "settings.json"), "w+") as cur_file:
        cur_file.write(
            json.dumps(
                {key: str(val) for key, val in vars(opt).items()},
                sort_keys=True,
                indent=2,
            )
        )

    # save both files
    wandb.save("settings.json")

    # Log metrics with wandb
    wandb.watch(model)

    # initialize logger
    logs = logger.Logger(opt)

    # get datasets and dataloaders
    train_loader, train_dataset, test_loader, test_dataset = dataset.get_dataloaders(
        opt
    )

    try:
        # Train the model
        train(opt, model)

    except KeyboardInterrupt:
        logging.warning("Training got interrupted, saving log-files now.")

    logs.create_log(model)
