import torch
from torch import nn
import numpy as np
import dataset
import glob
import pandas as pd
import tqdm
import sys
import json
import os
import logging

# own modules
from GreedyInfoMax.stock.models import load_stock_model
from GreedyInfoMax.stock.models.loss_supervised_fn import (
    Supervised_Loss,
    target_movement,
    target_up_movement,
    target_anomaly,
    target_anomaly_offset,
)

LOG_FORMAT = "[%(asctime)s] [%(levelname)-8s] %(filename)24s:%(lineno)-4d | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def reload_settings(path):
    # open the folder settings to repeat
    with open(os.path.join(path, "settings.json"), "r+") as f:
        # load the JSON
        opt = json.load(f)

        # specify all keys
        integers = [
            "batch_size",
            "batch_size_multiGPU",
            "negative_samples",
            "num_epochs",
            "prediction_step",
            "sampling_method",
            "seed",
            "start_epoch",
            "enc_hidden",
            "reg_hidden",
        ]
        floats = ["learning_rate"]
        bools = ["noshuffle", "nosubsampledata", "validate"]

        # update the options
        opt.update({k: int(opt[k]) for k in integers})
        opt.update({k: float(opt[k]) for k in floats})
        opt.update({k: bool(opt[k]) for k in bools})

        # convert to attributes
        opt = AttrDict(**opt)

        # return the options
        return opt

def evaluate(opt, model, loader, fname):
    # only do target movement for now
    loss_move = Supervised_Loss(opt, 1, True, target_movement)
    loss_move_up = Supervised_Loss(opt, 1, True, target_up_movement)
    loss_anomaly = Supervised_Loss(opt, 1, True, target_anomaly)
    loss_fut_anomaly = Supervised_Loss(opt, 1, True, target_anomaly_offset)

    include_loss = opt.loss == "VAE" or opt.loss == "InfoNCE"

    with torch.no_grad():
        features, commons, meta, targets = [], [], [], []

        for _, (audio, filename, profitability, start_idx) in tqdm.tqdm(
            enumerate(loader), total=len(loader)
        ):
            assert (audio == audio).all(), "NaN in audio!"

            model_input = audio.to(opt.device)

            assert (model_input == model_input).all(), "NaN in input!"

            # calculate the loss
            loss, acc, c, z = model(model_input, filename, start_idx)

            # forward pass
            c = c.permute(0, 2, 1)

            # average it over all timesteps
            pooled_c = nn.functional.adaptive_avg_pool1d(c, 1)
            pooled_c = pooled_c.permute(0, 2, 1).reshape(-1, 4)

            # add to the common list
            if include_loss:
                commons.append(loss.cpu().numpy().reshape(audio.shape[0], -1))

            # move to the cpu for stacking, and flatten
            features.append(pooled_c.cpu().numpy().reshape(audio.shape[0], -1))
            meta.extend(
                zip(filename, profitability.cpu().numpy(), start_idx.cpu().numpy())
            )

            # add the targets
            targets.append(
                np.hstack(
                    [
                        loss_move.get_targets(filename, start_idx).reshape(-1, 1),
                        loss_move_up.get_targets(filename, start_idx).reshape(-1, 1),
                        loss_anomaly.get_targets(filename, start_idx).reshape(-1, 1),
                        loss_fut_anomaly.get_targets(filename, start_idx).reshape(
                            -1, 1
                        ),
                    ]
                )
            )

        # convert to dataframes and merge the items
        df = pd.DataFrame(data=np.vstack(features))
        if include_loss:
            df2 = pd.DataFrame(data=np.vstack(commons))
        df3 = pd.DataFrame(data=np.vstack(targets))
        dfmeta = pd.DataFrame.from_records(
            meta, columns=["filename", "profitability", "startidx"]
        )
        df["filename"] = dfmeta["filename"]
        df["profitability"] = dfmeta["profitability"]
        df["startidx"] = dfmeta["startidx"]

        # set columns with loss prefix and set the targets
        if include_loss:
            df2.columns = ["loss_%s" % col for col in df2.columns]
        df3.columns = [
            "target_movement",
            "target_movement_up",
            "target_anomaly",
            "target_future_anomaly",
        ]

        # merge the keys
        if include_loss:
            df = df.merge(df2, left_index=True, right_index=True)
        df = df.merge(df3, left_index=True, right_index=True)

        # convert the columns to strings and write to a feather file
        df.columns = [str(col) for col in df.columns]
        df.to_feather(os.path.join(opt.log_path, fname))


if __name__ == "__main__":
    # get th folder
    folder = sys.argv[1]

    # open the folder settings to repeat
    with open(os.path.join(folder, "settings.json"), "r+") as f:
        # load the JSON
        opt = json.load(f)

        # specify all keys
        integers = [
            "batch_size",
            "batch_size_multiGPU",
            "negative_samples",
            "num_epochs",
            "prediction_step",
            "sampling_method",
            "seed",
            "start_epoch",
            "enc_hidden",
            "reg_hidden",
        ]
        floats = ["learning_rate"]
        bools = ["noshuffle", "nosubsampledata", "validate"]

        # update the options
        opt.update({k: int(opt[k]) for k in integers})
        opt.update({k: float(opt[k]) for k in floats})
        opt.update({k: bool(opt[k]) for k in bools})

        # convert to attributes
        opt = AttrDict(**opt)

    model_num = 0
    for models in glob.glob(os.path.join(opt.log_path, "model_*.ckpt")):
        # get the basename
        base = os.path.basename(models)
        model_num = max(model_num, int(base[6:-5]))
    logging.info("BEST MODEL NUM FOUND IS %d", model_num)

    # set reload arguments (some fixes)
    opt.model_num = model_num
    opt.device = torch.device(opt.device)
    opt.model_path = os.path.join(folder, opt.model_path)

    # we are going to convert ALL data, so train + test and no shuffling / subsampling
    opt.validate = False
    opt.noshuffle = True
    opt.nosubsampledata = True

    # larger batch size is a lot faster
    opt.batch_size = 256

    # set up the handler
    filehandler = logging.FileHandler(os.path.join(opt.log_path, "convert-stdout.txt"))
    formatter = logging.Formatter(LOG_FORMAT)
    filehandler.setFormatter(formatter)

    # add a 'logfile' handler now!
    logging.getLogger().addHandler(filehandler)

    # set random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # load model
    model, optimizer = load_stock_model.load_model_and_optimizer(opt, reload_model=True)

    # get datasets and dataloaders
    train_loader, train_dataset, test_loader, test_dataset = dataset.get_dataloaders(
        opt
    )

    try:
        # Train the model
        evaluate(opt, model, train_loader, "df-train.feather")
        evaluate(opt, model, test_loader, "df-test.feather")

    except KeyboardInterrupt:
        logging.info("Training got interrupted, saving log-files now.")
