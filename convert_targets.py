import torch
import numpy as np
import dataset

# own modules
from GreedyInfoMax.stock.arg_parser import arg_parser
from GreedyInfoMax.stock.models.loss_supervised_fn import (
    Supervised_Loss,
    target_movement,
    target_up_movement,
    target_anomaly,
    target_anomaly_offset,
)
import pandas as pd
import tqdm


def evaluate(opt, loader, fname):
    # only do target movement for now
    loss_move = Supervised_Loss(opt, 1, True, target_movement)
    loss_move_up = Supervised_Loss(opt, 1, True, target_up_movement)
    loss_anomaly = Supervised_Loss(opt, 1, True, target_anomaly)
    loss_fut_anomaly = Supervised_Loss(opt, 1, True, target_anomaly_offset)

    with torch.no_grad():
        losses, meta = [], []

        # iterate over the loader
        for i, (_, filename, _, start_idx) in tqdm.tqdm(
            enumerate(loader), total=len(loader)
        ):

            # add the movements
            losses.append(
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

            # move to the cpu for stacking, and flatten
            meta.extend(zip(filename, start_idx.cpu().numpy()))

        # convert to dataframes and merge the items
        df = pd.DataFrame(data=np.vstack(losses))
        dfmeta = pd.DataFrame.from_records(meta, columns=["filename", "startidx"])
        df["filename"] = dfmeta["filename"]
        df["startidx"] = dfmeta["startidx"]
        print(df)

        # convert the columns to strings and write to a feather file
        df.columns = [str(col) for col in df.columns]
        df.to_feather(fname)


if __name__ == "__main__":
    opt = arg_parser.parse_args()

    opt.batch_size_multiGPU = 128
    opt.noshuffle = True
    opt.nosubsampledata = True

    # get datasets and dataloaders
    train_loader, train_dataset, test_loader, test_dataset = dataset.get_dataloaders(
        opt
    )

    try:
        # Train the model
        evaluate(opt, train_loader, "df-targets-train-%s.feather" % opt.bar_type)
        evaluate(opt, test_loader, "df-targets-test-%s.feather" % opt.bar_type)

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")
