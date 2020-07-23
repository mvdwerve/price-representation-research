import random
import streambar
import tempfile
import pandas as pd
import os
import torch
import bar_utils as bu
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict
from statsmodels.tsa.stattools import adfuller

import re

import logging


def default_stock_loader(path, channels_first=True, as_df=False, cache="F:/cache"):
    cached = os.path.join(cache, re.sub("[^A-Za-z0-9]+", "-", path)) + "-time-60.csv"
    if not os.path.exists(cached):
        # parameter is the path
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # close file (otherwise it cannot be opened)
            f.close()

            # convert to bars
            streambar.time(path, f.name, size=60)

            # read back in as csv
            df = pd.read_csv(f.name)

            # and unlink the temporary file
            os.unlink(f.name)

        # fill the gaps
        df = bu.fill_timebar_gaps(df, use_tqdm=False)

        # write the cached version
        df.to_csv(cached, index=False)
    else:
        # there is a cached version, load it
        df = pd.read_csv(cached)

    # remove timestamps, skewness and kurtosis
    df.drop(columns=["first", "last", "skewness", "kurtosis"], inplace=True)
    df.dropna(inplace=True)

    # return which columns are prices as well
    isprice = np.array(
        [
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    )

    # return the dataframe values as tensor  (channels, time)
    return (torch.FloatTensor(df.values.T) if not as_df else df), isprice, cached


def iqr_mean(series, low=0.1, high=0.9):
    # filter the values
    vals = series[(series > series.quantile(low)) & (series < series.quantile(high))]
    return vals.mean()


def volume_stock_loader(path, channels_first=True, as_df=False, cache="F:/cache"):
    # load the time bars for this entry
    df_time, _, _ = default_stock_loader(
        path, channels_first=channels_first, as_df=True, cache=cache
    )

    # count the volume up and divide by 400
    per_bin = int(iqr_mean(df_time["volume"]))

    # get the cached value
    cached = os.path.join(cache, re.sub("[^A-Za-z0-9]+", "-", path)) + (
        "-volume-%d.csv" % per_bin
    )

    # logging
    logging.debug("Searching for cached file %s", cached)

    # some
    if not os.path.exists(cached):
        # cached file is not found
        logging.debug(" -- not found, generating.")

        # parameter is the path
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # close file (otherwise it cannot be opened)
            f.close()

            # convert to bars
            streambar.volume(path, f.name, size=per_bin)

            # read back in as csv
            df = pd.read_csv(f.name)

            # and unlink the temporary file
            os.unlink(f.name)

        # write the cached version
        df.to_csv(cached, index=False)
    else:
        # cached file is not found
        logging.debug(" -- found.")

        # there is a cached version, load it
        df = pd.read_csv(cached)

    # remove timestamps, skewness and kurtosis
    df.drop(columns=["first", "last", "skewness", "kurtosis"], inplace=True)
    df.dropna(inplace=True)

    # return which columns are prices as well
    isprice = np.array(
        [
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    )

    # return the dataframe values as tensor  (channels, time)
    return (torch.FloatTensor(df.values.T) if not as_df else df), isprice, cached


def dollars_stock_loader(path, channels_first=True, as_df=False, cache="F:/cache"):
    # load the time bars for this entry
    df_time, _, _ = default_stock_loader(
        path, channels_first=channels_first, as_df=True, cache=cache
    )

    # count the dollar up and take the median
    per_bin = int(iqr_mean(df_time["dollars"]))

    # try to find cached file
    cached = os.path.join(cache, re.sub("[^A-Za-z0-9]+", "-", path)) + (
        "-dollars-%d.csv" % per_bin
    )

    # logging
    logging.debug("Searching for cached file %s", cached)

    # if it does not exist, we have to make it
    if not os.path.exists(cached):
        # cached file is not found
        logging.debug(" -- not found, generating.")

        # parameter is the path
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # close file (otherwise it cannot be opened)
            f.close()

            # convert to bars
            streambar.dollar(path, f.name, size=per_bin)

            # read back in as csv
            df = pd.read_csv(f.name)

            # and unlink the temporary file
            os.unlink(f.name)

        # write the cached version
        df.to_csv(cached, index=False)
    else:
        # cached file is not found
        logging.debug(" -- found.")

        # there is a cached version, load it
        df = pd.read_csv(cached)

    # remove timestamps, skewness and kurtosis
    df.drop(columns=["first", "last", "skewness", "kurtosis"], inplace=True)
    df.dropna(inplace=True)

    # return which columns are prices as well
    isprice = np.array(
        [
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    )

    # return the dataframe values as tensor  (channels, time)
    return (torch.FloatTensor(df.values.T) if not as_df else df), isprice, cached


class StockDataset(Dataset):
    def __init__(
        self,
        opt,  # options
        flist,  # file list
        length=120,  # sequence length
        loader=default_stock_loader,
        redo_bars=False,
    ):
        # store root directory and options
        self.opt = opt

        # store the file list and the 'prev' list
        self.flist = flist["file"]
        self.plist = flist["prev"]

        # store file loader
        self.loader = loader

        # store length
        self.length = length

        # how to cache the data
        self.cache = {}

        # a self-filling dict of dicts, noting the offsets in the file
        self.profits = defaultdict(dict)

        self.mean, self.std = {}, {}

        # full mapping
        self.mapping = []
        self.mark = {}
        self.mark_raw = {}

        self.counter = defaultdict(int)

        self.adf = []

        # preload all data
        logging.info("Preloading data...")
        for i in range(len(flist)):
            self.getidx(i)
        logging.info("Done preloading.")

        # take all lengths for insight purposes
        lens = [d[0].shape[1] for d in self.cache.values()]

        # debug statement
        logging.info(
            "Loaded Dataset Number of Bars (%s): Mean %.3f and Std %.3f",
            self.opt.bar_type,
            np.mean(lens),
            np.std(lens),
        )

        print(np.mean(self.adf), np.std(self.adf))

    def getidx(self, idx):
        assert idx < len(self.flist), "index out of range"

        # if the index is not already in cache, store it
        if idx in self.cache:
            return self.cache[idx]

        # debugging info
        logging.debug("Going to load %s", self.flist[idx])

        # assign to the
        self.cache[idx] = self.loader(self.flist[idx])

        # get the data
        data, isprice, _ = self.cache[idx]

        # because prices are non-stationary we'll divide those (make logreturns sequence)
        data[isprice, :] = torch.FloatTensor(
            np.hstack(
                [
                    np.zeros_like(isprice[isprice]).reshape(-1, 1),
                    np.diff(np.log(data[isprice, :])),
                ]
            )
        )

        # calculate the profitability of getting in at time X, 4th channel is the close price!
        closes = self.cache[idx][0][3]

        # mark
        mark = np.exp(
            (
                closes[self.opt.prediction_step + self.length :]
                - closes[self.length : -self.opt.prediction_step]
            ).numpy()
        )

        # numpy range of everything
        indices = np.arange(len(mark))

        @np.vectorize
        def classify(mark):
            if mark < 0.995:
                return 0
            if mark >= 0.995 and mark <= 0.998:
                return 1
            if mark > 0.998 and mark < 1.002:
                return 2
            if mark >= 1.002 and mark <= 1.005:
                return 3
            else:
                return 4

        # store the mark
        self.mark[idx] = classify(mark)
        self.mark_raw[idx] = mark

        # subdivide
        self.profits[0][idx] = indices[self.mark[idx] == 0]
        self.profits[1][idx] = indices[self.mark[idx] == 1]
        self.profits[2][idx] = indices[self.mark[idx] == 2]
        self.profits[3][idx] = indices[self.mark[idx] == 3]
        self.profits[4][idx] = indices[self.mark[idx] == 4]

        self.counter[0] += len(self.profits[0][idx])
        self.counter[1] += len(self.profits[1][idx])
        self.counter[2] += len(self.profits[2][idx])
        self.counter[3] += len(self.profits[3][idx])
        self.counter[4] += len(self.profits[4][idx])

        if len(self.profits[0][idx]) == 0:
            del self.profits[0][idx]
        if len(self.profits[1][idx]) == 0:
            del self.profits[1][idx]
        if len(self.profits[2][idx]) == 0:
            del self.profits[2][idx]
        if len(self.profits[3][idx]) == 0:
            del self.profits[3][idx]
        if len(self.profits[4][idx]) == 0:
            del self.profits[4][idx]

        # get the data
        data, isprice, c = self.cache[idx]

        # mean for this day
        self.mean[idx] = torch.mean(data, axis=1).reshape(-1, 1)
        self.std[idx] = torch.std(data, axis=1).reshape(-1, 1)

        # use previous day, except on the first day
        previdx = max(idx - 1, 0)

        self.adf.append(adfuller(data[3, :])[0])

        # normalize the data using the previous day
        normalized = (data - self.mean[previdx]) / self.std[previdx]

        # check and fix if needed
        if (torch.max(torch.abs(normalized), axis=1).values > 100).any():
            logging.warning("HIGH NORMALIZED DATAPOINT IN %s...", self.flist[idx])
            logging.debug("mean %s std %s", self.mean[previdx], self.std[previdx])
            logging.debug("normalized %s", normalized.abs().max(axis=1))
            logging.debug("original %s %s", data.min(axis=1), data.max(axis=1))

            # we correct it during training...
            if not self.opt.nosubsampledata:
                logging.info(" -- corrected by removal")
                data = data[:, torch.max(torch.abs(normalized), axis=0).values < 100]

                # correct the mean for this day (otherwise it bleeds to the next day)
                self.mean[idx] = torch.mean(data, axis=1).reshape(-1, 1)
                self.std[idx] = torch.std(data, axis=1).reshape(-1, 1)

                # normalize the data using the previous day again
                normalized = (data - self.mean[previdx]) / self.std[previdx]

        # standardize the data on load using previous day
        self.cache[idx] = normalized, isprice, c

        # add the mapping of all possible samples
        self.mapping.extend([(idx, i) for i in range(0, data.shape[1] - self.length)])

        # we can return it (if it wasn't there, it was made by the previous statement)
        return self.cache[idx]

    def get_mark(self, idx, start_idx):
        if idx not in self.cache:
            self.getidx(idx)
        marks = self.mark[idx]
        return -1 if start_idx >= len(marks) else marks[start_idx]

    def get_mark_raw(self, idx, start_idx):
        if idx not in self.cache:
            self.getidx(idx)
        marks = self.mark_raw[idx]
        return 0.0 if start_idx >= len(marks) else marks[start_idx]

    def getprofits(self, idx, profit):
        if idx not in self.cache:
            self.getidx(idx)
        return self.profits[profit][idx]

    def get_item_at_index(self, index, start_idx):
        # get the file at the index
        data, _, fname = self.getidx(index)

        # cut the sequence
        data = data[:, start_idx : (start_idx + self.length)].clone()

        # return the data and the index
        return data, fname, self.get_mark_raw(index, start_idx), start_idx

    def __getitem__(self, index):
        if not self.opt.nosubsampledata:
            # get the file at the index
            data, _, _ = self.getidx(index)

            # pick a random number
            start_idx = random.randint(0, data.shape[1] - self.length - 1)
        else:
            # otherwise, we have a pre-selected mapping
            index, start_idx = self.mapping[index]

        # get the item at the start index
        return self.get_item_at_index(index, start_idx)

    def __len__(self):
        if not self.opt.nosubsampledata:
            return len(self.flist)
        else:
            return len(self.mapping)

    def sample_profit(self, profit):
        """get a single datapoint with future profits as in profit."""
        randidx = np.random.choice(np.array(list(self.profits[profit].keys())))
        return self.get_item_at_index(
            randidx, np.random.choice(self.profits[profit][randidx])
        )

    def get_data_by_profit(self, profit, batch_size=20):
        """
        get data by future profit id, 0, 1, 2, 3, 4.
        0 : < -2%
        1 : -2% to -0.5%
        2 : -0.5% to 0.5%
        3 : 0.5% to 2%
        4 : > 2%

        quite arbitrary classes. @todo rebalance?
        """
        assert profit >= 0 and profit <= 5, "Invalid profit enum value"

        batch = torch.zeros(batch_size, 20, self.length)
        for idx in range(batch_size):
            batch[idx, :, :], _, _, _ = self.sample_profit(profit)

        return batch


def get_dataloaders(opt):
    """creates and returns the stock dataset and dataloaders, either with
    train/val split, or train+val/test split.

    :param opt:
    :return: train_loader, train_dataset,
    test_loader, test_dataset - corresponds to validation or test set depending on opt.validate
    """
    num_workers = 1

    # find full file list, sort it (because of temporal dependence)
    train = pd.read_csv(os.path.join(opt.data_input_dir, "train.csv"))
    valid = pd.read_csv(os.path.join(opt.data_input_dir, "valid.csv"))
    test = pd.read_csv(os.path.join(opt.data_input_dir, "test.csv"))

    loader = None

    if opt.bar_type == "time":
        loader = default_stock_loader
    elif opt.bar_type == "volume":
        loader = volume_stock_loader
    elif opt.bar_type == "dollars":
        loader = dollars_stock_loader
    else:
        raise ValueError("Incorrect Bar Type!")

    if opt.validate:
        logging.info("Using Train / Val Split")

        train_dataset = StockDataset(opt, train, loader=loader)
        test_dataset = StockDataset(opt, valid, loader=loader)

    else:
        logging.info("Using Train+Val / Test Split")
        train_dataset = StockDataset(
            opt, pd.concat([train, valid], ignore_index=True), loader=loader
        )
        test_dataset = StockDataset(opt, test, loader=loader)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size_multiGPU,
        shuffle=(not opt.noshuffle),
        drop_last=True,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=opt.batch_size_multiGPU,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )

    return train_loader, train_dataset, test_loader, test_dataset
