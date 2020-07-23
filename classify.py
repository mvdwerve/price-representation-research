import torch
from torch import nn
import pandas as pd
import os
import sys
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
import argparse
from sklearn.preprocessing import RobustScaler, StandardScaler
import numpy as np
from tqdm import trange
from imblearn.over_sampling import SMOTE
import wandb
import json
import logging

LOG_FORMAT = "[%(asctime)s] [%(levelname)-8s] %(filename)24s:%(lineno)-4d | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

os.environ["WANDB_API_KEY"] = "55bfb66f97aa0be0e1b0f571ffa2c2817ce1c7ac"
#os.environ["WANDB_MODE"] = "dryrun"

# some CLI arguments
parser = argparse.ArgumentParser(description='Learn on the downstream task.')
parser.add_argument('folder', help='Folder where df-train.feather can be found.')
parser.add_argument("--target", default="movement", choices=["movement", "movement_up", "anomaly", "future_anomaly", "positive"])
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--retrain', default=False, action='store_true', help='force training (even if already exists')
parser.add_argument('--nobalance', default=False, action='store_true', help='disable SMOTE rebalancing')
args = parser.parse_args()

# set random seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

# create the downstream model folder
os.makedirs(os.path.join(args.folder, 'downstream', args.target), exist_ok=True)

# the simplest of logistic regressions
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

def df_to_Xy(df, target="movement"):
    # drop all na targets (selecting only notna targets)
    df = df[df['target_%s' % target].notna()]

    # TODO(do not hardcode the number of dimensions)
    X = df.values[:, :16].astype(np.float32)
    y = df['target_%s' % target].values.astype(np.int64)

    # if there is a representation loss, use it
    if 'loss_0' in df.columns:
        X = np.hstack([X, df['loss_0'].values.astype(np.float32).reshape(-1, 1)])

    # return only the non-na targets
    return X, y, df


def augment(df):
    def fname_to_tstamp(fname):
        try:
            return int(fname.split('-')[-5])
        except:
            print(fname)
    def fname_to_ticker(fname):
        try:
            return "-".join(fname.split('-')[1:-5])
        except:
            print('could not convert', fname)

    def nor_to_na(v):
        return np.nan if v == 0.0 else v

    df['timestamp'] = df['filename'].map(fname_to_tstamp) / 1000
    df['ticker'] = df['filename'].map(fname_to_ticker)
    df['target_positive'] = (df['profitability'].map(nor_to_na) > 1.0).astype(float)

    return df

def train_test_split(df, split=0.75):
    # calculate the quantile of the timestamp
    q = df["timestamp"].quantile(split)

    # split it into train and test
    return df_to_Xy(df[df["timestamp"] <= q], target=args.target), df_to_Xy(df[df["timestamp"] > q], target=args.target)

import cmf

# load the training data
df = pd.read_feather(os.path.join(args.folder, 'df-train.feather'))
dft = pd.read_feather(os.path.join(args.folder, 'df-test.feather'))

# split into train and validation
(X_train, y_train, df_train), (X_valid, y_valid, df_valid) = train_test_split(augment(df), split=0.75)
(X_test, y_test, df_test) = df_to_Xy(augment(dft), target=args.target)

# scale it now to a good range
scl = RobustScaler(quantile_range=(5.0, 95.0)).fit(X_train)
X_train = scl.transform(X_train)
X_valid = scl.transform(X_valid)
X_test = scl.transform(X_test)

# check if we need balancing
if not args.nobalance:
    # we want to balance...
    logging.info("performing dataset balancing...")

    # make SMOTE to resample
    sm = SMOTE()

    # we resample the training set
    X_train, y_train = sm.fit_resample(X_train, y_train)

# pick the cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create the dataloaders for tensorflow
train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(TensorDataset(torch.FloatTensor(X_valid), torch.LongTensor(y_valid)), batch_size=args.batch_size, shuffle=False)

# create the logistic regression
model = LogisticRegressionModel(X_train.shape[1], 2).to(device)

# we use binary cross entropy 
criterion = nn.CrossEntropyLoss().to(device)

# if the model already exists we leap out
if os.path.exists(os.path.join(args.folder, 'downstream', args.target, 'model.pt')) and not args.retrain:
    model.load_state_dict(torch.load(os.path.join(args.folder, 'downstream', args.target, 'model.pt')))
    logging.info("model reloaded")

# if we are either retraining or the model does not exist yet
if args.retrain or not os.path.exists(os.path.join(args.folder, 'downstream', args.target, 'model.pt')):
    logging.info("preparing for training")
    # get some meta keys
    # open the folder settings to repeat
    with open(os.path.join(args.folder, "settings.json"), "r+") as f:
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

    # convert to the run settings
    settings = {key: str(val) for key, val in vars(args).items()}
    settings.update({'upstream-%s' % key: str(val) for key, val in opt.items()})

    # initialize wandb and set the config
    wandb.init(project="representation-classifiers")
    wandb.config.update(settings)

    # dump the config file
    with open(os.path.join(wandb.run.dir, "settings.json"), "w+") as cur_file:
        cur_file.write(
            json.dumps(
                {key: str(val) for key, val in vars(args).items()},
                sort_keys=True,
                indent=2,
            )
        )

    # save the settings to wandb
    wandb.save("settings.json")

    # create the optimizer
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(args.epochs):
        # calculate the confusion matrix for the epoch 
        epoch_cm = np.zeros((2, 2))
        epoch_loss = 0

        # iterate over all training samples
        for i, (X, y) in enumerate(train_loader):
            # move to the GPU
            X, y = X.to(device), y.to(device)

            # clear gradients
            optimizer.zero_grad()

            # forward pass to get output/logits
            y_hat = model(X)

            # calculate cross-entropy loss
            loss = criterion(y_hat, y)

            # add the step loss
            epoch_loss += loss.mean().cpu().item()

            # make backward pass and update params
            loss.backward()
            optimizer.step()

            # max value is prediction
            _, predicted = torch.max(y_hat.data, 1)

            # add to confusion matrix
            epoch_cm += confusion_matrix(y.cpu().numpy(), predicted.cpu().numpy())

        # and the training confusion matrix for the epoch
        logging.info('EPOCH %d TRAINING CM (balanced)\n%s', epoch, epoch_cm / epoch_cm.sum())

        # confusion matrix
        cm = np.zeros((2, 2))
        valid_loss = 0

        # no gradient calculations please
        with torch.no_grad():
            # Iterate through test dataset
            for Xv, yv in valid_loader:
                # move to the GPU
                Xv, yv = Xv.to(device), yv.to(device)

                # forward pass
                yv_hat = model(Xv)

                # calculate validation loss
                valid_loss += criterion(yv_hat, yv).mean().cpu().item()

                # max value is prediction
                _, predicted = torch.max(yv_hat.data, 1)

                # add to the confusion matrix
                cm += confusion_matrix(yv.cpu().numpy(), predicted.cpu().numpy())

        # get all entries
        tn, fp, fn, tp = epoch_cm.ravel() / epoch_cm.sum()
        tnv, fpv, fnv, tpv = cm.ravel() / cm.sum()

        # message to wandb
        wandb.log({
            "Training Loss": epoch_loss / len(train_loader),
            "Training Accuracy": cmf.accuracy(epoch_cm),
            "Training Precision": cmf.precision(epoch_cm),
            "Training Recall": cmf.recall(epoch_cm),
            "Training TP": tp,
            "Training TN": tn,
            "Training FP": fp,
            "Training FN": fn,
            "Training samples": epoch_cm.sum(),

            "Validation Loss": valid_loss / len(valid_loader),
            "Validation Accuracy": cmf.accuracy(cm),
            "Validation Precision": cmf.precision(cm),
            "Validation Recall": cmf.recall(cm),
            "Validation TP": tpv,
            "Validation TN": tnv,
            "Validation FP": fpv,
            "Validation FN": fnv,
            "Validation samples": cm.sum(),
        })

        # also print validation loss
        logging.info('EPOCH %d VALIDATION CM\n%s', epoch, cm / cm.sum())

    # store the model state...
    logging.info("saving model to %s", os.path.join(args.folder, 'downstream', args.target, 'model.pt'))
    torch.save(model.state_dict(), os.path.join(args.folder, 'downstream', args.target, 'model.pt'))

if not args.nobalance:
    # log and leap out
    logging.info("cannot run evaluation with balanced dataset (run again with --nobalance")
    sys.exit(0)

# set the model to pure eval mode
model.eval()

# predict probability!
def predict_proba(model, X, device=device, batch_size=1024):
    # the result
    proba_full = []

    # no gradient calculations please
    with torch.no_grad():
        # Iterate through test dataset
        for i in range((X.shape[0] // batch_size) + 1):
            # grab the correct portion of x
            x = X[i*batch_size:(i+1)*batch_size, :]

            # forward pass
            y_hat = model(torch.FloatTensor(x).to(device))

            # calculate the softmax, bring to cpu and convert to numpy
            proba_full.append(nn.functional.softmax(y_hat, dim=1).cpu().numpy())

    # convert to the probabilities
    return np.vstack(proba_full)

# calculate the probabilities
y_train_proba = predict_proba(model, X_train)
y_valid_proba = predict_proba(model, X_valid)
y_test_proba = predict_proba(model, X_test)

# calculate confusion matrix
cmtr = confusion_matrix(y_train, np.argmax(y_train_proba, axis=1))
cmva = confusion_matrix(y_valid, np.argmax(y_valid_proba, axis=1))
cmte = confusion_matrix(y_test, np.argmax(y_test_proba, axis=1))

def predict_threshold(proba, r=0.5):
    return np.argmax(proba, axis=1)

from sklearn.metrics import precision_recall_curve

for (what, proba, dft) in [('train', y_train_proba, df_train), ('valid', y_valid_proba, df_valid), ('test', y_test_proba, df_test)]:
    # get the objective and check what is actually usable
    objective = dft['target_%s' % args.target]
    usable = ~np.isnan(objective)

    # the objective and probability is only the usable
    objective = objective[usable]
    proba = proba[usable]
    dftu = dft[usable]
    y_hat = np.argmax(proba, axis=1)

    # confusion matrix    
    cm = confusion_matrix(objective, y_hat)
    cm = cm / cm.sum()

    # log the confusion matrix
    logging.info("%s-%s:\n%s", what, args.target, cm)

    # the results
    results = {
        'precision': float(cmf.precision(cm)),
        'accuracy': float(cmf.accuracy(cm)),
        'recall': float(cmf.recall(cm)),
        'phi': float(cmf.phi(cm)),
        'positive': float(cmf.positive(cm))
    }

    # calculate precision recall curve
    p, r, t = precision_recall_curve(objective, proba[:, 1])

    # calculate the precision-recall curve
    results['pr'] = {
        'precision': list(p.astype(float)),
        'recall': list(r.astype(float)),
        'thresholds': list(t.astype(float))
    }

    # only on the 'positive' target
    if args.target == 'positive':
        # add the profitability of taken trades in BIPS
        results['trades'] = list(((dftu['profitability'][y_hat > 0] - 1) * 100 * 100).values.astype(float))

    tickers = {}

    # iterate over all tickers
    for ticker in dftu['ticker'].unique():
        # what to select?
        select = dftu['ticker'] == ticker

        # confusion matrix for ticker   
        cmt = confusion_matrix(objective[select], y_hat[select])
        cmt = cmt / cmt.sum()

        tickers[ticker] = {
            'precision': float(cmf.precision(cmt)),
            'accuracy': float(cmf.accuracy(cmt)),
            'recall': float(cmf.recall(cmt)),
            'phi': float(cmf.phi(cmt)),
            'positive': float(cmf.positive(cmt))
        }

    # store these tickers
    results['tickers'] = tickers

    # dump the result json
    with open(os.path.join(args.folder, 'downstream', args.target, what + ".json"), 'w+') as f:
        json.dump(results, f, sort_keys=True, indent=2)

# todo: eval per symbol
# todo: eval per day
# todo: eval per GICS
# todo: eval top 10% of repr loss
# ... 