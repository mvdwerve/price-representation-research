import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import copy

import logging
import json


class Logger:
    def __init__(self, opt):
        self.opt = opt

        if opt.validate:
            self.val_loss = []
        else:
            self.val_loss = None

        self.train_loss = []

        if opt.start_epoch > 0:
            self.loss_last_training = np.load(
                os.path.join(opt.model_path, "train_loss.npy")
            ).tolist()
            self.train_loss[: len(self.loss_last_training)] = copy.deepcopy(
                self.loss_last_training
            )

            if opt.validate:
                self.val_loss_last_training = np.load(
                    os.path.join(opt.model_path, "val_loss.npy")
                ).tolist()
                self.val_loss[: len(self.val_loss_last_training)] = copy.deepcopy(
                    self.val_loss_last_training
                )
            else:
                self.val_loss = None
        else:
            self.loss_last_training = None

            if opt.validate:
                self.val_loss = []
            else:
                self.val_loss = None

        self.num_models_to_keep = 1
        assert self.num_models_to_keep > 0, "Dont delete all models!!!"

        # store runtime settings
        with open(os.path.join(self.opt.log_path, "settings.json"), "w+") as cur_file:
            cur_file.write(
                json.dumps(
                    {key: str(val) for key, val in vars(self.opt).items()},
                    sort_keys=True,
                    indent=2,
                )
            )

        # print cli args to cli
        logging.info(
            "CLI args: \n%s",
            json.dumps(
                {key: str(val) for key, val in vars(self.opt).items()},
                sort_keys=True,
                indent=2,
            ),
        )

    def create_log(
        self,
        model,
        accuracy=None,
        epoch=0,
        optimizer=None,
        final_test=False,
        final_loss=None,
        acc5=None,
        classification_model=None,
    ):

        logging.info("Saving model and log-file to " + self.opt.log_path)

        # Save the model checkpoint
        torch.save(
            model.state_dict(),
            os.path.join(self.opt.log_path, "model_{}.ckpt".format(epoch)),
        )

        # remove old model files to keep dir uncluttered
        if (epoch - self.num_models_to_keep) % 10 != 0:
            try:
                os.remove(
                    os.path.join(
                        self.opt.log_path,
                        "model_{}.ckpt".format(epoch - self.num_models_to_keep),
                    )
                )
            except BaseException:
                logging.info("not enough models there yet, nothing to delete")

        if classification_model is not None:
            # Save the predict model checkpoint
            torch.save(
                classification_model.state_dict(),
                os.path.join(
                    self.opt.log_path, "classification_model_{}.ckpt".format(epoch)
                ),
            )

            # remove old model files to keep dir uncluttered
            try:
                os.remove(
                    os.path.join(
                        self.opt.log_path,
                        "classification_model_{}.ckpt".format(
                            epoch - self.num_models_to_keep
                        ),
                    )
                )
            except BaseException:
                logging.info("not enough models there yet, nothing to delete")

        if optimizer is not None:
            for idx, optims in enumerate(optimizer):
                torch.save(
                    optims.state_dict(),
                    os.path.join(
                        self.opt.log_path, "optim_{}_{}.ckpt".format(idx, epoch)
                    ),
                )

                try:
                    os.remove(
                        os.path.join(
                            self.opt.log_path,
                            "optim_{}_{}.ckpt".format(
                                idx, epoch - self.num_models_to_keep
                            ),
                        )
                    )
                except BaseException:
                    logging.info("not enough models there yet, nothing to delete")

        # Save hyper-parameters
        with open(os.path.join(self.opt.log_path, "log.txt"), "w+") as cur_file:
            cur_file.write(
                json.dumps(
                    {key: str(val) for key, val in vars(self.opt).items()},
                    sort_keys=True,
                    indent=2,
                )
                + "\n\n"
            )

        # Save losses throughout training and plot
        np.save(
            os.path.join(self.opt.log_path, "train_loss"), np.array(self.train_loss)
        )

        if self.val_loss is not None:
            np.save(
                os.path.join(self.opt.log_path, "val_loss"), np.array(self.val_loss)
            )

        self.draw_loss_curve()

        if accuracy is not None:
            np.save(os.path.join(self.opt.log_path, "accuracy"), accuracy)

    def draw_loss_curve(self):
        loss = self.train_loss
        lst_iter = np.arange(len(loss))
        plt.plot(lst_iter, np.array(loss), "-b", label="train loss")
        # plt.yscale('log')

        if self.loss_last_training is not None:
            lst_iter = np.arange(len(self.loss_last_training))
            plt.plot(lst_iter, self.loss_last_training, "-g")

        if self.val_loss is not None:
            lst_iter = np.arange(len(self.val_loss))
            plt.plot(lst_iter, np.array(self.val_loss), "-r", label="val loss")

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.legend(loc="upper right")
        # plt.axis([0, max(200,len(loss)+self.opt.start_epoch), 0, -round(np.log(1/(self.opt.negative_samples+1)),1)])

        # save image
        plt.savefig(os.path.join(self.opt.log_path, "loss_{}.png".format(0)))
        plt.close()

    def append_train_loss(self, train_loss):
        self.train_loss.append(train_loss)

    def append_val_loss(self, val_loss):
        self.val_loss.append(val_loss)
