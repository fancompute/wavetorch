import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import time
import os

from torch.utils.data import TensorDataset, DataLoader

from wavetorch import *

if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, required=True)
    argparser.add_argument('--sr', type=int, default=5000)
    argparser.add_argument('--batch_size', type=int, default=6)
    argparser.add_argument('--num_threads', type=int, default=4)
    argparser.add_argument('--pad_factor', type=float, default=1.0)
    argparser.add_argument('--test_size', type=int, default=30)
    argparser.add_argument('--cm', action='store_true')
    argparser.add_argument('--show', action='store_true')
    argparser.add_argument('--hist', action='store_true')
    argparser.add_argument('--animate', action='store_true')
    args = argparser.parse_args()

    torch.set_num_threads(args.num_threads)

    # Load the model and the training history
    model, hist_loss_batches, hist_train_acc, hist_test_acc, args_trained = load_model(args.model)

    if args.show:
        plot_c(model)

    if args.hist:
        epochs = range(1,len(hist_loss_batches)+1)
        fig, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True, figsize=(3,4))
        axs[0].plot(epochs, hist_loss_batches, "o-")
        axs[0].set_ylabel("Loss")
        axs[1].plot(epochs, hist_train_acc, "o-", label="Train")
        axs[1].plot(epochs, hist_test_acc, "o-", label="Test")
        axs[1].set_xlabel("Number of training epochs")
        axs[1].set_ylabel("Accuracy")
        axs[1].set_xticks(epochs)
        axs[1].set_ylim([0.5, 1.0])
        axs[1].legend(fontsize="smaller")
        plt.show(block=False)

    if args.cm:
        # Load the data
        directories_str = ("./data/vowels/a/",
                           "./data/vowels/e/",
                           "./data/vowels/o/")

        x_train, x_test, y_train, y_test = load_all_vowels(directories_str, sr=args_trained.sr, normalize=True, train_size=None, test_size=args.test_size, pad_factor=args_trained.pad_factor)

        # Put tensors into Datasets and then Dataloaders to let pytorch manage batching
        test_ds = TensorDataset(x_test, y_test)  
        test_dl = DataLoader(test_ds, batch_size=args.batch_size)

        cm_test = calc_cm(model, test_dl)
        fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(3.5,3.5))
        plot_cm(cm_test, title="Testing set", normalize=True, ax=axs, labels=["a", "e", "o"])
        plt.show(block=False)

    if args.animate:
        pass
        #TODO: just put animation stuff here...
