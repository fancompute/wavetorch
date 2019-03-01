import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader

from wavetorch import *

import argparse
import time
import os

if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, required=True)
    argparser.add_argument('--sr', type=int, default=5000)
    argparser.add_argument('--ratio_train', type=float, default=0.5)
    argparser.add_argument('--batch_size', type=int, default=10)
    argparser.add_argument('--num_of_each', type=int, default=2)
    argparser.add_argument('--num_threads', type=int, default=4)
    argparser.add_argument('--pad_fact', type=float, default=1.0)
    argparser.add_argument('--cm', action='store_true')
    args = argparser.parse_args()

    torch.set_num_threads(args.num_threads)

    # Load the model and the training history
    model, hist_loss_batches, hist_train_acc, hist_test_acc = load_model(args.model)

    # Load the data
    directories_str = ("./data/vowels/a/",
                       "./data/vowels/e/",
                       "./data/vowels/o/")

    x, y_labels = load_all_vowels(directories_str, sr=args.sr, normalize=True, num_of_each=args.num_of_each)
    x = pad(x, (1, int(x.shape[1] * args.pad_fact)))
    N_samples, N_classes = y_labels.shape

    full_ds = TensorDataset(x, y_labels)
    train_ds, test_ds = random_split(full_ds, [int(args.ratio_train*N_samples), N_samples-int(args.ratio_train*N_samples)])

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)

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
        cm_train, cm_test = calc_cm(model, train_dl, test_dl)
        fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(6,3))
        plot_cm(cm_train, title="Training set", normalize=True, ax=axs[0], labels=["a", "e", "o"])
        plot_cm(cm_test, title="Testing set", normalize=True, ax=axs[1], labels=["a", "e", "o"])
        plt.show(block=False)

