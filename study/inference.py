import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from torch.nn.functional import pad

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
    args = argparser.parse_args()

    torch.set_num_threads(args.num_threads)

    # Load the model
    model = torch.load(args.model)

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

    cm_train, cm_test = calc_cm(model, train_dl, test_dl)

    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(6,3))
    plot_cm(cm_train, title="Training", normalize=True, ax=axs[0], labels=["a", "e", "o"])
    print(cm_train)
    plot_cm(cm_test, title="Validation", normalize=True, ax=axs[1], labels=["a", "e", "o"])
    print(cm_test)
    plt.show(block=False)
    fig.savefig("cm.svg")
