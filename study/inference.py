import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import time
import os

from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

from wavetorch import *

if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default=None)
    argparser.add_argument('--sr', type=int, default=5000)
    argparser.add_argument('--batch_size', type=int, default=6)
    argparser.add_argument('--num_threads', type=int, default=4)
    argparser.add_argument('--pad_factor', type=float, default=1.0)
    argparser.add_argument('--Nx', type=int, default=140)
    argparser.add_argument('--Ny', type=int, default=140)
    argparser.add_argument('--dt', type=float, default=0.707)
    argparser.add_argument('--probe_space', type=int, default=30)
    argparser.add_argument('--probe_x', type=int, default=100)
    argparser.add_argument('--probe_y', type=int, default=40)
    argparser.add_argument('--src_x', type=int, default=40)
    argparser.add_argument('--src_y', type=int, default=70)
    argparser.add_argument('--c0', type=float, default=1.0)
    argparser.add_argument('--c1', type=float, default=0.9)
    argparser.add_argument('--cm', action='store_true')
    argparser.add_argument('--show', action='store_true')
    argparser.add_argument('--hist', action='store_true')
    argparser.add_argument('--save', action='store_true')
    argparser.add_argument('--field', action='store_true')
    argparser.add_argument('--animate', action='store_true')
    argparser.add_argument('--binarized', action='store_true')
    args = argparser.parse_args()

    torch.set_num_threads(args.num_threads)

    directories_str = ("./data/vowels/a/",
                       "./data/vowels/e/",
                       "./data/vowels/o/")
    N_classes = len(directories_str)

    # Load the model and the training history
    if args.model is not None:
        model, history, args_trained = load_model(args.model)
        sr = args_trained.sr
        pad_factor = args_trained.pad_factor
    else:
        probe_x = args.probe_x
        probe_y = torch.arange(args.probe_y, args.probe_y + N_classes*args.probe_space, args.probe_space)
        model = WaveCell(args.dt, args.Nx, args.Ny, args.src_x, args.src_y, probe_x, probe_y, c0=args.c0, c1=args.c1, binarized=args.binarized)
        sr = args.sr
        pad_factor = args.pad_factor

    # Load the data
    x_train, x_test, y_train, y_test = load_all_vowels(directories_str, sr=sr, normalize=True, train_size=3, test_size=132, pad_factor=pad_factor)

    # Put tensors into Datasets and then Dataloaders to let pytorch manage batching
    test_ds = TensorDataset(x_test, y_test)  
    train_ds = TensorDataset(x_train, y_train)  
    all_dl = DataLoader(ConcatDataset([test_ds, train_ds]), batch_size=args.batch_size)

    if args.show:
        plot_c(model)
        if args.save:
            plt.savefig(os.path.splitext(args.model)[0] + '_c.png', dpi=300)

    if args.hist:
        epochs = range(1,len(history["acc_test"])+1)
        fig, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True, figsize=(3,4))
        axs[0].plot(epochs, history["loss_avg"], "o-")
        axs[0].set_ylabel("Loss")
        axs[1].plot(epochs, history["acc_train"], "o-", label="Train")
        axs[1].plot(epochs, history["acc_test"], "o-", label="Test")
        axs[1].set_xlabel("Number of training epochs")
        axs[1].set_ylabel("Accuracy")
        axs[1].set_xticks(epochs)
        axs[1].set_ylim([0.5, 1.0])
        axs[1].legend(fontsize="smaller")
        if args.save:
            fig.savefig(os.path.splitext(args.model)[0] + '_hist.png', dpi=300)
        else:
            plt.show(block=False)

    if args.cm:
        cm = calc_cm(model, all_dl)
        fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(3.5,3.5))
        plot_cm(cm, title="Testing set", normalize=True, ax=axs, labels=["a", "e", "o"])
        if args.save:
            fig.savefig(os.path.splitext(args.model)[0] + '_cm.png', dpi=300)
        else:
            plt.show(block=False)

    if args.field:
        fig, axs = plt.subplots(3,1,constrained_layout=True, figsize=(3.5*1.15, 3.5*3*model.Ny/model.Nx))
        for xb, yb in DataLoader(train_ds, batch_size=1):
            with torch.no_grad():
                fig = plot_total_field(model, model(xb, probe_output=False), yb, ax=axs[yb.argmax().item()])

    if args.animate:
        for xb, yb in DataLoader(train_ds, batch_size=1):
            with torch.no_grad():
                model_animate(model, xb, block=True, batch_ind=0, filename=None, interval=1, fps=30, bitrate=768)

