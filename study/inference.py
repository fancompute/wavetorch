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
    argparser.add_argument('--model', type=str, default=None,
                                help='Model file to load. If not specified a blank model is created')
    argparser.add_argument('--batch_size', type=int, default=3, 
                                help='Batch size used during training and testing')
    argparser.add_argument('--num_threads', type=int, default=4,
                                help='Number of threads')

    argparser.add_argument('--cm', action='store_true',
                                help='Plot the confusion matrix over the whole dataset')
    argparser.add_argument('--show', action='store_true',
                                help='Show the model (distribution of wave speed)')
    argparser.add_argument('--hist', action='store_true',
                                help='Plot the training history from the loaded model')
    argparser.add_argument('--fields', action='store_true',
                                help='Plot the field distrubtion for three classes, STFTs, and simulation energy')
    argparser.add_argument('--animate', action='store_true',
                                help='Animate the field for the  classes')
    argparser.add_argument('--save', action='store_true',
                                help='Save figures')

    # Data options
    argparser.add_argument('--sr', type=int, default=10000,
                                help='Sampling rate to use for vowel data')
    argparser.add_argument('--pad_factor', type=float, default=0.0,
                                help='Amount of zero-padding applied to vowels in units of original length. For example, a value of 1 would double the sample length')
    
    # Simulation options
    argparser.add_argument('--c0', type=float, default=1.0,
                                help='Background wave speed')
    argparser.add_argument('--c1', type=float, default=0.9,
                                help='Second wave speed value used with --c0 when --binarized')
    argparser.add_argument('--Nx', type=int, default=140,
                                help='Number of grid cells in x-dimension of simulation domain')
    argparser.add_argument('--Ny', type=int, default=140,
                                help='Number of grid cells in y-dimension of simulation domain')
    argparser.add_argument('--dt', type=float, default=0.707,
                                help='Time step (spatial step size is determined automatically)')
    argparser.add_argument('--px', type=int, nargs='*',
                                help='Probe x-coordinates in grid cells')
    argparser.add_argument('--py', type=int, nargs='*',
                                help='Probe y-coordinates in grid cells')
    argparser.add_argument('--pd', type=int, default=30,
                                help='Spacing, in number grid cells, between probe points')
    argparser.add_argument('--src_x', type=int, default=None,
                                help='Source x-coordinate in grid cells')
    argparser.add_argument('--src_y', type=int, default=None,
                                help='Source y-coordinate in grid cells')
    argparser.add_argument('--binarized', action='store_true',
                                help='Binarize the distribution of wave speed between --c0 and --c1')
    argparser.add_argument('--init_rand', action='store_true',
                                help='Use a random initialization for c')
    argparser.add_argument('--pml_N', type=int, default=20,
                                help='PML thickness in grid cells')
    argparser.add_argument('--pml_p', type=float, default=4.0,
                                help='PML polynomial order')
    argparser.add_argument('--pml_max', type=float, default=3.0,
                                help='PML max dampening')

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
        for i in vars(args_trained):
            print('%16s = %s' % (i, vars(args_trained)[i]))
        print('\n')
    else:
        px, py = setup_probe_coords(N_classes, args.px, args.py, args.pd, args.Nx, args.Ny, args.pml_N)
        src_x, src_y = setup_src_coords(args.src_x, args.src_y, args.Nx, args.Ny, args.pml_N)
        model = WaveCell(args.dt, args.Nx, args.Ny, src_x, src_y, px, py, pml_N=args.pml_N, pml_p=args.pml_p, pml_max=args.pml_max, c0=args.c0, c1=args.c1, binarized=args.binarized, init_rand=args.init_rand)
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
        axs[0].plot(epochs, history["loss_train"], "o-", label="Train")
        axs[0].plot(epochs, history["loss_test"], "o-", label="Test")
        axs[0].set_ylabel("Loss")
        axs[1].plot(epochs, history["acc_train"], "o-", label="Train")
        axs[1].plot(epochs, history["acc_test"], "o-", label="Test")
        axs[1].set_xlabel("Number of training epochs")
        axs[1].set_ylabel("Accuracy")
        axs[1].set_ylim(top=1.01)
        axs[0].legend()
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

    if args.fields:
        fig, axs = plt.subplots(3, 4, constrained_layout=True, figsize=(8,5))
        for xb, yb in DataLoader(train_ds, batch_size=1):
            with torch.no_grad():
                field_dist = model(xb, probe_output=False)
                probe_series = field_dist[0, :, model.px, model.py]
                plot_total_field(model, field_dist, yb, ax=axs[yb.argmax().item(), 0])
                plot_stft_spectrum(xb.numpy().squeeze(), sr=args.sr, ax=axs[yb.argmax().item(), 1])
                plot_stft_spectrum(probe_series[:,yb.argmax().item()].numpy(), sr=args.sr, ax=axs[yb.argmax().item(), 2])
                axs[yb.argmax().item(),3].semilogy(field_dist.squeeze().abs().pow(2).sum(dim=1).sum(dim=1).numpy())
                axs[yb.argmax().item(),3].set_xlabel("Time")

        axs[0,0].set_title("Field distribution $\int dt$")
        axs[0,1].set_title("STFT input")
        axs[0,2].set_title("STFT reciever")
        axs[0,3].set_title("Simulation energy")
        plt.show(block=False)

    if args.animate:
        fig, axs = plt.subplots(3, 4, constrained_layout=True, figsize=(6,5))
        for xb, yb in DataLoader(train_ds, batch_size=1):
            with torch.no_grad():
                field_dist = model(xb, probe_output=False)
                animate_fields(model, field_dist, yb)
