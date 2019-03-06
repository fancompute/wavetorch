import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import time
import os

from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

import wavetorch

if __name__ == '__main__':
    args = wavetorch.options.parse_args()

    torch.set_num_threads(args.num_threads)

    # Load the model and the training history
    if args.name is not None:
        model, history, args_trained, cm_train, cm_test = wavetorch.utils.load_model(args.name)
        N_classes = len(args_trained.vowels)
        sr = args_trained.sr
        gender = args_trained.gender
        vowels = args_trained.vowels
        train_size = args_trained.train_size
        test_size = args_trained.test_size
        for i in vars(args_trained):
            print('%16s = %s' % (i, vars(args_trained)[i]))
        print('\n')
    else:
        N_classes = len(args.vowels)
        px, py = wavetorch.utils.setup_probe_coords(N_classes, args.px, args.py, args.pd, args.Nx, args.Ny, args.pml_N)
        src_x, src_y = wavetorch.utils.setup_src_coords(args.src_x, args.src_y, args.Nx, args.Ny, args.pml_N)
        model = wavetorch.utils.WaveCell(args.dt, args.Nx, args.Ny, src_x, src_y, px, py, pml_N=args.pml_N, pml_p=args.pml_p, pml_max=args.pml_max, c0=args.c0, c1=args.c1, binarized=args.binarized, init_rand=args.init_rand)
        sr = args.sr
        gender = args.gender
        vowels = args.vowels
        train_size = args.train_size
        test_size = args.test_size

    x_train, x_test, y_train, y_test = wavetorch.data.load_selected_vowels(
                                            vowels,
                                            gender=gender, 
                                            sr=sr, 
                                            normalize=True, 
                                            train_size=N_classes, 
                                            test_size=N_classes
                                        )

    # Put tensors into Datasets and then Dataloaders to let pytorch manage batching
    test_ds = TensorDataset(x_test, y_test)  

    if args.show:
        wavetorch.plot.plot_c(model)
        if args.save:
            plt.savefig(os.path.splitext(args.model)[0] + '_c.png', dpi=300)

    if args.hist:
        epochs = range(0,len(history["acc_test"]))
        fig, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True, figsize=(3,4))
        axs[0].plot(epochs, history["loss_train"], "o-", label="Training dataset")
        axs[0].plot(epochs, history["loss_test"], "o-", label="Testing dataset")
        axs[0].set_ylabel("Loss")
        axs[1].plot(epochs, history["acc_train"], "o-", label="Training dataset")
        axs[1].plot(epochs, history["acc_test"], "o-", label="Testing dataset")
        axs[1].set_xlabel("Number of training epochs")
        axs[1].set_ylabel("Accuracy")
        axs[1].set_ylim(top=1.01)
        axs[0].legend()
        if args.save:
            fig.savefig(os.path.splitext(args.model)[0] + '_hist.png', dpi=300)
        else:
            plt.show(block=False)

    if args.cm:
        fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(4,2))
        wavetorch.plot.plot_cm(cm_train, title="Training dataset", normalize=False, ax=axs[0], labels=vowels)
        wavetorch.plot.plot_cm(cm_test, title="Testing dataset", normalize=False, ax=axs[1], labels=vowels)
        if args.save:
            fig.savefig(os.path.splitext(args.model)[0] + '_cm.png', dpi=300)
        else:
            plt.show(block=False)

    if args.fields:
        fig_conf, axs_conf = plt.subplots(N_classes, N_classes, constrained_layout=True, figsize=(5,5), sharex=True, sharey=True)
        fig_field, axs_field = plt.subplots(N_classes, 1, constrained_layout=True, figsize=(5,5))
        axs_field = axs_field.ravel()

        i=0
        for xb, yb in DataLoader(test_ds, batch_size=1):
            with torch.no_grad():
                field_dist = model(xb, probe_output=False)
                probe_series = field_dist[0, :, model.px, model.py]

                wavetorch.plot.plot_total_field(model, field_dist, yb, ax=axs_field[yb.argmax().item()])

                for j in range(0, probe_series.shape[1]):
                    wavetorch.plot.plot_stft_spectrum(probe_series[:,j].numpy(), sr=sr, ax=axs_conf[yb.argmax().item(), j])

            i += 1

    if args.animate:
        for xb, yb in DataLoader(train_ds, batch_size=1):
            with torch.no_grad():
                field_dist = model(xb, probe_output=False)
                animate_fields(model, field_dist, yb)
