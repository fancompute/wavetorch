import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from torch.nn.functional import pad

from . import core
from . import data
from . import viz

parser = argparse.ArgumentParser() 
subargs = parser.add_subparsers(prog='wavetorch', title="commands", dest="command") 

# Global options
args_global = argparse.ArgumentParser(add_help=False)
args_global.add_argument('--name', type=str, default=None,
                            help='Name to use when saving or loading the model file')
args_global.add_argument('--num_threads', type=int, default=4,
                            help='Number of threads to use')
args_global.add_argument('--use-cuda', action='store_true',
                            help='Use CUDA to perform computations')
###

# Data options
args_data = argparse.ArgumentParser(add_help=False)
args_data.add_argument('--sr', type=int, default=10000,
                            help='Sampling rate to use for vowel data')
args_data.add_argument('--gender', type=str, default='men',
                            help='Which gender to use for vowel data. Options are: women, men, or both')
args_data.add_argument('--vowels', type=str, nargs='*', default=['ei', 'iy', 'oa'],
                            help='Which vowel classes to run on')
###

# Simulation options
args_sim = argparse.ArgumentParser(add_help=False)
args_sim.add_argument('--binarized', action='store_true',
                            help='Binarize the distribution of wave speed between --c0 and --c1')
args_sim.add_argument('--design_region', action='store_true',
                            help='Set design region')
args_sim.add_argument('--init_rand', action='store_true',
                            help='Use a random initialization for c')
args_sim.add_argument('--c0', type=float, default=1.0,
                            help='Background wave speed')
args_sim.add_argument('--c1', type=float, default=0.9,
                            help='Second wave speed value used with --c0 when --binarized')
args_sim.add_argument('--Nx', type=int, default=140,
                            help='Number of grid cells in x-dimension of simulation domain')
args_sim.add_argument('--Ny', type=int, default=140,
                            help='Number of grid cells in y-dimension of simulation domain')
args_sim.add_argument('--dt', type=float, default=0.707,
                            help='Time step (spatial step size is determined automatically)')
args_sim.add_argument('--px', type=int, nargs='*',
                            help='Probe x-coordinates in grid cells')
args_sim.add_argument('--py', type=int, nargs='*',
                            help='Probe y-coordinates in grid cells')
args_sim.add_argument('--pd', type=int, default=30,
                            help='Spacing, in number grid cells, between probe points')
args_sim.add_argument('--src_x', type=int, default=None,
                            help='Source x-coordinate in grid cells')
args_sim.add_argument('--src_y', type=int, default=None,
                            help='Source y-coordinate in grid cells')
args_sim.add_argument('--pml_N', type=int, default=20,
                            help='PML thickness in grid cells')
args_sim.add_argument('--pml_p', type=float, default=4.0,
                            help='PML polynomial order')
args_sim.add_argument('--pml_max', type=float, default=3.0,
                            help='PML max dampening')
###

### Training moode
args_train = subargs.add_parser('train', parents=[args_global, args_data, args_sim])
args_train.add_argument('--N_epochs', type=int, default=5, 
                            help='Number of training epochs')
args_train.add_argument('--lr', type=float, default=0.001, 
                            help='Optimizer learning rate')
args_train.add_argument('--batch_size', type=int, default=3, 
                            help='Batch size used during training and testing')
args_train.add_argument('--train_size', type=int, default=3,
                            help='Size of randomly selected training set')
args_train.add_argument('--test_size', type=int, default=3,
                            help='Size of randomly selected testing set')
###

### Inference mode
args_inference = subargs.add_parser('inference', parents=[args_global, args_data])
args_inference.add_argument('--cm', action='store_true',
                            help='Plot the confusion matrix over the whole dataset')
args_inference.add_argument('--show', action='store_true',
                            help='Show the model (distribution of wave speed)')
args_inference.add_argument('--hist', action='store_true',
                            help='Plot the training history from the loaded model')
args_inference.add_argument('--fields', action='store_true',
                            help='Plot the field distrubtion for three classes, STFTs, and simulation energy')
args_inference.add_argument('--animate', action='store_true',
                            help='Animate the field for the  classes')
args_inference.add_argument('--save', action='store_true',
                            help='Save figures')
###

class WaveTorch(object):

    def __init__(self):
        args = parser.parse_args()

        if args.use_cuda and torch.cuda.is_available():
            args.dev = torch.device('cuda')
        else:
            args.dev = torch.device('cpu')

        torch.set_num_threads(args.num_threads)

        for i in vars(args):
            print('%16s = %s' % (i, vars(args)[i]))
        print('\n')

        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)

        getattr(self, args.command)(args)

    def train(self, args):
        N_classes = len(args.vowels)

        x_train, x_test, y_train, y_test = data.load_selected_vowels(
                                                args.vowels,
                                                gender=args.gender, 
                                                sr=args.sr, 
                                                normalize=True, 
                                                train_size=args.train_size, 
                                                test_size=args.test_size
                                            )

        x_train = x_train.to(args.dev)
        x_test  = x_test.to(args.dev)
        y_train = y_train.to(args.dev)
        y_test  = y_test.to(args.dev)

        train_ds = TensorDataset(x_train, y_train)
        test_ds  = TensorDataset(x_test, y_test)

        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_dl  = DataLoader(test_ds, batch_size=args.batch_size)

        ### Define model
        px, py = core.setup_probe_coords(N_classes, args.px, args.py, args.pd, args.Nx, args.Ny, args.pml_N)
        src_x, src_y = core.setup_src_coords(args.src_x, args.src_y, args.Nx, args.Ny, args.pml_N)

        if args.design_region: # Limit the design region
            design_region = torch.zeros(args.Nx, args.Ny, dtype=torch.uint8)
            design_region[src_x+5:np.min(px)-5] = 1 # For now, just hardcode this in
        else: # Let the design region be the enire non-PML area
            design_region = None

        model = core.WaveCell(args.dt, args.Nx, args.Ny, src_x, src_y, px, py, pml_N=args.pml_N, pml_p=args.pml_p, pml_max=args.pml_max, c0=args.c0, c1=args.c1, binarized=args.binarized, init_rand=args.init_rand, design_region=design_region)
        model.to(args.dev)

        ### Train
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()
        history   = core.train(model, optimizer, criterion, train_dl, test_dl, args.N_epochs, args.batch_size)
        
        ### Print confusion matrix
        cm_test  = core.calc_cm(model, test_dl)
        cm_train = core.calc_cm(model, train_dl)

        ### Save model and results
        core.save_model(model, args.name, history, args, cm_train, cm_test)

    def inference(self, args):
        if args.name is not None:
            model, history, args_trained, cm_train, cm_test = core.load_model(args.name)
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
            px, py = core.setup_probe_coords(N_classes, args.px, args.py, args.pd, args.Nx, args.Ny, args.pml_N)
            src_x, src_y = core.setup_src_coords(args.src_x, args.src_y, args.Nx, args.Ny, args.pml_N)
            model = core.WaveCell(args.dt, args.Nx, args.Ny, src_x, src_y, px, py, pml_N=args.pml_N, pml_p=args.pml_p, pml_max=args.pml_max, c0=args.c0, c1=args.c1, binarized=args.binarized, init_rand=args.init_rand)
            sr = args.sr
            gender = args.gender
            vowels = args.vowels
            train_size = args.train_size
            test_size = args.test_size

        x_train, x_test, y_train, y_test = data.load_selected_vowels(
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
            viz.plot_cm(cm_train, title="Training dataset", normalize=False, ax=axs[0], labels=vowels)
            viz.plot_cm(cm_test, title="Testing dataset", normalize=False, ax=axs[1], labels=vowels)
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

                    viz.plot_total_field(model, field_dist, yb, ax=axs_field[yb.argmax().item()])

                    for j in range(0, probe_series.shape[1]):
                        viz.plot.plot_stft_spectrum(probe_series[:,j].numpy(), sr=sr, ax=axs_conf[yb.argmax().item(), j])

                i += 1

        if args.animate:
            for xb, yb in DataLoader(train_ds, batch_size=1):
                with torch.no_grad():
                    field_dist = model(xb, probe_output=False)
                    animate_fields(model, field_dist, yb)

if __name__ == '__main__':
    WaveTorch()

