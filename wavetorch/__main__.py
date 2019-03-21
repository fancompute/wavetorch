import argparse
import yaml
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from torch.nn.functional import pad

from sklearn.model_selection import StratifiedKFold

import librosa
import librosa.display

from . import core
from . import data
from . import viz

import os

parser = argparse.ArgumentParser() 
subargs = parser.add_subparsers(prog='wavetorch', title="commands", dest="command") 

# Global options
args_global = argparse.ArgumentParser(add_help=False)
args_global.add_argument('--num_threads', type=int, default=4,
                            help='Number of threads to use')
args_global.add_argument('--use-cuda', action='store_true',
                            help='Use CUDA to perform computations')

### Training mode
args_train = subargs.add_parser('train', parents=[args_global])
args_train.add_argument('config', type=str, 
                            help='Configuration file for geometry, training, and data preparation')
args_train.add_argument('--name', type=str, default=None,
                            help='Name to use when saving or loading the model file. If not specified when saving a time and date stamp is used')
args_train.add_argument('--savedir', type=str, default='./study/',
                            help='Directory in which the model file is saved. Defaults to ./study/')

### Analysis modes
args_summary = subargs.add_parser('summary', parents=[args_global])
args_summary.add_argument('--vmin', type=float, default=1e-3)
args_summary.add_argument('--vmax', type=float, default=1.0)
args_summary.add_argument('--fig', type=str, default=None)
args_summary.add_argument('--title_off', action='store_true')
args_summary.add_argument('filename', type=str)

args_fields = subargs.add_parser('fields', parents=[args_global])
args_fields.add_argument('--vmin', type=float, default=1e-3)
args_fields.add_argument('filename', type=str)

args_stft = subargs.add_parser('stft', parents=[args_global])
args_stft.add_argument('filename', type=str)

args_animate = subargs.add_parser('animate', parents=[args_global])
args_animate.add_argument('filename', type=str)

class WaveTorch(object):

    def __init__(self):
        args = parser.parse_args()

        if args.use_cuda and torch.cuda.is_available():
            args.dev = torch.device('cuda')
        else:
            args.dev = torch.device('cpu')

        torch.set_num_threads(args.num_threads)

        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)

        getattr(self, args.command)(args)

    def train(self, args):
        print("Using configuration from %s: " % args.config)
        with open(args.config, 'r') as ymlfile:
             cfg = yaml.load(ymlfile)
             print(yaml.dump(cfg, default_flow_style=False))

        N_classes = len(cfg['data']['vowels'])

        X, Y = data.load_all_vowels(
                    cfg['data']['vowels'],
                    gender=cfg['data']['gender'], 
                    sr=cfg['data']['sr'], 
                    normalize=True,
                    max_samples=cfg['training']['max_samples']
                    )

        skf = StratifiedKFold(n_splits=cfg['training']['train_test_divide'], random_state=None, shuffle=True)
        samps = [y.argmax().item() for y in Y]
        num = 1
        for train_index, test_index in skf.split(np.zeros(len(samps)), samps):
            if cfg['training']['use_cross_validation']: print("Cross validation fold #%d" % num)

            x_train = torch.nn.utils.rnn.pad_sequence([X[i] for i in train_index], batch_first=True)
            x_test = torch.nn.utils.rnn.pad_sequence([X[i] for i in test_index], batch_first=True)
            y_train = torch.nn.utils.rnn.pad_sequence([Y[i] for i in train_index], batch_first=True)
            y_test = torch.nn.utils.rnn.pad_sequence([Y[i] for i in test_index], batch_first=True)

            x_train = x_train.to(args.dev)
            x_test  = x_test.to(args.dev)
            y_train = y_train.to(args.dev)
            y_test  = y_test.to(args.dev)

            train_ds = TensorDataset(x_train, y_train)
            test_ds  = TensorDataset(x_test, y_test)

            train_dl = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True)
            test_dl  = DataLoader(test_ds, batch_size=cfg['training']['batch_size'])

            ### Define model
            px, py = core.setup_probe_coords(
                                N_classes, cfg['geom']['px'], cfg['geom']['py'], cfg['geom']['pd'], 
                                cfg['geom']['Nx'], cfg['geom']['Ny'], cfg['geom']['pml']['N']
                                )
            src_x, src_y = core.setup_src_coords(
                                cfg['geom']['src_x'], cfg['geom']['src_y'], cfg['geom']['Nx'],
                                cfg['geom']['Ny'], cfg['geom']['pml']['N']
                                )

            if cfg['geom']['use_design_region']: # Limit the design region
                design_region = torch.zeros(cfg['geom']['Nx'], cfg['geom']['Ny'], dtype=torch.uint8)
                design_region[src_x+5:np.min(px)-5] = 1 # For now, just hardcode this in
            else: # Let the design region be the enire non-PML area
                design_region = None

            model = core.WaveCell(
                        cfg['geom']['dt'], cfg['geom']['Nx'], cfg['geom']['Ny'], src_x, src_y, px, py,
                        pml_N=cfg['geom']['pml']['N'], pml_p=cfg['geom']['pml']['p'], pml_max=cfg['geom']['pml']['max'], 
                        c0=cfg['geom']['c0'], c1=cfg['geom']['c1'], eta=cfg['geom']['binarization']['eta'], beta=cfg['geom']['binarization']['beta'], 
                        init_rand=cfg['geom']['use_rand_init'], design_region=design_region,
                        nl_b0=cfg['geom']['nonlinearity']['b0'], nl_uth=cfg['geom']['nonlinearity']['uth'],
                        nl_c=cfg['geom']['nonlinearity']['cnl'] 
                        )
            model.to(args.dev)

            ### Train
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])
            criterion = torch.nn.CrossEntropyLoss()
            history   = core.train(model, optimizer, criterion, train_dl, test_dl, cfg['training']['N_epochs'], cfg['training']['batch_size'])
            
            ### Print confusion matrix
            cm_test  = core.calc_cm(model, test_dl)
            cm_train = core.calc_cm(model, train_dl)

            ### Save model and results
            if args.name is None:
                args.name = time.strftime("%Y_%m_%d-%H_%M_%S")
            if cfg['training']['prefix'] is not None:
                args.name = cfg['training']['prefix'] + '_' + args.name

            if cfg['training']['use_cross_validation']:
                # If we are doing cross validation, then save this model's iteration
                core.save_model(model, args.name + "_cv" + str(num), args.savedir, history, cfg, cm_train, cm_test)
                num += 1
            else:
                # If not doing cross validation, save and finish
                core.save_model(model, args.name, args.savedir, history, cfg, cm_train, cm_test)
                break

    def summary(self, args):
        model, history, cfg, cm_train, cm_test = core.load_model(args.filename)

        print("Configuration for model in %s is:" % args.filename)
        print(yaml.dump(cfg, default_flow_style=False))

        sr = cfg['data']['sr']
        gender = cfg['data']['gender']
        vowels = cfg['data']['vowels']
        N_classes = len(vowels)

        fig = plt.figure( figsize=(7, 4.75), constrained_layout=True)

        gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.35])
        gs_left  = gs[0].subgridspec(3, 2, width_ratios=[1, 0.5], height_ratios=[1, 0.7, 0.7])
        gs_right = gs[1].subgridspec(N_classes+1, 1, height_ratios=[0.05] + [1 for i in range(0,N_classes)])

        ax_c = fig.add_subplot(gs_left[0,:])

        ax_loss = fig.add_subplot(gs_left[1,0])
        ax_acc = fig.add_subplot(gs_left[2,0])

        ax_cm1 = fig.add_subplot(gs_left[1,1])
        ax_cm2 = fig.add_subplot(gs_left[2,1],sharex=ax_cm1)

        ax_fields = [fig.add_subplot(gs_right[i]) for i in range(0, N_classes+1)] 

        epochs = range(0,len(history["acc_test"]))
        ax_loss.plot(epochs, history["loss_train"], "o-", label="Training dataset", ms=4, color="#1f77b4")
        ax_loss.plot(epochs, history["loss_test"], "o-", label="Testing dataset", ms=4, color="#2ca02c")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_xlabel("Training epoch #")
        ltrain,=ax_acc.plot(epochs, history["acc_train"], "o-", label="Training dataset", ms=4, color="#1f77b4")
        ltest, =ax_acc.plot(epochs, history["acc_test"], "o-", label="Testing dataset", ms=4, color="#2ca02c")
        ax_acc.set_xlabel("Training epoch #")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(top=1.01)
        ax_loss.legend()

        ax_acc.annotate("%.1f%% testing set accuracy" % (history["acc_test"][-1]*100), xy=(0.1,0.1), xycoords="axes fraction", ha="left", va="bottom", color=ltest.get_color())
        ax_acc.annotate("%.1f%% training set accuracy" % (history["acc_train"][-1]*100), xy=(0.1,0.1), xytext=(0,10), textcoords="offset points",  xycoords="axes fraction", ha="left", va="bottom", color=ltrain.get_color())

        viz.plot_structure(model, ax=ax_c, quantity='c', vowels=vowels, cbar=True)
        if not args.title_off:
            ax_c.set_title("$b_0$: %.2f / $u_{th}$: %.2f / lr: %.0e" % (cfg['geom']['nonlinearity']['b0'], cfg['geom']['nonlinearity']['uth'], cfg['training']['lr']))

        viz.plot_confusion_matrix(cm_train, title="Training dataset", normalize=False, ax=ax_cm1, labels=vowels)
        viz.plot_confusion_matrix(cm_test, title="Testing dataset", normalize=False, ax=ax_cm2, labels=vowels)

        X, Y = data.load_all_vowels(
                        vowels,
                        gender='men', 
                        sr=sr, 
                        normalize=True, 
                        max_samples=N_classes
                    )
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        Y = torch.nn.utils.rnn.pad_sequence(Y, batch_first=True)
        test_ds = TensorDataset(X, Y)


        for xb, yb in DataLoader(test_ds, batch_size=1):
            with torch.no_grad():
                field_dist = model(xb, probe_output=False)
                probe_series = field_dist[0, :, model.px, model.py]
                viz.plot_total_field(model, field_dist, yb, ax=ax_fields[1+yb.argmax().item()], cbar=True, cax=ax_fields[0], vmin=args.vmin, vmax=args.vmax)

        viz.apply_sublabels([ax_c, ax_loss, ax_acc, ax_cm1, ax_cm2] + ax_fields[1::], x=[-30, -35, -35, -35, -35, -15, -15, -15])

        plt.show()
        if args.fig is not None:
            fig.savefig(args.fig, dpi=300)
        else:
            fig.savefig(os.path.splitext(args.filename)[0]+"_summary.png", dpi=300)

    def fields(self, args):
        model, history, cfg, cm_train, cm_test = core.load_model(args.filename)

        print("Configuration for model in %s is:" % args.filename)
        print(yaml.dump(cfg, default_flow_style=False))

        sr = cfg['data']['sr']
        gender = cfg['data']['gender']
        vowels = cfg['data']['vowels']
        N_classes = len(vowels)

        X, Y = data.load_all_vowels(
                                vowels,
                                gender='men', 
                                sr=sr, 
                                normalize=True, 
                                max_samples=N_classes
                            )
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        Y = torch.nn.utils.rnn.pad_sequence(Y, batch_first=True)
        test_ds = TensorDataset(X, Y)  
        fig, axs = plt.subplots(1, N_classes, figsize=(7, 2), constrained_layout=True)

        for xb, yb in DataLoader(test_ds, batch_size=1):
            with torch.no_grad():
                field_dist = model(xb, probe_output=False)
                probe_series = field_dist[0, :, model.px, model.py]
                viz.plot_total_field(model, field_dist, yb, ax=axs[yb.argmax().item()], cbar=True, vmin=args.vmin)

        plt.show()
        fig.savefig(os.path.splitext(args.filename)[0]+"_fields.png", dpi=300)

    def stft(self, args):
        model, history, cfg, cm_train, cm_test = core.load_model(args.filename)

        print("Configuration for model in %s is:" % args.filename)
        print(yaml.dump(cfg, default_flow_style=False))

        sr = cfg['data']['sr']
        gender = cfg['data']['gender']
        vowels = cfg['data']['vowels']
        N_classes = len(vowels)

        X, Y = data.load_all_vowels(
                                vowels,
                                gender='men', 
                                sr=sr, 
                                normalize=True, 
                                max_samples=N_classes
                            )
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        Y = torch.nn.utils.rnn.pad_sequence(Y, batch_first=True)
        test_ds = TensorDataset(X, Y)  

        fig, axs = plt.subplots(N_classes, N_classes+1, constrained_layout=True, figsize=(5.5*(N_classes+1)/N_classes,5.5), sharex=True, sharey=True)

        for xb, yb in DataLoader(test_ds, batch_size=1):
            with torch.no_grad():
                i = yb.argmax().item()
                ax = axs[i, 0]

                field_dist = model(xb, probe_output=False)
                probe_series = field_dist[0, :, model.px, model.py]

                input_stft = np.abs(librosa.stft(xb.numpy().squeeze(), n_fft=256))

                librosa.display.specshow(
                    librosa.amplitude_to_db(input_stft,ref=np.max(input_stft)),
                    sr=sr,
                    vmax=0,
                    ax=ax,
                    vmin=-50,
                    y_axis='linear',
                    x_axis='time',
                    cmap=plt.cm.inferno
                )
                ax.set_ylim([0,sr/2])
                if i == 0:
                    ax.set_title("input signal", weight="bold")

                for j in range(1, probe_series.shape[1]+1):
                    ax = axs[i, j]
                    
                    output_stft = np.abs(librosa.stft(probe_series[:,j-1].numpy(), n_fft=256))

                    librosa.display.specshow(
                        librosa.amplitude_to_db(output_stft,ref=np.max(input_stft)),
                        sr=sr,
                        vmax=0,
                        ax=ax,
                        vmin=-50,
                        y_axis='linear',
                        x_axis='time',
                        cmap=plt.cm.inferno
                    )
                    ax.set_ylim([0,sr/2])

                    if i == 0:
                        ax.set_title("probe %d" % (j), weight="bold")
                    if j == N_classes:
                        ax.text(1.05, 0.5, vowels[i], transform=ax.transAxes, ha="left", va="center", fontsize="large", rotation=-90, weight="bold")
                    
                    if j > 0:
                        ax.set_ylabel('')
                    if i < N_classes-1:
                        ax.set_xlabel('')
                    # if i == j:
                        # ax.text(0.5, 0.95, '%s at probe #%d' % (vowels[i], j+1), color="w", transform=ax.transAxes, ha="center", va="top", fontsize="large")
        plt.show()

    def animate(self, args):
        model, history, cfg, cm_train, cm_test = core.load_model(args.filename)

        print("Configuration for model in %s is:" % args.filename)
        print(yaml.dump(cfg, default_flow_style=False))

        sr = cfg['data']['sr']
        gender = cfg['data']['gender']
        vowels = cfg['data']['vowels']
        N_classes = len(vowels)

        x_train, x_test, y_train, y_test = data.load_selected_vowels(
                            vowels,
                            gender=gender, 
                            sr=sr, 
                            normalize=True, 
                            train_size=N_classes, 
                            test_size=N_classes
                        )

        test_ds = TensorDataset(x_test, y_test)  
        for xb, yb in DataLoader(test_ds, batch_size=1):
            with torch.no_grad():
                field_dist = model(xb, probe_output=False)
                viz.animate_fields(model, field_dist, yb)

if __name__ == '__main__':
    WaveTorch()
