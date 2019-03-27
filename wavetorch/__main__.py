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

import pandas as pd

import librosa
import librosa.display

from . import core
from . import data
from . import viz

import os


COL_TRAIN = "#1f77b4"
COL_TEST  = "#2ca02c"

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

        if args.name is None:
            args.name = time.strftime('%Y%m%d%H%M%S')
        if cfg['training']['prefix'] is not None:
            args.name = cfg['training']['prefix'] + '_' + args.name
        if cfg['training']['use_cross_validation']:
            args.name += "_cv"

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

        history = None
        history_model_state = []
        for num, (train_index, test_index) in enumerate(skf.split(np.zeros(len(samps)), samps)):
            if cfg['training']['use_cross_validation']: print("Cross Validation Fold %2d/%2d" % (num+1, cfg['training']['train_test_divide']))

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
            
            model.train()

            history, history_model_state = core.train(
                                                model,
                                                optimizer,
                                                criterion, 
                                                train_dl, 
                                                test_dl, 
                                                cfg['training']['N_epochs'], 
                                                cfg['training']['batch_size'], 
                                                history=history,
                                                history_model_state=history_model_state,
                                                fold=num if cfg['training']['use_cross_validation'] else -1,
                                                name=args.name,
                                                savedir=args.savedir,
                                                cfg=cfg)
            
            core.save_model(model, args.name, args.savedir, history, history_model_state, cfg)

            if not cfg['training']['use_cross_validation']:
                break

    def summary(self, args):
        model, history, history_state, cfg = core.load_model(args.filename)

        print("Configuration for model in %s is:" % args.filename)
        print(yaml.dump(cfg, default_flow_style=False))

        sr = cfg['data']['sr']
        gender = cfg['data']['gender']
        vowels = cfg['data']['vowels']
        N_classes = len(vowels)

        fig = plt.figure( figsize=(7, 4.75), constrained_layout=True)

        gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.35])
        gs_left  = gs[0].subgridspec(3, 3, width_ratios=[0.4, 0.4, 0.75], height_ratios=[0.5, 0.7, 0.7])
        gs_right = gs[1].subgridspec(N_classes+1, 1, height_ratios=[0.05] + [1 for i in range(0,N_classes)])
        gs_top  = gs_left[0,:].subgridspec(1, 2)

        ax_c0 = fig.add_subplot(gs_top[0])
        ax_c1 = fig.add_subplot(gs_top[1])

        ax_loss = fig.add_subplot(gs_left[1,2])
        ax_acc = fig.add_subplot(gs_left[2,2])

        ax_cm_train0 = fig.add_subplot(gs_left[1,0])
        ax_cm_test0  = fig.add_subplot(gs_left[2,0],sharex=ax_cm_train0)
        ax_cm_train1 = fig.add_subplot(gs_left[1,1])
        ax_cm_test1  = fig.add_subplot(gs_left[2,1],sharex=ax_cm_train1)

        ax_fields = [fig.add_subplot(gs_right[i]) for i in range(0, N_classes+1)] 

        history_mean = history.groupby('epoch').mean()
        history_std  = history.groupby('epoch').std()

        epochs = history_mean.index.values

        ax_loss.fill_between(epochs,
                             history_mean['loss_train'].values-history_std['loss_train'].values,
                             history_mean['loss_train'].values+history_std['loss_train'].values, color=COL_TRAIN, alpha=0.15)
        ax_loss.plot(epochs, history_mean['loss_train'].values, "-", label="Training dataset", ms=4, color=COL_TRAIN)
        ax_loss.fill_between(epochs,
                             history_mean['loss_test'].values-history_std['loss_test'].values,
                             history_mean['loss_test'].values+history_std['loss_test'].values, color=COL_TEST, alpha=0.15)
        ax_loss.plot(epochs, history_mean['loss_test'].values, "-", label="Testing dataset", ms=4, color=COL_TEST)
        ax_loss.set_ylabel('Loss')
        ax_loss.set_xlabel('Training epoch #')

        ax_acc.plot(epochs, history_mean['acc_train'].values*100, "-", label="Training dataset", ms=4, color=COL_TRAIN)
        ax_acc.fill_between(epochs,
                            history_mean['acc_train'].values*100-history_std['acc_train'].values*100,
                            history_mean['acc_train'].values*100+history_std['acc_train'].values*100, color=COL_TRAIN, alpha=0.15)
        ax_acc.plot(epochs, history_mean['acc_test'].values*100, "-", label="Testing dataset", ms=4, color=COL_TEST)
        ax_acc.fill_between(epochs,
                            history_mean['acc_test'].values*100-history_std['acc_test'].values*100,
                            history_mean['acc_test'].values*100+history_std['acc_test'].values*100, color=COL_TEST, alpha=0.15)
        ax_acc.set_xlabel('Training epoch #')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.set_ylim(top=100)

        ax_acc.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f%%'))


        ax_loss.legend()

        # ax_acc.annotate("%.1f%% training set accuracy" % (history_mean['acc_train'].tail(1).item()*100), xy=(0.1,0.1), xytext=(0,10), textcoords="offset points",  xycoords="axes fraction", ha="left", va="bottom", color=COL_TRAIN)
        # ax_acc.annotate("%.1f%% testing set accuracy" % (history_mean['acc_test'].tail(1).item()*100), xy=(0.1,0.1), xycoords="axes fraction", ha="left", va="bottom", color=COL_TEST)
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.75)
        ax_acc.annotate('%.1f%%' % (history_mean['acc_train'].tail(1).item()*100),
                    xy=(epochs[-1], history_mean['acc_train'].tail(1).item()*100), xycoords='data',
                    xytext=(-3, 5), textcoords='offset points', ha='left', va='center', fontsize='smaller',
                    color=COL_TRAIN, bbox=bbox_props)
        ax_acc.annotate('%.1f%%' % (history_mean['acc_test'].tail(1).item()*100),
                    xy=(epochs[-1], history_mean['acc_test'].tail(1).item()*100), xycoords='data',
                    xytext=(-3, -5), textcoords='offset points', ha='left', va='center', fontsize='smaller',
                    color=COL_TEST, bbox=bbox_props)
        viz.plot_structure(model, state=history_state[0], ax=ax_c0, quantity='c', vowels=vowels, cbar=True)
        viz.plot_structure(model, state=history_state[-1], ax=ax_c1, quantity='c', vowels=vowels, cbar=True)

        # if not args.title_off:
            # ax_c0.annotate("$c_{nl}$ = %.2f \n $b_0$ = %.2f \n $u_{th}$ = %.2f \n lr = %.0e" % (cfg['geom']['nonlinearity']['cnl'], cfg['geom']['nonlinearity']['b0'], cfg['geom']['nonlinearity']['uth'], cfg['training']['lr']),
                            # xy=(0,0), xytext=(-75,0), xycoords="axes points", textcoords="offset points", ha="left", va="bottom")

        cm_train = history.groupby('epoch')['cm_train'].apply(np.mean).head(1).item()
        cm_test = history.groupby('epoch')['cm_test'].apply(np.mean).head(1).item()
        viz.plot_confusion_matrix(cm_train, title="Training dataset", normalize=True, ax=ax_cm_train0, labels=vowels)
        viz.plot_confusion_matrix(cm_test, title="Testing dataset", normalize=True, ax=ax_cm_test0, labels=vowels)

        cm_train = history.groupby('epoch')['cm_train'].apply(np.mean).tail(1).item()
        cm_test = history.groupby('epoch')['cm_test'].apply(np.mean).tail(1).item()
        viz.plot_confusion_matrix(cm_train, title="Training dataset", normalize=True, ax=ax_cm_train1, labels=vowels)
        viz.plot_confusion_matrix(cm_test, title="Testing dataset", normalize=True, ax=ax_cm_test1, labels=vowels)

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

        model.load_state_dict(history_state[cfg['training']['N_epochs']])

        for xb, yb in DataLoader(test_ds, batch_size=1):
            with torch.no_grad():
                field_dist = model(xb, probe_output=False)
                probe_series = field_dist[0, :, model.px, model.py]
                viz.plot_total_field(model, field_dist, yb, ax=ax_fields[1+yb.argmax().item()], cbar=True, cax=ax_fields[0], vmin=args.vmin, vmax=args.vmax)

        viz.apply_sublabels([ax_c0, ax_cm_train0, ax_cm_test0, ax_c1, ax_cm_train1, ax_cm_test1, ax_loss, ax_acc] + ax_fields[1::], x=[-15, -15, -15, -15, -15, -15, -35, -35, -10, -10, -10])

        plt.show()
        if args.fig is not None:
            fig.savefig(args.fig, dpi=300)
        else:
            fig.savefig(os.path.splitext(args.filename)[0]+"_summary.png", dpi=300)

    def fields(self, args):
        model, history, _, cfg = core.load_model(args.filename)

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
        model, history, cfg, cm_train, cm_test, cm_train0, cm_test0 = core.load_model(args.filename)

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
        model, history, cfg, cm_train, cm_test, cm_train0, cm_test0 = core.load_model(args.filename)

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
