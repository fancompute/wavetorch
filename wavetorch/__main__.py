import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

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
args_global.add_argument('--dtype', type=str, default='float32',
                            help='Data type to use for tensors. Either float32 or float64')

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
args_summary.add_argument('--vowel_samples', nargs='+', type=int, default=None)
args_summary.add_argument('filename', type=str)

args_fields = subargs.add_parser('fields', parents=[args_global])
args_fields.add_argument('filename', type=str)
args_fields.add_argument('times', nargs='+', type=int)
args_fields.add_argument('--vowel_samples', nargs='+', type=int, default=None)

args_stft = subargs.add_parser('stft', parents=[args_global])
args_stft.add_argument('filename', type=str)
args_stft.add_argument('--vowel_samples', nargs='+', type=int, default=None)

args_animate = subargs.add_parser('animate', parents=[args_global])
args_animate.add_argument('filename', type=str)
args_animate.add_argument('saveprefix', type=str, default=None)
args_animate.add_argument('--vowel_samples', nargs='+', type=int, default=None)

class WaveTorch(object):

    def __init__(self):
        args = parser.parse_args()

        if args.use_cuda and torch.cuda.is_available():
            args.dev = torch.device('cuda')
        else:
            args.dev = torch.device('cpu')

        if args.dtype == 'float32':
            torch.set_default_dtype(torch.float32)
        elif args.dtype == 'float64':
            torch.set_default_dtype(torch.float64)
        else:
            raise ValueError('Unsupported data type: %s; should be either float32 or float64' % args.dtype)

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

        X, Y, _ = data.load_all_vowels(cfg['data']['vowels'], gender=cfg['data']['gender'], sr=cfg['data']['sr'], normalize=True, max_samples=cfg['training']['max_samples'])

        skf = StratifiedKFold(n_splits=cfg['training']['train_test_divide'], random_state=None, shuffle=True)
        samps = [y.argmax().item() for y in Y]

        history = None
        history_model_state = []
        for num, (train_index, test_index) in enumerate(skf.split(np.zeros(len(samps)), samps)):
            if cfg['training']['use_cross_validation']: print("Cross Validation Fold %2d/%2d" % (num+1, cfg['training']['train_test_divide']))

            if cfg['data']['window_size']:
                crop = cfg['data']['window_size']
                x_train = torch.nn.utils.rnn.pad_sequence([X[i][int(len(X[i])/2-crop/2):int(len(X[i])/2+crop/2)] for i in train_index], batch_first=True)
            else:
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
                        init_rand=cfg['geom']['use_rand_init'], design_region=design_region, h=cfg['geom']['h'],
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

        gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.4])
        gs_left  = gs[0].subgridspec(3, 2)
        gs_right = gs[1].subgridspec(N_classes+1, 1, height_ratios=[1 for i in range(0,N_classes)] + [0.05])
        gs_bot  = gs_left[2,:].subgridspec(1, 2)

        ax_cm_train0 = fig.add_subplot(gs_left[0,0])
        ax_cm_test0  = fig.add_subplot(gs_left[0,1], sharex=ax_cm_train0, sharey=ax_cm_train0)

        ax_cm_train1 = fig.add_subplot(gs_left[1,0], sharex=ax_cm_train0, sharey=ax_cm_train0)
        ax_cm_test1  = fig.add_subplot(gs_left[1,1], sharex=ax_cm_train0, sharey=ax_cm_train0)

        ax_loss = fig.add_subplot(gs_bot[0])
        ax_acc = fig.add_subplot(gs_bot[1])

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
        ax_loss.set_xlabel('Training epoch \#')

        ax_acc.plot(epochs, history_mean['acc_train'].values*100, "-", label="Training dataset", ms=4, color=COL_TRAIN)
        ax_acc.fill_between(epochs,
                            history_mean['acc_train'].values*100-history_std['acc_train'].values*100,
                            history_mean['acc_train'].values*100+history_std['acc_train'].values*100, color=COL_TRAIN, alpha=0.15)
        ax_acc.plot(epochs, history_mean['acc_test'].values*100, "-", label="Testing dataset", ms=4, color=COL_TEST)
        ax_acc.fill_between(epochs,
                            history_mean['acc_test'].values*100-history_std['acc_test'].values*100,
                            history_mean['acc_test'].values*100+history_std['acc_test'].values*100, color=COL_TEST, alpha=0.15)
        ax_acc.set_xlabel('Training epoch \#')
        ax_acc.set_ylabel('Accuracy')
        
        ax_acc.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=10))
        ax_acc.set_ylim([20,100])
        ax_loss.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=0.1))
        ax_loss.set_ylim([0.7,1.2])

        ax_acc.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f\%%'))

        ax_loss.legend(fontsize='small')

        # ax_acc.annotate("%.1f%% training set accuracy" % (history_mean['acc_train'].tail(1).item()*100), xy=(0.1,0.1), xytext=(0,10), textcoords="offset points",  xycoords="axes fraction", ha="left", va="bottom", color=COL_TRAIN)
        # ax_acc.annotate("%.1f%% testing set accuracy" % (history_mean['acc_test'].tail(1).item()*100), xy=(0.1,0.1), xycoords="axes fraction", ha="left", va="bottom", color=COL_TEST)
        ax_acc.annotate('%.1f\%%' % (history_mean['acc_train'].tail(1).item()*100),
                    xy=(epochs[-1], history_mean['acc_train'].tail(1).item()*100), xycoords='data',
                    xytext=(-1, 5), textcoords='offset points', ha='left', va='center', fontsize='small',
                    color=COL_TRAIN, bbox=viz.bbox_white)
        ax_acc.annotate('%.1f\%%' % (history_mean['acc_test'].tail(1).item()*100),
                    xy=(epochs[-1], history_mean['acc_test'].tail(1).item()*100), xycoords='data',
                    xytext=(-1, -5), textcoords='offset points', ha='left', va='center', fontsize='small',
                    color=COL_TEST, bbox=viz.bbox_white)
        print('Accuracy (train): %.1f%% +/- %.1f%%' % (history_mean['acc_train'].tail(1).item()*100, history_std['acc_train'].tail(1).item()*100))
        print('Accuracy  (test): %.1f%% +/- %.1f%%' % (history_mean['acc_test'].tail(1).item()*100, history_std['acc_test'].tail(1).item()*100))

        cm_train = history.groupby('epoch')['cm_train'].apply(np.mean).head(1).item()
        cm_test = history.groupby('epoch')['cm_test'].apply(np.mean).head(1).item()
        viz.plot_confusion_matrix(cm_train, title="Training dataset", normalize=True, ax=ax_cm_train0, labels=vowels)
        viz.plot_confusion_matrix(cm_test, title="Testing dataset", normalize=True, ax=ax_cm_test0, labels=vowels)

        cm_train = history.groupby('epoch')['cm_train'].apply(np.mean).tail(1).item()
        cm_test = history.groupby('epoch')['cm_test'].apply(np.mean).tail(1).item()
        viz.plot_confusion_matrix(cm_train, title="Training dataset", normalize=True, ax=ax_cm_train1, labels=vowels)
        viz.plot_confusion_matrix(cm_test, title="Testing dataset", normalize=True, ax=ax_cm_test1, labels=vowels)

        X, Y, F = data.load_all_vowels(vowels, gender='both', sr=sr, random_state=0)

        # model.load_state_dict(history_state[0])

        for i in range(N_classes):
            xb, yb = data.select_vowel_sample(X, Y, F, i, ind=args.vowel_samples[i] if args.vowel_samples is not None else None)
            with torch.no_grad():
                field_dist = model(xb, probe_output=False)
                probe_series = field_dist[0, :, model.px, model.py]
                viz.plot_total_field(model, field_dist, yb, ax=ax_fields[yb.argmax().item()], cbar=True, cax=ax_fields[-1], vmin=args.vmin, vmax=args.vmax)

        viz.apply_sublabels([ax_cm_train0, ax_cm_test0, ax_cm_train1, ax_cm_test1, ax_loss, ax_acc] + ax_fields[0:-1],
                            xy=[(-35,0), (-35,0), (-35,0), (-35,0), (-25,0), (-40,0), (8,-6), (8,-6), (8,-6)],
                            colors=['k', 'k', 'k', 'k', 'k', 'k', 'w', 'w', 'w'])

        plt.show()
        if args.fig is not None:
            fig.savefig(args.fig, dpi=300)
        else:
            fig.savefig(os.path.splitext(args.filename)[0]+"_summary.png", dpi=300)

    def fields(self, args):
        model, history, history_state, cfg = core.load_model(args.filename)

        print("Configuration for model in %s is:" % args.filename)
        print(yaml.dump(cfg, default_flow_style=False))

        sr = cfg['data']['sr']
        gender = cfg['data']['gender']
        vowels = cfg['data']['vowels']
        N_classes = len(vowels)

        X, Y, F = data.load_all_vowels(vowels, gender='both', sr=sr, normalize=True, random_state=0)

        # fig, axs = plt.subplots(N_classes, 1, constrained_layout=True, figsize=(4, 3), sharex=True, sharey=True)
        fig, axs = plt.subplots(N_classes, len(args.times), constrained_layout=True, figsize=(6.5, 6.5), sharex=True, sharey=True)
        fig2, axs2 = plt.subplots(3, 2, constrained_layout=True, figsize=(3.7, 2))
        for i in range(N_classes):
            xb, yb = data.select_vowel_sample(X, Y, F, i, ind=args.vowel_samples[i] if args.vowel_samples is not None else None)
            with torch.no_grad():
                fields = model(xb, probe_output=False)
                viz.plot_probe_integrals(model, fields, yb, xb, ax=axs2)
                viz.plot_field_snapshot(model, fields, args.times, yb, fig_width=6, block=False, axs=axs[i,:])
                axs[i,0].text(-0.05, 0.5, vowels[i] + ' vowel', transform=axs[i,0].transAxes, ha="right", va="center")
                # axs[i].set_ylabel(r"Probe $\int \vert u_n \vert^2 dt$")

        # axs[-1].set_xlabel("Time")
        viz.apply_sublabels(axs.ravel(), xy=[(5,-5)], size='medium', weight='bold', ha='left', va='top')
        plt.show()

    def stft(self, args):
        model, history, history_state, cfg = core.load_model(args.filename)

        print("Configuration for model in %s is:" % args.filename)
        print(yaml.dump(cfg, default_flow_style=False))

        sr = cfg['data']['sr']
        gender = cfg['data']['gender']
        vowels = cfg['data']['vowels']
        N_classes = len(vowels)

        X, Y, F = data.load_all_vowels(vowels, gender='both', sr=sr, normalize=True, random_state=0)

        fig, axs = plt.subplots(N_classes, N_classes+1, constrained_layout=True, figsize=(4.5*(N_classes+1)/N_classes,4.5), sharex=True, sharey=True)

        for i in range(N_classes):
            xb, yb = data.select_vowel_sample(X, Y, F, i, ind=args.vowel_samples[i] if args.vowel_samples is not None else None)
            with torch.no_grad():
                j = yb.argmax().item()
                ax = axs[j, 0]
                ax.set_facecolor('black')

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
                if j == 0:
                    ax.set_title("Input signal")

                for k in range(1, probe_series.shape[1]+1):
                    ax = axs[j, k]
                    
                    output_stft = np.abs(librosa.stft(probe_series[:,k-1].numpy(), n_fft=256))

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

                    if j == 0:
                        ax.set_title("Output probe %d" % (k))
                    if k == 1:
                        ax.text(-0.3, 0.5, vowels[j] + ' vowel', transform=ax.transAxes, ha="right", va="center")
                    
                    if k > 0:
                        ax.set_ylabel('')
                    if j < N_classes-1:
                        ax.set_xlabel('')
                    # if j == k:
                        # ax.text(0.5, 0.95, '%s at probe #%d' % (vowels[j], k+1), color="w", transform=ax.transAxes, ha="center", va="top", fontsize="large")
        plt.show()

    def animate(self, args):
        model, history, history_state, cfg = core.load_model(args.filename)

        print("Configuration for model in %s is:" % args.filename)
        print(yaml.dump(cfg, default_flow_style=False))

        X, Y = data.load_all_vowels(
                        cfg['data']['vowels'],
                        gender='men', 
                        sr=cfg['data']['sr'], 
                        normalize=True, 
                        max_samples=len(cfg['data']['vowels'])
                    )

        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        Y = torch.nn.utils.rnn.pad_sequence(Y, batch_first=True)
        test_ds = TensorDataset(X, Y)

        model.load_state_dict(history_state[cfg['training']['N_epochs']])

        for i, (xb, yb) in enumerate(DataLoader(test_ds, batch_size=1)):
            with torch.no_grad():
                this_savename = None if args.saveprefix is None else args.saveprefix + str(i) + '.mp4'
                field_dist = model(xb, probe_output=False)
                viz.animate_fields(model, field_dist, yb, filename=this_savename)

if __name__ == '__main__':
    WaveTorch()
