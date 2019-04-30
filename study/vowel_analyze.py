"""Perform various analysis tasks on a saved model.
"""

import torch
import wavetorch
from torch.utils.data import TensorDataset, DataLoader

import argparse
import yaml
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

import pandas as pd

import librosa
import librosa.display

COL_TRAIN = "#1f77b4"
COL_TEST  = "#2ca02c"

parser = argparse.ArgumentParser() 
parser.add_argument('command', type=str)
parser.add_argument('filename', type=str)
parser.add_argument('--times', nargs='+', type=int, default=None)
parser.add_argument('--saveprefix', type=str, default=None)
parser.add_argument('--vowel_samples', nargs='+', type=int, default=None)
parser.add_argument('--num_threads', type=int, default=4)
parser.add_argument('--labels', action='store_true')
parser.add_argument('--use-cuda', action='store_true')

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

    def fields(self, args):
        model, history, history_state, cfg = wavetorch.core.load_model(args.filename)

        print("Configuration for model in %s is:" % args.filename)
        print(yaml.dump(cfg, default_flow_style=False))

        sr = cfg['data']['sr']
        gender = cfg['data']['gender']
        vowels = cfg['data']['vowels']
        N_classes = len(vowels)

        X, Y, F = wavetorch.data.load_all_vowels(vowels, gender='both', sr=sr, normalize=True, random_state=0)

        # fig, axs = plt.subplots(N_classes, 1, constrained_layout=True, figsize=(4, 3), sharex=True, sharey=True)
        fig, axs = plt.subplots(N_classes, len(args.times), constrained_layout=True, figsize=(6.5, 6.5), sharex=True, sharey=True)
        fig2, axs2 = plt.subplots(3, 2, constrained_layout=True, figsize=(3.7, 2))
        for i in range(N_classes):
            xb, yb = wavetorch.data.select_vowel_sample(X, Y, F, i, ind=args.vowel_samples[i] if args.vowel_samples is not None else None)
            with torch.no_grad():
                fields = model(xb, probe_output=False)
                wavetorch.viz.plot_probe_integrals(model, fields, yb, xb, ax=axs2)
                wavetorch.viz.plot_field_snapshot(model, fields, args.times, yb, fig_width=6, block=False, axs=axs[i,:])
                axs[i,0].text(-0.05, 0.5, vowels[i] + ' vowel', transform=axs[i,0].transAxes, ha="right", va="center")
                # axs[i].set_ylabel(r"Probe $\int \vert u_n \vert^2 dt$")

        # axs[-1].set_xlabel("Time")
        if args.labels:
            wavetorch.viz.apply_sublabels(axs.ravel(), xy=[(5,-5)], size='medium', weight='bold', ha='left', va='top')
        plt.show()

    def stft(self, args):
        model, history, history_state, cfg = wavetorch.core.load_model(args.filename)

        print("Configuration for model in %s is:" % args.filename)
        print(yaml.dump(cfg, default_flow_style=False))

        sr = cfg['data']['sr']
        gender = cfg['data']['gender']
        vowels = cfg['data']['vowels']
        N_classes = len(vowels)

        X, Y, F = wavetorch.data.load_all_vowels(vowels, gender='both', sr=sr, normalize=True, random_state=0)

        fig, axs = plt.subplots(N_classes, N_classes+1, constrained_layout=True, figsize=(4.5*(N_classes+1)/N_classes,4.5), sharex=True, sharey=True)

        for i in range(N_classes):
            xb, yb = wavetorch.data.select_vowel_sample(X, Y, F, i, ind=args.vowel_samples[i] if args.vowel_samples is not None else None)
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
        model, history, history_state, cfg = wavetorch.core.load_model(args.filename)

        print("Configuration for model in %s is:" % args.filename)
        print(yaml.dump(cfg, default_flow_style=False))

        X, Y, F = wavetorch.data.load_all_vowels(cfg['data']['vowels'], gender=cfg['data']['gender'], sr=cfg['data']['sr'], normalize=True, random_state=0)

        model.load_state_dict(history_state[cfg['training']['N_epochs']])

        for i in range(len(cfg['data']['vowels'])):
            xb, yb = wavetorch.data.select_vowel_sample(X, Y, F, i, ind=args.vowel_samples[i] if args.vowel_samples is not None else None)
            with torch.no_grad():
                this_savename = None if args.saveprefix is None else args.saveprefix + str(i) + '.mp4'
                field_dist = model(xb, probe_output=False)
                wavetorch.viz.animate_fields(model, field_dist, yb, filename=this_savename, interval=1)

if __name__ == '__main__':
    WaveTorch()
