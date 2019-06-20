"""Generate a summary of a previously trained vowel recognition model.
"""

import torch
import wavetorch
from torch.utils.data import TensorDataset, DataLoader

import argparse
import yaml
import time
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

import pandas as pd

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [
r"\usepackage{tgheros}",
r"\usepackage{bm}", 
r"\usepackage{sansmath}",
r"\sansmath",
r"\usepackage{siunitx}",
r"\sisetup{detect-all}",
r"\usepackage{amsmath}",
r"\usepackage{amsfonts}",
r"\usepackage{amssymb}",
r"\usepackage{braket}",
r"\renewcommand{\rmdefault}{\sfdefault}"
]

COL_TRAIN = "#1f77b4"
COL_TEST  = "#2ca02c"

parser = argparse.ArgumentParser() 
parser.add_argument('filename', type=str)
parser.add_argument('--vmin', type=float, default=1e-3)
parser.add_argument('--vmax', type=float, default=1.0)
parser.add_argument('--fig', type=str, default=None)
parser.add_argument('--title_off', action='store_true')
parser.add_argument('--labels', action='store_true')
parser.add_argument('--vowel_samples', nargs='+', type=int, default=None)

if __name__ == '__main__':
    args = parser.parse_args()

    model, history, history_state, cfg = wavetorch.core.load_model(args.filename)

    try:
        if cfg['seed'] is not None:
            torch.manual_seed(cfg['seed'])
    except:
        pass
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
    ax_cm_test0  = fig.add_subplot(gs_left[0,1])

    ax_cm_train1 = fig.add_subplot(gs_left[1,0])
    ax_cm_test1  = fig.add_subplot(gs_left[1,1])

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
                color=COL_TRAIN, bbox=wavetorch.viz.bbox_white)
    ax_acc.annotate('%.1f\%%' % (history_mean['acc_test'].tail(1).item()*100),
                xy=(epochs[-1], history_mean['acc_test'].tail(1).item()*100), xycoords='data',
                xytext=(-1, -5), textcoords='offset points', ha='left', va='center', fontsize='small',
                color=COL_TEST, bbox=wavetorch.viz.bbox_white)
    print('Accuracy (train): %.1f%% +/- %.1f%%' % (history_mean['acc_train'].tail(1).item()*100, history_std['acc_train'].tail(1).item()*100))
    print('Accuracy  (test): %.1f%% +/- %.1f%%' % (history_mean['acc_test'].tail(1).item()*100, history_std['acc_test'].tail(1).item()*100))

    cm_train = history.groupby('epoch')['cm_train'].apply(np.mean).head(1).item()
    cm_test = history.groupby('epoch')['cm_test'].apply(np.mean).head(1).item()
    wavetorch.viz.plot_confusion_matrix(cm_train, title="Training dataset", normalize=True, ax=ax_cm_train0, labels=vowels)
    wavetorch.viz.plot_confusion_matrix(cm_test, title="Testing dataset", normalize=True, ax=ax_cm_test0, labels=vowels)

    cm_train = history.groupby('epoch')['cm_train'].apply(np.mean).tail(1).item()
    cm_test = history.groupby('epoch')['cm_test'].apply(np.mean).tail(1).item()
    wavetorch.viz.plot_confusion_matrix(cm_train, title="Training dataset", normalize=True, ax=ax_cm_train1, labels=vowels)
    wavetorch.viz.plot_confusion_matrix(cm_test, title="Testing dataset", normalize=True, ax=ax_cm_test1, labels=vowels)

    X, Y, F = wavetorch.data.load_all_vowels(vowels, gender='both', sr=sr, random_state=0)

    model.load_state_dict(history_state[cfg['training']['N_epochs']])

    for i in range(N_classes):
        xb, yb = wavetorch.data.select_vowel_sample(X, Y, F, i, ind=args.vowel_samples[i] if args.vowel_samples is not None else None)
        with torch.no_grad():
            field_dist = model(xb, probe_output=False)
            probe_series = field_dist[0, :, model.px, model.py]
            print(yb.argmax().item())
            wavetorch.viz.plot_total_field(model, field_dist, yb, ax=ax_fields[yb.argmax().item()], cbar=True, cax=ax_fields[-1], vmin=args.vmin, vmax=args.vmax)

    if args.labels:
        wavetorch.viz.apply_sublabels([ax_cm_train0, ax_cm_test0, ax_cm_train1, ax_cm_test1, ax_loss, ax_acc] + ax_fields[0:-1],
                            xy=[(-35,0), (-35,0), (-35,0), (-35,0), (-25,0), (-40,0), (8,-6), (8,-6), (8,-6)],
                            colors=['k', 'k', 'k', 'k', 'k', 'k', 'w', 'w', 'w'])

    plt.show()
    if args.fig is not None:
        fig.savefig(args.fig, dpi=300)
    else:
        fig.savefig(os.path.splitext(args.filename)[0]+"_summary.png", dpi=300)
