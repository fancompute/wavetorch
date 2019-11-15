"""Generate plot of the mean vowel sample spectra
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

import librosa

try:
    from helpers.plot import mpl_set_latex
    mpl_set_latex()
except ImportError:
    import warnings
    warnings.warn('The helpers package is unavailable', ImportWarning)

n_fft = 2048
sr = 10000
vowels = ['ae', 'ei', 'iy']
colors = ['#fcaf3e', '#ad7fa8', '#ef2929']

# vowels = ['ae', 'eh', 'ih', 'oo', 'ah', 'ei', 'iy', 'uh', 'aw', 'er', 'oa', 'uw']

gender = 'both'

fig,ax=plt.subplots(1,1,constrained_layout=True, figsize=(3.5,2.75))

for i, vowel in enumerate(vowels):
    X, _, _ = wavetorch.data.load_all_vowels([vowel], gender=gender, sr=sr)
    X_ft = [np.abs(librosa.core.stft(Xi.numpy(),n_fft=n_fft)) for Xi in X]

    X_ft_int = np.vstack([Xi.sum(axis=1) for Xi in X_ft])

    X_ft_mean = np.mean(X_ft_int,axis=0)
    X_ft_std = np.std(X_ft_int,axis=0)

    ax.fill_between(librosa.core.fft_frequencies(sr=sr, n_fft=n_fft),
                     X_ft_mean,
                     alpha=0.30, color=colors[i], edgecolor="none", zorder=i ,lw=0)
    ax.plot(librosa.core.fft_frequencies(sr=sr, n_fft=n_fft),
                     X_ft_mean,
                     color=colors[i],zorder=i, label=vowel + ' vowel class', lw=1.0)
    # ax.plot(librosa.core.fft_frequencies(sr=sr, n_fft=n_fft),
    #                  X_ft_std, '-',
    #                  label=vowel + ' vowel class', color=colors[i], lw=1, zorder=i)

# ax.set_xlim([0,5000])
# ax.set_ylim([0,13])
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Mean energy spectrum (a.u.)")
ax.legend()
plt.show(block=False)


