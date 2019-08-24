"""Propagate some waves
"""

import torch
import wavetorch
import numpy as np
import skimage
import matplotlib.pyplot as plt
import librosa
import argparse
parser = argparse.ArgumentParser() 
parser.add_argument('--fields', '-fields', '-f', action='store_true')
parser.add_argument('--spectrum', '-spectrum', '-s', action='store_true')
parser.add_argument('--vowel', action='store_true')
args = parser.parse_args()

# Define the geometry and model

Nx = 201
Ny = 201
dt = 0.707
h  = 1.0

sr = 10000

domain = torch.zeros(Nx, Ny)
rr, cc = skimage.draw.circle(101, 101, 30)
domain[rr, cc] = 1

# probe_list = [
#     wavetorch.Probe(55, 101),
#     wavetorch.Probe(150, 101) 
# ]
probe_list = []
src_list = [
    wavetorch.Source(50, 101)
]

model = wavetorch.WaveCell(
    Nx,
    Ny, 
    h, 
    dt, 
    # c_nl=-7e2,
    satdamp_b0=0.2, 
    satdamp_uth=9e-5, 
    c0=1.0, 
    c1=0.8, 
    sigma=3.0, 
    N=20, 
    p=4.0, 
    probes=probe_list, 
    sources=src_list, 
    init=domain)


# Define the source
if args.vowel:
    x, _, _ = wavetorch.data.load_all_vowels(
        ['ae', 'ei', 'iy'],
        gender='men', 
        sr=sr, 
        normalize=True, 
        max_samples=3)
    X = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    X = X[0,1000:3500].unsqueeze(0)
else:
    t = np.arange(0, 500*dt, dt)
    omega1 = 2*np.pi*1/dt/15
    X = np.sin(omega1*t) * t / (1 + t)
    X = torch.tensor(X, dtype=torch.get_default_dtype()).unsqueeze(0)

with torch.no_grad():
    u = model.forward(X)

if args.fields:
    Nshots = 10
    Ntime  = X.shape[1]
    times = [i for i in range(int(Ntime/Nshots), Ntime, int(Ntime/Nshots))]
    wavetorch.plot.field_snapshot(
        model, 
        u, 
        times, 
        ylabel=None, 
        label=True, 
        cbar=True, 
        Ny=3,
        fig_width=10)

if args.spectrum:
    out = model.measure_probes(u).squeeze(-1)
    out = out.squeeze().numpy()

    n_fft=1024
    out_ft = [np.abs(librosa.core.stft(out[:,i],n_fft=n_fft)) for i in range(0, out.shape[1])]
    out_ft_int = np.vstack([x.sum(axis=1) for x in out_ft])

    fig, ax = plt.subplots(2,1,constrained_layout=True,figsize=(4,6))

    ax[0].plot(out)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Probe amplitude')

    for i in range(0, out_ft_int.shape[0]):
        ax[1].fill_between(librosa.core.fft_frequencies(sr=sr, n_fft=n_fft),
                         out_ft_int[i,:],
                         alpha=0.45)
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Energy')
    fig.align_labels()
    plt.show();

# fig, axs = plt.subplots(len(out_ft),1, constrained_layout=True, figsize=(4,5))
# for (i, ax) in enumerate(axs):
#     librosa.display.specshow(
#         librosa.amplitude_to_db(out_ft[i]),
#         sr=sr,
#         vmax=0,
#         ax=ax,
#         vmin=-60,
#         y_axis='linear',
#         x_axis='time',
#         cmap=plt.cm.inferno
#     )
# plt.show()
