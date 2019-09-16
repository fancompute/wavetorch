"""Propagate some waves through a domain
"""

import torch
import wavetorch
import numpy as np
import skimage
import matplotlib.pyplot as plt
import librosa
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--show_fields', '-fields', '-f', action='store_true')
parser.add_argument('--show_spectrum', '-spectrum', '-s', action='store_true')
parser.add_argument('--use_vowel', action='store_true')
args = parser.parse_args()

domain_shape = (201, 101)

dt = 0.707
h  = 1.0

sr = 10000

domain = torch.zeros(domain_shape)
rr, cc = skimage.draw.circle( int(domain_shape[0]/2) , int(domain_shape[1]/2), 30)
domain[rr, cc] = 1

geom  = wavetorch.WaveGeometryFreeForm(domain_shape, h, c0=1.0, c1=0.5, rho=domain)
cell  = wavetorch.WaveCell(dt, geom)
src   = wavetorch.WaveSource(25, 50)
probe = [wavetorch.WaveIntensityProbe(175, 75),
         wavetorch.WaveIntensityProbe(175, 50),
         wavetorch.WaveIntensityProbe(175, 25)]

model = wavetorch.WaveRNN(cell, src, probe)

# Define the source
if args.use_vowel:
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
    u = model.forward(X, output_fields=args.show_fields)

if args.show_fields:
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

if args.show_spectrum:
    out = u.squeeze().numpy()

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
                         alpha=0.2)
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Energy')
    fig.align_labels()
    plt.show()


# wavetorch.plot.plot_structure(model)
# wavetorch.plot.plot_structure(model, outline=True)
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

torch.save(model.state_dict(), './tmp.pt')
new_cell  = wavetorch.WaveCell(1.0, None)
new_src   = wavetorch.WaveSource(0, 0)
newprobe = [wavetorch.WaveIntensityProbe(0, 0)]

model = wavetorch.WaveRNN(new_cell, new_src, new_probe)
