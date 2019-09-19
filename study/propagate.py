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
parser.add_argument('--use_vowel', action='store_true')
args = parser.parse_args()

domain_shape = (151, 151)

dt = 0.707
h  = 1.0

sr = 10000

domain = torch.zeros(domain_shape)
rr, cc = skimage.draw.circle( int(domain_shape[0]/2) , int(domain_shape[1]/2), 30)
domain[rr, cc] = 1

geom  = wavetorch.WaveGeometryFreeForm(domain_shape, h, c0=1.0, c1=0.5, rho=domain)
cell  = wavetorch.WaveCell(dt, geom)
# src   = wavetorch.WaveSource(25, 75) # Point source
src   = wavetorch.WaveLineSource(25, 50, 25, 100) # Line source
probe = [wavetorch.WaveIntensityProbe(125, 100),
         wavetorch.WaveIntensityProbe(125, 75),
         wavetorch.WaveIntensityProbe(125, 50)]

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

# torch.save(model.state_dict(), './tmp.pt')
# new_cell  = wavetorch.WaveCell(1.0, None)
# new_src   = wavetorch.WaveSource(0, 0)
# newprobe = [wavetorch.WaveIntensityProbe(0, 0)]
# model = wavetorch.WaveRNN(new_cell, new_src, new_probe)
