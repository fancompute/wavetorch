"""Propagate some waves
"""

import torch
import wavetorch
import numpy as np
import skimage.draw as draw
import matplotlib.pyplot as plt

# Define the geometry and model

Nx = 120
Ny = 60
dt = 0.707
h  = 1.0

domain = torch.zeros(Nx, Ny)
domain[40:84,20:40] = 1

probe_list = [
    wavetorch.Probe(35, 30),
    wavetorch.Probe(90, 30) 
]
src_list = [
    wavetorch.Source(25, 30)
]

model = wavetorch.WaveCell(
    Nx,
    Ny, 
    h, 
    dt, 
    satdamp_b0=0.1, 
    satdamp_uth=0.0001, 
    c0=1.0, 
    c1=0.5, 
    sigma=3.0, 
    N=20, 
    p=4.0, 
    probes=probe_list, 
    sources=src_list, 
    init=domain)

# Define the source
# t = np.arange(0, 500*dt, dt)
# omega1 = 2*np.pi*1/dt/15
# X = np.sin(omega1*t) * t / (1 + t)
# X = torch.tensor(X, dtype=torch.get_default_dtype()).unsqueeze(0)

x, _, _ = wavetorch.data.load_all_vowels(
    ['ae', 'ei', 'iy'],
    gender='men', 
    sr=10000, 
    normalize=True, 
    max_samples=3)
X = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
X = X[0,700:2500].unsqueeze(0)

with torch.no_grad():
    u = model.forward(X)

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

out = model.measure_probes(u).squeeze(-1)

plt.figure();
plt.plot(out.squeeze().numpy())
plt.xlabel('Time')
plt.ylabel('Probe amplitude')

plt.show()
