"""Propagate some waves
"""

import torch
import wavetorch
import numpy as np
import skimage.draw as draw
import matplotlib.pyplot as plt

# Define the geometry and model

Nx = 120
Ny = 80
dt = 0.707
h  = 1.0

domain = torch.zeros(Nx, Ny)
domain[40:84,20:99] = 1

probe_list = [
    wavetorch.IntensityProbe(90, 40) 
]
src_list = [
    wavetorch.LineSource(25, 35, 25, 45)
]

model = wavetorch.WaveCell(Nx, Ny, h, dt, satdamp_b0=0.25, satdamp_uth=0.01, c0=1.0, c1=0.5, sigma=3.0, N=20, p=4.0, probes=probe_list, sources=src_list, init=domain)

# Define the source
t = np.arange(0, 500*dt, dt)
omega1 = 2*np.pi*1/dt/15
x = np.sin(omega1*t) * t / (1 + t)
x = torch.tensor(x, dtype=torch.get_default_dtype()).unsqueeze(0)

with torch.no_grad():
    u = model.forward(x)

wavetorch.plot.field_snapshot(model, u, [50, 100, 150, 200, 250, 300, 350, 400, 450, 499], ylabel=None, label=True, cbar=True, Ny=2)
# wavetorch.plot.structure(model)
