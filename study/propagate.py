"""Propagate some waves
"""

import torch
import wavetorch

import numpy as np
import skimage

import matplotlib.pyplot as plt

dt = 1.0
t = np.arange(0, 300*dt, dt)

freq0 = 1/dt/15

Nx = 201
Ny = 201

c0 = 1.0 # Background wave speed
c1 = 0.9

src_x, src_y = skimage.draw.line(50, 80, 50, 120)
px = [150]
py = [105]

x = np.sin(2*np.pi*freq0*t) * t / (1 + t)
x = torch.tensor(x, dtype=torch.get_default_dtype()).unsqueeze(0)

design_region = np.zeros((Nx, Ny), dtype=np.uint8)
rr, cc = skimage.draw.circle(101, 101, 30)
design_region[rr, cc] = 1

model = wavetorch.core.WaveCell(dt, Nx, Ny, src_x, src_y, px, py, c0=c0, c1=c1, design_region=torch.tensor(design_region))

with torch.no_grad():
    fields = model.forward(x, probe_output=False)

wavetorch.viz.plot_field_snapshot(model, fields, [25, 50, 75, 100, 125, 150, 175, 199], ylabel=None, label=True, cbar=True, Ny=2)

plt.figure()
plt.plot(t, fields[0,:,px,py].squeeze().numpy())
plt.plot(t, x.squeeze().numpy())
plt.show()
