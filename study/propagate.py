"""Propagate some waves
"""

import torch
import wavetorch
import numpy as np
import skimage.draw as draw
import matplotlib.pyplot as plt

Nx = 100
Ny = 100
dt = 1e-4

# Define the geometry

design_region = np.zeros((Nx, Ny), dtype=np.uint8)
design_region[30:70, 21:79] = 1

geom = wavetorch.Geometry(Nx, Ny, h=5e-2, init='half', c0=331, c1=150, design_region=design_region)
geom.add_boundary_absorber(sigma=1e4, N=20, p=4.0)
geom.add_source(wavetorch.LineSource(geom, 25, 40, 25, 60))
geom.add_probe(wavetorch.IntensityProbe(75, 25, label='1'))
geom.add_probe(wavetorch.IntensityProbe(75, 50, label='2'))
geom.add_probe(wavetorch.IntensityProbe(75, 75, label='3'))

# Define the model
model = wavetorch.WaveCell(dt, geom)

# Define the source
t = np.arange(0, 500*dt, dt)
omega1 = 2*np.pi*1/dt/15
x = np.sin(omega1*t) * t / (1 + t)
x = torch.tensor(x, dtype=torch.get_default_dtype()).unsqueeze(0)

###

optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-3)
criterion = torch.nn.CrossEntropyLoss()

beta_schedule       = np.array([100, 200, 400, 600, 800, 1000])
beta_schedule_epoch = np.array([-1,  10,  20,  30,  40, 50])

loss_iter = []
for i in range(0, 60):
    model.geometry.beta = beta_schedule[beta_schedule_epoch<i][-1]

    def closure():
        optimizer.zero_grad()
        u = model.forward(x)
        y = model.measure_probes(u, integrated=True, normalized=True)
        loss = criterion(y, torch.tensor([2]))
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    model.geometry.clip_to_design_region()
    print("Epoch: {} -- Loss: {}".format(i, loss))
    loss_iter.append(loss.item())

plt.figure()
plt.plot(loss_iter, 'o-')
plt.xlabel("Epoch")
plt.ylabel("Cross entropy loss")

with torch.no_grad():
    u = model.forward(x)

wavetorch.plot.field_snapshot(geom, u, [25, 50, 75, 100, 125, 150, 175, 199], ylabel=None, label=True, cbar=True, Ny=2)
wavetorch.plot.structure(geom)
