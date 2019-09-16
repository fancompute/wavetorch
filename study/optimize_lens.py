"""Optimize a toy lens model
"""
import torch
import wavetorch
import numpy as np
import skimage
import matplotlib.pyplot as plt
import librosa
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--use_vowel', action='store_true')
args = parser.parse_args()

domain_shape = (151, 151)

dt = 0.707
h  = 1.0

sr = 10000

domain = torch.zeros(domain_shape)
rr, cc = skimage.draw.circle( int(domain_shape[0]/2) , int(domain_shape[1]/2), 40)
domain[rr, cc] = 0.5

geom  = wavetorch.WaveGeometryFreeForm(domain_shape, h, c0=1.0, c1=0.5, domain=domain, design_region=None)
cell  = wavetorch.WaveCell(dt, geom)
src   = wavetorch.WaveSource(25, 75)
probe = [wavetorch.WaveIntensityProbe(120, 120),
         wavetorch.WaveIntensityProbe(125, 75),
         wavetorch.WaveIntensityProbe(120, 25)]

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

###

optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-3)
criterion = torch.nn.CrossEntropyLoss()

beta_schedule       = torch.tensor([100, 400, 800, 1000, 1500, 2000])
beta_schedule_epoch = torch.tensor([-1,  10,  20,  30,  40, 50])

loss_iter = []
for i in range(0, 60):
    # model.cell.geom.beta = beta_schedule[beta_schedule_epoch<i][-1]

    def closure():
        optimizer.zero_grad()
        u = model(X).sum(dim=1)
        loss = criterion(u, torch.tensor([2]))
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    model.cell.geom.constrain_to_design_region()
    print("Epoch: {} -- Loss: {}".format(i, loss))
    loss_iter.append(loss.item())

plt.figure()
plt.plot(loss_iter, 'o-')
plt.xlabel("Epoch")
plt.ylabel("Cross entropy loss")

with torch.no_grad():
    u = model(X, output_fields=True)

Nshots = 6
Ntime  = X.shape[1]
times = [i for i in range(int(Ntime/Nshots), Ntime, int(Ntime/Nshots))]
wavetorch.plot.field_snapshot(
    model,
    u, 
    times, 
    ylabel=None, 
    label=True, 
    cbar=True, 
    Ny=2,
    fig_width=7)

wavetorch.plot.plot_structure(model)
