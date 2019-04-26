"""Perform vowel recognition training.
"""

import torch
import wavetorch
from torch.utils.data import TensorDataset, DataLoader

import yaml
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold
import skimage

Nx = 201
Ny = 201

vowels = ['ae', 'ei', 'iy']
N_classes = len(vowels)
gender = 'men'
window_size = 1000
batch_size = 3
max_samples = 3
lr = 0.01
sr = 10000

pd = 40

seed = 2019

N_epochs = 5

torch.set_num_threads(4)
torch.manual_seed(seed)

X, Y, F = wavetorch.data.load_all_vowels(vowels, gender=gender, sr=sr, max_samples=3, random_state=seed)
print(F)

if window_size:
    x_train = torch.nn.utils.rnn.pad_sequence([wavetorch.core.window_data(X[i], window_size) for i in range(len(X))], batch_first=True)
else:
    x_train = torch.nn.utils.rnn.pad_sequence([X[i] for i in range(len(X))], batch_first=True)

x_full = torch.nn.utils.rnn.pad_sequence([X[i] for i in range(len(X))], batch_first=True)

y_train = torch.nn.utils.rnn.pad_sequence([Y[i] for i in range(len(Y))], batch_first=True)

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

src_x, src_y = skimage.draw.line(50, 95, 50, 105)
src_x = src_x.tolist()
src_y = src_y.tolist()

# src_x = [50]
# src_y = [101]
px = [150, 150, 150]
py = [75, 100, 125]

# design_region = torch.zeros((Nx, Ny), dtype=torch.uint8)
# rr, cc = skimage.draw.circle(101, 101, 30)
# design_region[rr, cc] = 1

# design_region = torch.zeros((Nx, Ny), dtype=torch.uint8)
# rr, cc = skimage.draw.rectangle((60, 20), (140, 180))
# design_region[rr, cc] = 1

design_region = torch.ones((Nx, Ny), dtype=torch.uint8)
rad = 6
for (x, y) in zip(px+src_x,py+src_y):
    rr, cc = skimage.draw.circle(x, y, rad)
    design_region[rr, cc] = 0

model = wavetorch.core.WaveCell(1.0, Nx, Ny, src_x, src_y, px, py, design_region=design_region, c0=1.0, c1=0.5)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
        
history, _ = wavetorch.core.train(model, optimizer, criterion, train_dl, None, N_epochs, batch_size, accuracy=wavetorch.core.accuracy_onehot)

with torch.no_grad():
    field = model.forward(x_train, probe_output=False)

wavetorch.viz.plot_structure(model)

wavetorch.viz.animate_fields(model, field, y_train)
