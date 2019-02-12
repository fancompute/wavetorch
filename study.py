import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
from wavetorch.wave import WaveCell
from wavetorch.src import src_gaussian

import argparse
import os
import time

def plot_c(model):       
    plt.figure()
    plt.imshow(np.sqrt(model.c2.detach().numpy()).transpose())
    plt.colorbar()
    plt.show(block=False)

if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--N_epochs', type=int, default=3)
    argparser.add_argument('--Nx', type=int, default=140)
    argparser.add_argument('--Ny', type=int, default=140)
    argparser.add_argument('--Nt', type=int, default=500)
    argparser.add_argument('--dt', type=int, default=0.707)
    argparser.add_argument('--learning_rate', type=float, default=0.1)
    args = argparser.parse_args()

    h  = args.dt * 2.01 / 1.0

    # --- Calculate source
    t = torch.arange(0.0, args.Nt*args.dt, args.dt)
    x = src_gaussian(t, 130*args.dt, 40*args.dt) * torch.sin(2*3.14 * t / 20 / args.dt)

    pt_src = (21, 70)
    mask_src = torch.zeros(args.Nx, args.Ny, requires_grad=False)
    mask_src[pt_src[0], pt_src[1]] = 1

    # --- Define probe and loss function
    pt_probe = (119, 70)
    mask_probe = torch.zeros(args.Nx, args.Ny, requires_grad=False)
    mask_probe[pt_probe[0], pt_probe[1]] = 1

    def integrate_probe(un):
        return torch.sum(torch.abs(un * mask_probe).pow(2))

    # --- Define model
    wave_model = WaveCell(args.Nt, args.dt, args.Nx, args.Ny, h, mask_src, mask_probe, pml_max=3, pml_p=4.0, pml_N=20)

    # --- Define optimizer
    wave_optimizer = torch.optim.LBFGS(wave_model.parameters(), lr=args.learning_rate)

    # --- Define training function
    def train(x):
        def closure():
            wave_model.zero_grad()
            _, loss = wave_model.forward(x, loss_func=integrate_probe)
            loss.backward()
            return loss

        loss = wave_optimizer.step(closure)

        return loss

    # --- Run training

    loss_avg = 0.0
    print("Training for %d epochs..." % args.N_epochs)
    t_start = time.time()
    for epoch in range(1, args.N_epochs + 1):
        t_epoch = time.time()

        loss = train(x)
        loss_avg += loss

        print('Epoch: %d/%d %d%%  |  %.1f sec  |  L = %.3e' % (epoch, args.N_epochs, epoch/args.N_epochs*100, time.time()-t_epoch, loss))
    
    print('Total time: %.1f min' % ((time.time()-t_epoch)/60))

    plot_c(wave_model)


