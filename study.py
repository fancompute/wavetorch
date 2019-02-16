import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
from wavetorch.wave import WaveCell
from wavetorch.src import src_gaussian
from wavetorch.data import load_vowel

import argparse
import os
import time

# Plot the final c distribution, which is the local propagation speed
def plot_c(model):       
    plt.figure()
    plt.imshow(np.sqrt(model.c2.detach().numpy()).transpose())
    plt.colorbar()
    plt.show(block=False)

if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--N_epochs', type=int, default=1)
    argparser.add_argument('--Nx', type=int, default=140)
    argparser.add_argument('--Ny', type=int, default=140)
    argparser.add_argument('--Nt', type=int, default=500)
    argparser.add_argument('--dt', type=int, default=0.707)
    argparser.add_argument('--learning_rate', type=float, default=0.1)
    argparser.add_argument('--use-cuda', action='store_true')
    args = argparser.parse_args()

    if args.use_cuda and torch.cuda.is_available():
        print("Using GPU...")
        args.dev = torch.device('cuda')
    else:
        print("Using CPU...")
        args.dev = torch.device('cpu')

    h  = args.dt * 2.01 / 1.0

    # --- Calculate source
    x, _ = load_vowel('a', sr=5000)
    Nt = x.shape[0]
    x = x.to(args.dev)

    # Nt = 500
    # t = torch.arange(0.0, args.Nt*args.dt, args.dt)
    # x = src_gaussian(t, 130*args.dt, 40*args.dt) * torch.sin(2*3.14 * t / 20 / args.dt)

    

    pt_src = (21, 70)
    mask_src = torch.zeros(args.Nx, args.Ny, requires_grad=False)
    mask_src[pt_src[0], pt_src[1]] = 1

    # --- Define probe and loss function
    pt_probe = (119, 70)
    mask_probe = torch.zeros(args.Nx, args.Ny, requires_grad=False)
    mask_probe[pt_probe[0], pt_probe[1]] = 1

    # --- Define model
    wave_model = WaveCell(args.Nt, args.dt, args.Nx, args.Ny, h, mask_src, mask_probe, pml_max=3, pml_p=4.0, pml_N=20)
    wave_model.to(args.dev)

    def integrate_probe(un):
        return torch.sum(torch.abs(un * wave_model.mask_probe).pow(2))

    # --- Define optimizer
    wave_optimizer = torch.optim.LBFGS(wave_model.parameters(), lr=args.learning_rate)

    # --- Define training function
    def train(x):
        def closure():
            wave_model.zero_grad()
            h1 = torch.zeros(wave_model.Nx, wave_model.Ny, device=args.dev).unsqueeze(0).unsqueeze(0)
            h2 = torch.zeros(wave_model.Nx, wave_model.Ny, device=args.dev).unsqueeze(0).unsqueeze(0)
            _, loss = wave_model(x, (h1, h2), loss_func=integrate_probe)
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

    print('Total time: %.1f min' % ((time.time()-t_start)/60))

    # ani = wave_model.animate(x, block=False)
    # from matplotlib import animation
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=30, bitrate=256)
    # ani.save('test.mp4', writer=writer)

    plot_c(wave_model)


