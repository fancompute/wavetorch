import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
from wavetorch.wave import WaveCell
from wavetorch.src import src_gaussian
from wavetorch.data import load_all_vowels

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
    argparser.add_argument('--N_epochs', type=int, default=5)
    argparser.add_argument('--Nx', type=int, default=140)
    argparser.add_argument('--Ny', type=int, default=70)
    argparser.add_argument('--Nt', type=int, default=500)
    argparser.add_argument('--dt', type=float, default=0.707)
    argparser.add_argument('--probe_space', type=int, default=10)
    argparser.add_argument('--probe_x', type=int, default=120)
    argparser.add_argument('--probe_y', type=int, default=30)
    argparser.add_argument('--src_x', type=int, default=21)
    argparser.add_argument('--src_y', type=int, default=35)
    argparser.add_argument('--sr', type=int, default=500)
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
    directories_str = ("./data/vowels/a", "./data/vowels/e/", "./data/vowels/o/")

    x, y_true = load_all_vowels(directories_str, sr=args.sr, normalize=True)
    N_classes = y_true.shape[1]

    mask_src = torch.zeros(args.Nx, args.Ny, requires_grad=False)
    mask_src[args.src_x, args.src_y] = 1

    # --- Setup probe coords and loss func
    probe_x = args.probe_x
    probe_y = torch.arange(args.probe_y, args.probe_y + N_classes*args.probe_space, args.probe_space)

    def integrate_probes(y):
        return torch.sum(torch.abs(y[:,:,probe_x, probe_y]).pow(2), dim=1)

    criterion = torch.nn.CrossEntropyLoss()
    sm = torch.nn.Softmax()

    # --- Define model
    model = WaveCell(args.dt, args.Nx, args.Ny, h, args.src_x, args.src_y, pml_max=3, pml_p=4.0, pml_N=20)
    # model.to(args.dev)

    model.animate(x, batch_ind=0, block=True)

    # --- Define optimizer
    optimizer = torch.optim.LBFGS(model.parameters(), lr=args.learning_rate)

    #--- Define training function
    def train(x):
        def closure():
            optimizer.zero_grad()
            y = model(x)
            loss = criterion(integrate_probes(y), y_true.argmax(dim=1))
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        return loss

    # --- Run training
    print("Training for %d epochs..." % args.N_epochs)
    t_start = time.time()
    for epoch in range(1, args.N_epochs + 1):
        t_epoch = time.time()

        loss = train(x)

        print('Epoch: %d/%d %d%%  |  %.1f sec  |  L = %.3e' % (epoch, args.N_epochs, epoch/args.N_epochs*100, time.time()-t_epoch, loss))

    print('Total time: %.1f min' % ((time.time()-t_start)/60))
