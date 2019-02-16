import torch
from torch.nn.functional import conv2d

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

class WaveCell(torch.nn.Module):

    def __init__(self, Nt, dt, Nx, Ny, h, mask_src, mask_probe, c2=None, pml_N=20, pml_p=3.0, pml_max=0.5):
        super(WaveCell, self).__init__()

        self.dt = dt

        self.Nx = Nx
        self.Ny = Ny
        self.h  = h

        if c2 is None:
            c2 = torch.ones(Nx, Ny, requires_grad=True)
        self.c2  = torch.nn.Parameter(c2)

        self.register_buffer("mask_src", mask_src)
        self.register_buffer("mask_probe", mask_probe)

        # Setup the PML/adiabatic absorber
        b_vals = pml_max * torch.linspace(0.0, 1.0, pml_N+1) ** pml_p

        b_x = torch.zeros(Nx, Ny, requires_grad=False).unsqueeze(0).unsqueeze(0)
        b_x[0,0,0:pml_N+1,:]   = torch.flip(b_vals, [0]).repeat(Ny,1).transpose(0, 1)
        b_x[0,0,-pml_N-2:-1,:] = b_vals.repeat(Ny,1).transpose(0, 1)

        b_y = torch.zeros(Nx, Ny, requires_grad=False).unsqueeze(0).unsqueeze(0)
        b_y[0,0,:,0:pml_N+1]   = torch.flip(b_vals, [0]).repeat(Nx,1)
        b_y[0,0,:,-pml_N-2:-1] = b_vals.repeat(Nx,1)

        self.register_buffer("b", torch.sqrt( b_x**2 + b_y**2 ))

        # Define the finite differencing coeffs for convenience
        self.register_buffer("C1", (self.dt**(-2) + self.b * 0.5 * self.dt**(-1)).pow(-1))
        self.register_buffer("C2", torch.tensor(2 * self.dt**(-2)))
        self.register_buffer("C3", self.dt**(-2) - self.b * 0.5 * self.dt**(-1))

        # Define the laplacian conv kernel
        self.register_buffer("laplacian", h**(-2) * torch.tensor([[[[0.0,  1.0, 0.0],
                                                                    [1.0, -4.0, 1.0],
                                                                    [0.0,  1.0, 0.0]]]],
                                                                    requires_grad=False))

    def step(self, F, un1, un2):
        un = self.C1 * ( self.C2 * un1 - self.C3 * un2 + self.c2 * conv2d(un1, self.laplacian, padding=1) + self.mask_src * F )
        return un, un, un1

    def forward(self, input, hidden, loss_func=None):
        un1, un2 = hidden

        loss = 0.0
        for xi in input.split(1):
            un, un1, un2 = self.step(xi, un1, un2)

            if loss_func is not None:
                loss += loss_func(un)

        return un, loss

    def animate(self, x, block=True):
        fig, ax = plt.subplots()

        un_max = 0.0
        uns = []
        ims = []

        with torch.no_grad():
            un1 = torch.zeros(self.Nx, self.Ny).unsqueeze(0).unsqueeze(0)
            un2 = torch.zeros(self.Nx, self.Ny).unsqueeze(0).unsqueeze(0)
            for xi in x.split(1):
                un, un1, un2 = self.step(xi, un1, un2)
                tmp = un.detach().numpy().squeeze().transpose()
                tmp_max = np.abs(tmp).max()
                if tmp_max > un_max:
                    un_max = tmp_max

                uns.append(tmp)

        for i in range(0, len(uns)):
            im = plt.imshow(uns[i], cmap=plt.cm.RdBu, animated=True, vmin=-un_max, vmax=+un_max)
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True, repeat_delay=1000)

        plt.show(block=block)

        return ani
