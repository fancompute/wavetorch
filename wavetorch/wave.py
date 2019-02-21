import torch
from torch.nn.functional import conv2d

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

class WaveCell(torch.nn.Module):

    def __init__(self, dt, Nx, Ny, h, src_x, src_y, c2=None, pml_N=20, pml_p=3.0, pml_max=0.5):
        super(WaveCell, self).__init__()

        self.dt = dt

        self.Nx = Nx
        self.Ny = Ny
        self.h  = h

        self.src_x = src_x
        self.src_y = src_y

        if c2 is None:
            c2 = torch.ones(Nx, Ny, requires_grad=True)
        self.c2  = torch.nn.Parameter(c2)

        # Setup the PML/adiabatic absorber
        b_vals = pml_max * torch.linspace(0.0, 1.0, pml_N+1) ** pml_p

        b_x = torch.zeros(Nx, Ny, requires_grad=False)
        b_x[0:pml_N+1,   :] = torch.flip(b_vals, [0]).repeat(Ny,1).transpose(0, 1)
        b_x[-pml_N-2:-1, :] = b_vals.repeat(Ny,1).transpose(0, 1)

        b_y = torch.zeros(Nx, Ny, requires_grad=False)
        b_y[:,   0:pml_N+1] = torch.flip(b_vals, [0]).repeat(Nx,1)
        b_y[:, -pml_N-2:-1] = b_vals.repeat(Nx,1)

        self.register_buffer("b", torch.sqrt( b_x**2 + b_y**2 ))

        # Define the finite differencing coeffs for convenience
        self.register_buffer("A1", (self.dt**(-2) + self.b * 0.5 * self.dt**(-1)).pow(-1))
        self.register_buffer("A2", torch.tensor(2 * self.dt**(-2)))
        self.register_buffer("A3", self.dt**(-2) - self.b * 0.5 * self.dt**(-1))

        # Define the laplacian conv kernel
        self.register_buffer("laplacian", h**(-2) * torch.tensor([[[[0.0,  1.0, 0.0],
                                                                    [1.0, -4.0, 1.0],
                                                                    [0.0,  1.0, 0.0]]]],
                                                                    requires_grad=False))

    def step(self, F, un1, un2):
        un = torch.mul( self.A1, ( torch.mul(self.A2, un1) 
                                   - torch.mul(self.A3, un2) 
                                   + torch.mul( self.c2, conv2d(un1.unsqueeze(1), self.laplacian, padding=1).squeeze(1) ) ))
        un[:, self.src_x, self.src_y] = un[:, self.src_x, self.src_y] + F.squeeze(1)
        return un, un, un1

    def forward(self, x):
        batch_size = x.shape[0]
        un1 = torch.zeros(batch_size, self.Nx, self.Ny)
        un2 = torch.zeros(batch_size, self.Nx, self.Ny)
        un_all = []

        for i, xi in enumerate(x.chunk(x.size(1), dim=1)):
            un, un1, un2 = self.step(xi, un1, un2)
            un_all.append(un)

        un = torch.stack(un_all, dim=1)

        return un

    def animate(self, x, block=True, batch_ind=0):
        fig, ax = plt.subplots()

        ims = []
        with torch.no_grad():
            y = self.forward(x[batch_ind].unsqueeze(0))

        y_max = torch.max(y).item()
        for i in range(0, y.shape[1]):
            im = plt.imshow(y[0,i,:,:].numpy().transpose(), cmap=plt.cm.RdBu, animated=True, vmin=-y_max, vmax=+y_max, origin="bottom")
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True, repeat_delay=1000)
        plt.show(block=block)

        return ani
