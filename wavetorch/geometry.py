import torch
from torch.nn.functional import conv2d
import numpy as np
from .absorber import BoundaryAbsorber

KERNEL_LPF = [[0,   1/8, 0],
              [1/8, 1/2, 1/8],
              [0,   1/8, 0]]

KERNEL_LPF = [[1/9, 1/9, 1/9],
              [1/9, 1/9, 1/9],
              [1/9, 1/9, 1/9]]

class Geometry(object):
    """
    Defines the geometry to be used by one of the physics cells
    """
    def __init__(self, Nx, Ny, h, c0=1.0, c1=0.9, design_region = None, init : str = 'half', eta : float = 0.5, beta: float = 100.0):

        super(Geometry, self).__init__()

        self.h    = h
        self.Nx   = Nx 
        self.Ny   = Ny
        self.eta  = eta
        self.beta = beta
        self.c0   = c0
        self.c1   = c1

        self.sources = []
        self.probes = []

        self.boundary_absorber = None

        if design_region is not None:
            # Use specified design region
            assert design_region.shape == (Nx, Ny), "Design region mask dims must match spatial dims"
            self.design_region = torch.tensor( design_region )
        else:
            # Use all non-PML area as the design region
            self.design_region = torch.ones(Nx, Ny)

        if init == 'rand':
            self.rho = torch.nn.Parameter( torch.round(torch.rand(Nx, Ny)) )
        elif init == 'half':
            self.rho = torch.nn.Parameter( torch.ones(Nx, Ny) * 0.5 )
        elif init == 'blank':
            self.rho = torch.nn.Parameter( torch.zeros(Nx, Ny) )
        else:
            raise ValueError('The geometry initialization defined by `init = %s` is invalid' % init)

        self.clip_to_design_region()

    def __repr__(self):
        return "Geometry\n   Nx={}, Ny={}, h={}".format(self.Nx, self.Ny, self.h)

    def add_boundary_absorber(self, N=20, p=3.0, sigma=11.1):
        self.boundary_absorber = BoundaryAbsorber(self.Nx, self.Ny, N, p, sigma)

    def add_source(self, source):
        self.sources.append(source)

    def add_probe(self, probe):
        self.probes.append(probe)

    def clip_to_design_region(self):
        """Clip the wave speed to its background value outside of the design region."""
        with torch.no_grad():
            self.rho[self.design_region==0] = 0.0

    def _project_rho(self):
        """Perform the projection of the density, rho"""
        eta = self.eta
        beta = self.beta
        LPF_rho = torch.nn.functional.conv2d(self.rho.unsqueeze(0).unsqueeze(0), torch.tensor([[KERNEL_LPF]]), padding=1).squeeze()
        return (np.tanh(beta*eta) + torch.tanh(beta*(LPF_rho-eta))) / (np.tanh(beta*eta) + np.tanh(beta*(1-eta)))

    @property
    def c(self):
        return self.c0 + (self.c1-self.c0)*self._project_rho()

    def get_cmax(self):
        """Helper function for getting the maximum wave speed for calculating CFL"""
        return np.max([self.c0, self.c1])
