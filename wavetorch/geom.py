import torch
from torch.nn.functional import conv2d
import numpy as np
from typing import Tuple

KERNEL_LPF = [[1 / 9, 1 / 9, 1 / 9],
              [1 / 9, 1 / 9, 1 / 9],
              [1 / 9, 1 / 9, 1 / 9]]


class WaveGeometry(object):
    def __init__(self, domain_shape: Tuple, h: float, c0: float, c1: float, abs_N: int = 20, abs_sig: float = 11,
                 abs_p: float = 4.0):
        super().__init__()

        assert len(domain_shape) == 2, "len(shape) must be equal to 2; Only 2D domains are supported"

        self.domain_shape = domain_shape
        self.h = h
        self.c0 = c0
        self.c1 = c1

        self._init_b(abs_N, abs_sig, abs_p)

    def __repr__(self):
        return "Geometry {}, {}".format(self.domain_shape, self.h)

    @property
    def c(self):
        raise NotImplementedError

    @property
    def b(self):
        return self._b

    @property
    def cmax(self):
        """Helper function for getting the maximum wave speed for calculating CFL"""
        return np.max([self.c0, self.c1])

    def parameters(self):
        raise NotImplementedError

    def constrain_to_design_region(self):
        pass

    def _init_b(self, abs_N: int, abs_sig: float, abs_p: float):
        """Initialize the distribution of the damping parameter for the PML"""

        Nx, Ny = self.domain_shape

        assert Nx > 2 * abs_N + 1, "The domain isn't large enough in the x-direction to fit absorbing layer. Nx = {} and N = {}".format(
            Nx, abs_N)
        assert Ny > 2 * abs_N + 1, "The domain isn't large enough in the y-direction to fit absorbing layer. Ny = {} and N = {}".format(
            Ny, abs_N)

        self.abs_N = abs_N
        self.abs_p = abs_p
        self.abs_sig = abs_sig

        b_vals = abs_sig * torch.linspace(0.0, 1.0, abs_N + 1) ** abs_p

        b_x = torch.zeros(Nx, Ny)
        b_y = torch.zeros(Nx, Ny)

        if abs_N > 0:
            b_x[0:N + 1, :] = torch.flip(b_vals, [0]).repeat(Ny, 1).transpose(0, 1)
            b_x[(Nx - N - 1):Nx, :] = b_vals.repeat(Ny, 1).transpose(0, 1)

            b_y[:, 0:N + 1] = torch.flip(b_vals, [0]).repeat(Nx, 1)
            b_y[:, (Ny - N - 1):Ny] = b_vals.repeat(Nx, 1)

        self._b = torch.sqrt(b_x ** 2 + b_y ** 2)


class HoleyWaveGeometry(WaveGeometry):
    def __init__(self, domain_shape: Tuple, h: float, c0: float, c1: float, abs_N: int = 20, abs_sig: float = 11,
                 abs_p: float = 4.0):
        super().__init__(domain_shape, h, c0, c1, abs_N, abs_sig, abs_p)


class ProjectedWaveGeometry(WaveGeometry):
    def __init__(self, domain_shape: Tuple, h: float, c0: float, c1: float, abs_N: int = 20, abs_sig: float = 11,
                 abs_p: float = 4.0, eta: float = 0.5, beta: float = 100.0, design_region=None):

        super().__init__(domain_shape, h, c0, c1, abs_N, abs_sig, abs_p)

        self.eta = eta
        self.beta = beta

        self._init_design_region(design_region, domain_shape)
        self._init_rho(init, domain_shape)
        self.clip_to_design_region()

    def _init_design_region(self, design_region, domain_shape):
        if design_region is not None:
            # Use the specified design region
            assert design_region.shape == domain_shape, "The design region shape must match domain shape; design_region.shape = {} domain_shape = {}".format(
                design_region.shape, domain_shape)
            if type(design_region) is np.ndarray:
                design_region = torch.from_numpy(design_region)
            self.design_region = design_region.type(torch.uint8)
        else:
            # Just use the whole domain as the design region
            self.design_region = torch.ones(domain_shape).type(torch.uint8)

    def _init_rho(self, init, domain_shape):
        if init == 'rand':
            self.rho = torch.nn.Parameter(torch.round(torch.rand(domain_shape)))
        elif init == 'half':
            self.rho = torch.nn.Parameter(torch.ones(domain_shape) * 0.5)
        elif init == 'blank':
            self.rho = torch.nn.Parameter(torch.zeros(domain_shape))
        else:
            raise ValueError('The geometry initialization defined by `init = %s` is invalid' % init)

    def constrain_to_design_region(self):
        """Clip the wave speed to its background value outside of the design region."""
        with torch.no_grad():
            self.rho[self.design_region == 0] = 0.0
            if self.boundary_absorber is not None:
                self.rho[self.boundary_absorber.b > 0] = 0.0

    def _project_rho(self):
        """Perform the projection of the density, rho"""
        eta = self.eta
        beta = self.beta
        LPF_rho = torch.nn.functional.conv2d(self.rho.unsqueeze(0).unsqueeze(0), torch.tensor([[KERNEL_LPF]]),
                                             padding=1).squeeze()
        return (np.tanh(beta * eta) + torch.tanh(beta * (LPF_rho - eta))) / (
                np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))

    @property
    def c(self):
        return self.c0 + (self.c1 - self.c0) * self._project_rho()

# xv = torch.linspace(0.0, 10.0, 99)
# yv = torch.linspace(0.0, 10.0, 99)
# x, y = torch.meshgrid(xv, yv)

# r0 = torch.tensor([1.0], requires_grad=True)
# x0 = torch.tensor([2.0], requires_grad=True)
# y0 = torch.tensor([2.0], requires_grad=True)

# def gen(r0, x0, y0):
#     rho = torch.zeros(x.shape)

#     for i, (ri, xi, yi) in enumerate(zip(r0, x0, y0)):
#         r = torch.sqrt((x-xi).pow(2) + (y-yi).pow(2))
#         rho = rho + torch.exp(-r/ri)

#     return rho

# def proj(rho, eta=torch.tensor(0.5), beta=torch.tensor(200)):
#     return (torch.tanh(beta*eta) + torch.tanh(beta*(rho-eta))) / (torch.tanh(beta*eta) + torch.tanh(beta*(1-eta)))

# rho = gen(r0, x0, y0)
# fig, ax = plt.subplots(2,1, constrained_layout=True)
# ax[0].pcolormesh(x.numpy(), y.numpy(), rho.detach().numpy())
# ax[1].pcolormesh(x.numpy(), y.numpy(), proj(rho).detach().numpy())
# ax[0].axis('image')
# ax[1].axis('image')
