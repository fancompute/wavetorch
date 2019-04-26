from typing import List

import torch
from torch.nn.functional import conv2d
from torch import tanh
import numpy as np

# Low pass filter kernel used for binarization
KERNEL_LPF = [[0,   1/8, 0],
              [1/8, 1/2, 1/8],
              [0,   1/8, 0]]

# Laplacian kernel
KERNEL_LAP = [[0.0,  1.0, 0.0],
              [1.0, -4.0, 1.0],
              [0.0,  1.0, 0.0]]

class WaveCell(torch.nn.Module):
    """The recurrent neural network cell that implements the scalar wave equation."""

    def __init__(self, dt : float, Nx : int, Ny : int,
                 src_x : int, src_y : int, px : List[int], py : List[int], 
                 nl_c : float = 0.0, nl_uth : float = 1.0, nl_b0 : float = 0.0, 
                 eta : float = 0.5, beta: float = 100.0,
                 pml_N : int = 20, pml_p : float = 4.0, pml_max : float = 3.0, 
                 c0 : float = 1.0, c1 : float = 0.9, h : float= None,
                 init : str = 'half', design_region = None):
        """Initialize the wave equation recurrent neural network cell.

        Parameters
        ----------
        dt : float
            The time step size
        Nx : int
            Number of x-direction grid cells in the computational domain
        Ny : int
            Number of y-direction grid cells in the computational domain
        src_x : int
            x-coordinate of the source
        src_y : int
            y-coordinate of the source
        px : list or array
            x-coordinates of the probes
        py : list or array
            y-coordinates of the probes
        nl_c : float
            Nonlinear wave speed parameter
        nl_uth : float
            Nonlinear saturable absorption threshold
        nl_b0 : float
            Nonlinear saturable absorption strength
        eta : float
            Binarize eta parameter
        beta : float
            Binarize beta parameter
        pml_N : int
            PML layer thickness in grid cells
        pml_p : float
            PML polynomial order
        pml_max : float
            PML maximum dampening strength
        c0 : float
            Background wave speed value
        c1 : float
            Optimized (material) wave speed value
        h : float
            Spatial step size
        init_rand : bool
            Toggle
        design_region : array or tensor
            The mask for the design region
        """

        super(WaveCell, self).__init__()

        px = np.atleast_1d(px)
        py = np.atleast_1d(py)
        src_x = np.atleast_1d(src_x)
        src_y = np.atleast_1d(src_y)

        if px.size != py.size:
            raise ValueError("Length of px and py must be the equal")

        if src_x.size != src_y.size:
            raise ValueError("Length of src_x and src_y must be the equal")

        # Time step
        self.register_buffer("dt", torch.tensor(dt))

        # Nonlinearity parameters
        self.register_buffer("nl_uth", torch.tensor(nl_uth))
        self.register_buffer("nl_b0", torch.tensor(nl_b0))
        self.register_buffer("use_satabs_nonlinearity", torch.tensor(False if nl_b0 == 0 else True))

        self.register_buffer("nl_c", torch.tensor(nl_c))
        self.register_buffer("use_speed_nonlinearity", torch.tensor(False if nl_c == 0 else True))

        # Spatial domain dims
        self.register_buffer("Nx", torch.tensor(Nx))
        self.register_buffer("Ny", torch.tensor(Ny))

        # Source coordinates
        self.register_buffer("src_x", torch.tensor(src_x))
        self.register_buffer("src_y", torch.tensor(src_y))

        # Probe coordinates (list)
        self.register_buffer("px", torch.tensor(px))
        self.register_buffer("py", torch.tensor(py))

        # Bounds on wave speed
        self.register_buffer("c0", torch.tensor(c0))
        self.register_buffer("c1", torch.tensor(c1))

        # Binarization parameters
        self.register_buffer("beta", torch.tensor(beta))
        self.register_buffer("eta", torch.tensor(eta))

        # Setup the PML/adiabatic absorber
        self.register_buffer("b_boundary", self.init_b(Nx, Ny, pml_N, pml_p, pml_max))

        if design_region is not None:
            # Use specified design region
            assert design_region.shape == (Nx, Ny), "Design region mask dims must match spatial dims"
            self.register_buffer("design_region", design_region * (self.b_boundary == 0))
        else:
            # Use all non-PML area as the design region
            self.register_buffer("design_region", self.b_boundary == 0)

        if init == 'rand':
            self.rho = torch.nn.Parameter( torch.round(torch.rand(Nx, Ny)) )
        elif init == 'half':
            self.rho = torch.nn.Parameter( torch.ones(Nx, Ny) * 0.5 )
        elif init == 'blank':
            self.rho = torch.nn.Parameter( torch.zeros(Nx, Ny) )
        else:
            raise ValueError('The domain initialization defined by `init = %s` is invalid' % init)

        self.clip_to_design_region()

        cmax = np.max([c0, c1])
        if h is None:
            h = dt * cmax * np.sqrt(2) * 1.01

        if dt > 1 / cmax * h / np.sqrt(2):
            raise ValueError('The discretization defined by `h = %f` and `dt = %f` does not satisfy the CFL stability criteria' % (h, dt))

        self.register_buffer("laplacian", h**(-2) * torch.tensor([[KERNEL_LAP]]))

    def clip_to_design_region(self):
        """Clip the wave speed to its background value outside of the design region."""
        with torch.no_grad():
            self.rho[self.design_region==0] = 0.0

    def proj_rho(self):
        """Perform the projection of the density, rho, to the wave speed, c."""
        eta = self.eta
        beta = self.beta
        LPF_rho = conv2d(self.rho.unsqueeze(0).unsqueeze(0), torch.tensor([[KERNEL_LPF]]), padding=1).squeeze()
        return (tanh(beta*eta) + tanh(beta*(LPF_rho-eta))) / (tanh(beta*eta) + tanh(beta*(1-eta)))

    @staticmethod
    def init_b(Nx, Ny, pml_N, pml_p, pml_max):
        """Initialize the distribution of the dampening parameter for the PML."""
        b_vals = pml_max * torch.linspace(0.0, 1.0, pml_N+1) ** pml_p

        b_x = torch.zeros(Nx, Ny)
        b_x[0:pml_N+1,   :] = torch.flip(b_vals, [0]).repeat(Ny,1).transpose(0, 1)
        b_x[(Nx-pml_N-1):Nx, :] = b_vals.repeat(Ny,1).transpose(0, 1)

        b_y = torch.zeros(Nx, Ny)
        b_y[:,   0:pml_N+1] = torch.flip(b_vals, [0]).repeat(Nx,1)
        b_y[:, (Ny-pml_N-1):Ny] = b_vals.repeat(Nx,1)

        return torch.sqrt( b_x**2 + b_y**2 )

    def step(self, x, y1, y2, c_linear, proj_rho):
        """Take a step through time.

        Parameters
        ----------
        x : 
            Input value(s) at current time step, batched in first dimension
        y1 : 
            Scalar wave field one time step ago (part of the hidden state)
        y2 : 
            Scalar wave field two time steps ago (part of the hidden state)
        c_linear :
            The distribution of the linear wave speed 
            This is passed in to avoid adding extra operations by projecting on each time step
        proj_rho :
            The distribution of the material density
            This is passed in to generate the necessary nonlinearities 
        """
        dt = self.dt

        if self.use_satabs_nonlinearity:
            b = self.b_boundary + proj_rho*sat_damp(y1, uth=self.nl_uth, b0=self.nl_b0)
        else:
            b = self.b_boundary

        if self.use_speed_nonlinearity:
            c = c_linear + proj_rho * self.nl_c * torch.abs(y1).pow(2)
        else:
            c = c_linear

        y = torch.mul((dt**(-2) + b * 0.5 * dt**(-1)).pow(-1),
                      (2/dt**2*y1 - torch.mul( (dt**(-2) - b * 0.5 * dt**(-1)), y2)
                               + torch.mul(c.pow(2), conv2d(y1.unsqueeze(1), self.laplacian, padding=1).squeeze(1)))
                     )
        
        # Insert the source
        y[:, self.src_x, self.src_y] = y[:, self.src_x, self.src_y] + x.expand_as(y[:, self.src_x, self.src_y])
        
        return y, y, y1

    def forward(self, x, probe_output=True):
        """Propagate forward in time for the length of the input.

        Parameters
        ----------
        x : 
            Input sequence(s), batched in first dimension
        probe_output : bool
            Defines whether the output is the probe vector or the entire spatial
            distribution of the scalar wave field in time
        """

        # hacky way of figuring out if we're on the GPU from inside the model
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"
        
        # First dim is batch
        batch_size = x.shape[0]
        
        # init hidden states
        y1 = torch.zeros(batch_size, self.Nx, self.Ny, device=device)
        y2 = torch.zeros(batch_size, self.Nx, self.Ny, device=device)
        y_all = []

        # loop through time
        proj_rho = self.proj_rho()
        c = self.c0 + (self.c1-self.c0)*proj_rho
        for i, xi in enumerate(x.chunk(x.size(1), dim=1)):
            y, y1, y2 = self.step(xi, y1, y2, c, proj_rho)
            y_all.append(y)

        # combine into output field dist 
        y = torch.stack(y_all, dim=1)

        if probe_output:
            # Return only the one-hot output
            return self.integrate_probe_points(self.px, self.py, y)
        else:
            # Return the full field distribution in time
            return y

    @staticmethod
    def integrate_probe_points(px, py, y):
        """Perform the integration of the field at the probe point."""
        I = torch.sum(torch.abs(y[:, :, px, py]).pow(2), dim=1)
        return I / torch.sum(I, dim=1, keepdim=True)

def sat_damp(u, uth=1.0, b0=1.0):
    return b0 / (1 + torch.abs(u/uth).pow(2))

def setup_src_coords(src_x, src_y, Nx, Ny, Npml):
    if (src_x is not None) and (src_y is not None):
        # Coordinate are specified
        return src_x, src_y
    else:
        # Center at left
        return Npml+20, int(Ny/2)

def setup_probe_coords(N_classes, px, py, pd, Nx, Ny, Npml):
    if (py is not None) and (px is not None):
        # All probe coordinate are specified
        assert len(px) == len(py), "Length of px and py must match"

        return px, py

    if (py is None) and (pd is not None):
        # Center the probe array in y
        span = (N_classes-1)*pd
        y0 = int((Ny-span)/2)
        assert y0 > Npml, "Bottom element of array is inside the PML"
        y = [y0 + i*pd for i in range(N_classes)]

        if px is not None:
            assert len(px) == 1, "If py is not specified then px must be of length 1"
            x = [px[0] for i in range(N_classes)]
        else:
            x = [Nx-Npml-20 for i in range(N_classes)]

        return x, y

    raise ValueError("px = {}, py = {}, pd = {} is an invalid probe configuration".format(pd))
