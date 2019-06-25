import torch
from torch.nn.functional import conv2d
import numpy as np

KERNEL_LPF = [[1/9, 1/9, 1/9],
              [1/9, 1/9, 1/9],
              [1/9, 1/9, 1/9]]

class WaveCell(torch.nn.Module):
    """The recurrent neural network cell that implements the scalar wave equation."""

    def __init__(self,
                 Nx : int, 
                 Ny : int, 
                 h : float,
                 dt : float, 
                 output_probe = False,
                 c0 : float = 1.0, 
                 c1 : float = 0.9,
                 eta : float = 0.5, 
                 beta: float = 100.0,
                 N : int = 20,
                 sigma : float = 11,
                 p :float = 4.0,
                 design_region = None, 
                 init : str = 'half',
                 sources = [],
                 probes = []):
        """Initialize the wave equation recurrent neural network cell.
        """

        super(WaveCell, self).__init__()

        self.register_buffer('Nx', torch.tensor(Nx))
        self.register_buffer('Ny', torch.tensor(Ny))
        self.register_buffer('h', torch.tensor(h))
        self.register_buffer('dt', torch.tensor(dt))
        self.register_buffer('c0', torch.tensor(c0))
        self.register_buffer('c1', torch.tensor(c1))
        self.register_buffer('eta', torch.tensor(eta))
        self.register_buffer('beta', torch.tensor(beta))

        self.register_buffer('output_probe', torch.tensor(output_probe))

        self.sources = torch.nn.ModuleList([])
        self.probes  = torch.nn.ModuleList([])

        assert type(sources) == list and type(probes) == list, "`sources` and `probes` must both be lists"

        for source in sources:
            self.sources.append(source)

        for probe in probes:
            self.probes.append(probe)

        self._init_design_region(design_region, Nx, Ny)
        self._init_rho(init, Nx, Ny)
        self._init_b(N, p, sigma)

        self.clip_to_design_region()

        # Validate inputs
        cmax = self.get_cmax()
        if self.dt > 1 / cmax * self.h / np.sqrt(2):
            raise ValueError('The spatial discretization defined by the geometry `h = %f` and the temporal discretization defined by the model `dt = %f` do not satisfy the CFL stability criteria' % (self.h, self.dt))

    def _init_design_region(self, design_region, Nx, Ny):
        if design_region is not None:
            # Use the specified design region
            assert design_region.shape == (Nx, Ny), "Design region mask dims must match spatial dims"
            if type(design_region) is np.ndarray:
                design_region = torch.from_numpy(design_region)
        else:
            # Just use the whole domain as the design region
            design_region = torch.ones(Nx, Ny)

        self.register_buffer('design_region', design_region)

    def _init_b(self, N, p, sigma):
        """Initialize the distribution of the dampening parameter for the PML."""

        Nx = self.Nx
        Ny = self.Ny

        b_vals = sigma * torch.linspace(0.0, 1.0, N+1) ** p

        b_x = torch.zeros(Nx, Ny)
        b_x[0:N+1,   :] = torch.flip(b_vals, [0]).repeat(Ny,1).transpose(0, 1)
        b_x[(Nx-N-1):Nx, :] = b_vals.repeat(Ny,1).transpose(0, 1)

        b_y = torch.zeros(Nx, Ny)
        b_y[:,   0:N+1] = torch.flip(b_vals, [0]).repeat(Nx,1)
        b_y[:, (Ny-N-1):Ny] = b_vals.repeat(Nx,1)

        self.register_buffer('b', torch.sqrt( b_x**2 + b_y**2 ))

    def _init_rho(self, init, Nx, Ny):
        if init == 'rand':
            raw_rho = torch.round(torch.rand(Nx, Ny))
        elif init == 'half':
            raw_rho = torch.ones(Nx, Ny) * 0.5
        elif init == 'blank':
            raw_rho = torch.zeros(Nx, Ny)
        else:
            raise ValueError('The geometry initialization defined by `init = %s` is invalid' % init)

        self.register_parameter('raw_rho', torch.nn.Parameter(raw_rho))

    def clip_to_design_region(self):
        """Clip the density to its background value outside of the design region."""
        with torch.no_grad():
            self.raw_rho[self.design_region==0] = 0.0
            self.raw_rho[self.b>0] = 0.0

    @property
    def rho(self):
        """Perform the projection of the density, rho"""
        eta = self.eta
        beta = self.beta
        LPF_rho = conv2d(self.raw_rho.unsqueeze(0).unsqueeze(0), torch.tensor([[KERNEL_LPF]]), padding=1).squeeze()
        return (torch.tanh(beta*eta) + torch.tanh(beta*(LPF_rho-eta))) / (torch.tanh(beta*eta) + torch.tanh(beta*(1-eta)))

    @property
    def c(self):
        return self.c0 + (self.c1-self.c0)*self.rho

    def get_cmax(self):
        """Helper function for getting the maximum wave speed for calculating CFL"""
        return np.max([self.c0, self.c1])

    def _laplacian(self, y):
        h = self.h
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"
        operator = h**(-2) * torch.tensor([[[[0.0,  1.0, 0.0], [1.0, -4.0, 1.0], [0.0,  1.0, 0.0]]]], device=device)
        return conv2d(y.unsqueeze(1), operator, padding=1).squeeze(1)

    def add_source(self, source):
        self.sources.append(source)

    def add_probe(self, probe):
        self.probes.append(probe)

    def step(self, x, y1, y2):
        """Take a step through time.

        Parameters
        ----------
        x : 
            Input value(s) at current time step, batched in first dimension
        y1 : 
            Scalar wave field one time step ago (part of the hidden state)
        y2 : 
            Scalar wave field two time steps ago (part of the hidden state)
        """

        dt  = self.dt
        c   = self.c
        rho = self.rho
        b   = self.b

        y = torch.mul((dt.pow(-2) + b * 0.5 * dt.pow(-1)).pow(-1),
                      (2/dt.pow(2)*y1 - torch.mul( (dt.pow(-2) - b * 0.5 * dt.pow(-1)), y2)
                               + torch.mul(c.pow(2), self._laplacian(y1)))
                     )

        # Inject all sources
        for source in self.sources:
            source(y, x)

        return y, y, y1

    def forward(self, x):
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
        for i, xi in enumerate(x.chunk(x.size(1), dim=1)):
            y, y1, y2 = self.step(xi.squeeze(-1), y1, y2)
            y_all.append(y)

        # combine into output field dist 
        y = torch.stack(y_all, dim=1)

        if self.output_probe:
            y = self.measure_probes(y, integrated=True, normalized=True)

        return y

    def measure_probes(self, y, integrated = False, normalized = False):
        p_out = []
        for probe in self.probes:
            p_out.append(probe(y, integrated=integrated))

        p_out = torch.stack(p_out, dim=-1)
        if normalized and integrated:
            p_out = p_out / torch.sum(p_out, dim=1, keepdim=True)

        return p_out
