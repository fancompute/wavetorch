import torch
from torch.nn.functional import conv2d
import numpy as np

KERNEL_LPF = [[1/9, 1/9, 1/9],
              [1/9, 1/9, 1/9],
              [1/9, 1/9, 1/9]]

def sat_damp(u, uth, b0):
    return b0 / (1 + torch.abs(u/uth).pow(2))

def _laplacian(y, h):
    """Laplacian operator"""
    operator = h**(-2) * torch.tensor([[[[0.0,  1.0, 0.0], [1.0, -4.0, 1.0], [0.0,  1.0, 0.0]]]])
    y = y.unsqueeze(1)
    # y = pad(y,pad=(0,0,1,1), mode='circular')
    # y = pad(y,pad=(1,1,0,0),mode='circular')
    return conv2d(y, operator, padding=1).squeeze(1)

def _time_step(b, c, y1, y2, dt, h):
        y = torch.mul((dt.pow(-2) + b * dt.pow(-1)).pow(-1),
              (2/dt.pow(2)*y1 - torch.mul( (dt.pow(-2) - b * dt.pow(-1)), y2)
                       + torch.mul(c.pow(2), _laplacian(y1, h)))
             )
        return y

class TimeStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b, c, y1, y2, dt, h):
        ctx.save_for_backward(b, c, y1, y2, dt, h)
        return _time_step(b, c, y1, y2, dt, h)

    @staticmethod
    def backward(ctx, grad_output):
        b, c, y1, y2, dt, h = ctx.saved_tensors

        grad_b = grad_c = grad_y1 = grad_y2 = grad_dt = grad_h = None

        if ctx.needs_input_grad[0]:
            grad_b = - (dt * b + 1).pow(-2) * dt * (c.pow(2) * dt.pow(2) * _laplacian(y1, h) + 2*y1 - 2 * y2 ) * grad_output
        if ctx.needs_input_grad[1]:
            grad_c = (b*dt + 1).pow(-1) * (2 * c * dt.pow(2) * _laplacian(y1, h) ) * grad_output
        if ctx.needs_input_grad[2]:
            # grad_y1 = ( dt.pow(2) * _laplacian(c.pow(2) *grad_output, h) + 2*grad_output) * (b*dt + 1).pow(-1)
            c2_grad =  (b*dt + 1).pow(-1) * c.pow(2) * grad_output
            grad_y1 = dt.pow(2) * _laplacian(c2_grad, h) + 2*grad_output * (b*dt + 1).pow(-1)
        if ctx.needs_input_grad[3]:
            grad_y2 = (b*dt -1) * (b*dt + 1).pow(-1) * grad_output

        return grad_b, grad_c, grad_y1, grad_y2, grad_dt, grad_h

class WaveCell(torch.nn.Module):
    """The recurrent neural network cell implementing the scalar wave equation"""

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
                 satdamp_b0 : float = 0.0,
                 satdamp_uth : float = 0.0,
                 c_nl : float = 0.0,
                 sources = [],
                 probes = []):

        super().__init__()

        self.register_buffer('Nx', torch.tensor(Nx))
        self.register_buffer('Ny', torch.tensor(Ny))
        self.register_buffer('h', torch.tensor(h))
        self.register_buffer('dt', torch.tensor(dt))
        self.register_buffer('c0', torch.tensor(c0))
        self.register_buffer('c1', torch.tensor(c1))
        self.register_buffer('eta', torch.tensor(eta))
        self.register_buffer('beta', torch.tensor(beta))

        self.register_buffer('satdamp_b0', torch.tensor(satdamp_b0))
        self.register_buffer('satdamp_uth', torch.tensor(satdamp_uth))
        self.register_buffer('c_nl', torch.tensor(c_nl))

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
        """Initialize the design region"""
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
        """Initialize the distribution of the damping parameter for the PML"""
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
        """Initialize the raw density distribution (which gets trained)"""
        if type(init) == str:
            if init == 'rand':
                raw_rho = torch.round(torch.rand(Nx, Ny))
            elif init == 'half':
                raw_rho = torch.ones(Nx, Ny) * 0.5
            elif init == 'blank':
                raw_rho = torch.zeros(Nx, Ny)
            else:
                raise ValueError('The geometry initialization defined by `init = %s` is invalid' % init)
        elif type(init) == torch.Tensor:
            raw_rho = init
        elif type(init) == np.ndarray:
            raw_rho = torch.from_numpy(init)
        else:
            raise ValueError('The geometry initialization defined by `init` is invalid')

        self.register_parameter('raw_rho', torch.nn.Parameter(raw_rho))

    def clip_to_design_region(self):
        """Clip the density to its background value outside of the design region"""
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
        """The wave speed distribution"""
        return self.c0 + (self.c1-self.c0)*self.rho

    def get_cmax(self):
        """Helper function for getting the maximum wave speed for calculating CFL"""
        return np.max([self.c0, self.c1])

    def add_source(self, source):
        self.sources.append(source)

    def add_probe(self, probe):
        self.probes.append(probe)

    def step(self, x, y1, y2, c_lin, rho):
        """Take a step through time

        Parameters
        ----------
        x : 
            Input value(s) at current time step, batched in first dimension
        y1 : 
            Scalar wave field one time step ago (part of the hidden state)
        y2 : 
            Scalar wave field two time steps ago (part of the hidden state)
        c : 
            Scalar wave speed distribution (this gets passed in to avoid projecting on each time step)
        rho : 
            Projected density (this gets passed in to avoid projecting on each time step)
        """

        dt  = self.dt
        h   = self.h

        if self.satdamp_b0 > 0:
            b = self.b + rho*sat_damp(y1, uth=self.satdamp_uth, b0=self.satdamp_b0)
        else:
            b = self.b

        if self.c_nl != 0:
            c = c_lin + rho * self.c_nl * y1.pow(2)
        else:
            c = c_lin

        y = TimeStep.apply(b, c, y1, y2, dt, h)
        # y = _time_step(b, c, y1, y2, dt, h)

        # Inject all sources
        for source in self.sources:
            source(y, x, dt)

        return y, y, y1

    def forward(self, x):
        """Propagate forward in time for the length of the input

        Parameters
        ----------
        x : 
            Input sequence(s), batched in first dimension
        """

        # hacky way of figuring out if we're on the GPU from inside the model
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"
        
        # First dim is batch
        batch_size = x.shape[0]
        
        # init hidden states
        y1 = torch.zeros(batch_size, self.Nx, self.Ny, device=device)
        y2 = torch.zeros(batch_size, self.Nx, self.Ny, device=device)
        y_all = []

        # Since these will not change with time it's important we pull them
        # outside of the time loop. This dramatically reduces the memory load
        c   = self.c
        rho = self.rho

        # loop through time
        for i, xi in enumerate(x.chunk(x.size(1), dim=1)):
            y, y1, y2 = self.step(xi.squeeze(-1), y1, y2, c, rho)
            y_all.append(y)

        # combine into output field dist 
        y = torch.stack(y_all, dim=1)

        if self.output_probe:
            y = self.measure_probes(y, integrated=True, normalized=True)

        return y

    def measure_probes(self, y, integrated = False, normalized = False):
        """Applies the transformation from the field distribution into the probes"""
        p_out = []
        for probe in self.probes:
            p_out.append(probe(y, integrated=integrated))

        p_out = torch.stack(p_out, dim=-1)
        if normalized and integrated:
            p_out = p_out / torch.sum(p_out, dim=1, keepdim=True)

        return p_out


class WaveCell_Holes(WaveCell):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
