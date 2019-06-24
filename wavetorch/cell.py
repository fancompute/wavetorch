import torch
from torch.nn.functional import conv2d
import numpy as np

class WaveCell(torch.nn.Module):
    """The recurrent neural network cell that implements the scalar wave equation."""

    def __init__(self, dt : float, geometry, output_probe = False):
        """Initialize the wave equation recurrent neural network cell.
        """

        super(WaveCell, self).__init__()

        self.dt = dt
        self.geometry = geometry

        self.output_probe = output_probe

        self.register_parameter('rho', geometry.rho)

        cmax = self.geometry.get_cmax()

        if self.geometry.h is None:
            self.geometry.h = dt * cmax * np.sqrt(2) * 1.01

        if self.dt > 1 / cmax * self.geometry.h / np.sqrt(2):
            raise ValueError('The spatial discretization defined by the geometry `h = %f` and the temporal discretization defined by the model `dt = %f` do not satisfy the CFL stability criteria' % (self.geometry.h, self.dt))

    def __repr__(self):
        return "WaveCell \n" + self.geometry.__repr__()

    def _laplacian(self, y):
        h = self.geometry.h
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"
        operator = h**(-2) * torch.tensor([[[[0.0,  1.0, 0.0], [1.0, -4.0, 1.0], [0.0,  1.0, 0.0]]]], device=device)
        return conv2d(y.unsqueeze(1), operator, padding=1).squeeze(1)

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
        c   = self.geometry.c
        rho = self.geometry.rho

        b = self.geometry.boundary_absorber.b if self.geometry.boundary_absorber is not None else torch.zeros(rho.size())

        y = torch.mul((dt**(-2) + b * 0.5 * dt**(-1)).pow(-1),
                      (2/dt**2*y1 - torch.mul( (dt**(-2) - b * 0.5 * dt**(-1)), y2)
                               + torch.mul(c.pow(2), self._laplacian(y1)))
                     )

        # Inject all sources
        for source in self.geometry.sources:
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
        y1 = torch.zeros(batch_size, self.geometry.Nx, self.geometry.Ny, device=device)
        y2 = torch.zeros(batch_size, self.geometry.Nx, self.geometry.Ny, device=device)
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
        for probe in self.geometry.probes:
            p_out.append(probe(y, integrated=integrated))

        p_out = torch.stack(p_out, dim=-1)
        if normalized and integrated:
            p_out = p_out / torch.sum(p_out, dim=1, keepdim=True)

        return p_out
