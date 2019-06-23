import torch

class BoundaryAbsorber(object):
    def __init__(self, Nx, Ny, N, p, sigma):

        super(BoundaryAbsorber, self).__init__()

        self.N = N
        self.p = p
        self.sigma = sigma

        self.b = self._init_b(Nx, Ny, N, p, sigma)

    @staticmethod
    def _init_b(Nx, Ny, N, p, sigma):
        """Initialize the distribution of the dampening parameter for the PML."""
        b_vals = sigma * torch.linspace(0.0, 1.0, N+1) ** p

        b_x = torch.zeros(Nx, Ny)
        b_x[0:N+1,   :] = torch.flip(b_vals, [0]).repeat(Ny,1).transpose(0, 1)
        b_x[(Nx-N-1):Nx, :] = b_vals.repeat(Ny,1).transpose(0, 1)

        b_y = torch.zeros(Nx, Ny)
        b_y[:,   0:N+1] = torch.flip(b_vals, [0]).repeat(Nx,1)
        b_y[:, (Ny-N-1):Ny] = b_vals.repeat(Nx,1)

        return torch.sqrt( b_x**2 + b_y**2 )
