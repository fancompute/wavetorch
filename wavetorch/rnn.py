import torch


class WaveRNN(torch.nn.Module):
    def __init__(self, cell, source, probe):
        self.cell = cell
        self.source = source
        self.probe = probe

    def forward(self, x):
        """Propagate forward in time for the length of the inputs

        Parameters
        ----------
        x :
            Input sequence(s), batched in first dimension
        """

        # Hacky way of figuring out if we're on the GPU from inside the model
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        # First dim is batch
        batch_size = x.shape[0]

        # Init hidden states
        h1 = torch.zeros((batch_size,) + self.WaveCell.domain_shape, device=device)
        h2 = torch.zeros((batch_size,) + self.WaveCell.domain_shape, device=device)
        y_all = []

        # Because these will not change with time we should pull them out here to avoid unnecessary calculations on each
        # tme step, dramatically reducing the memory load from backpropagation
        c = self.WaveCell.c
        rho = self.WaveCell.rho

        # Loop through time
        for i, xi in enumerate(x.chunk(x.size(1), dim=1)):

            # Propagate the fields
            h1, h2 = self.cell(h1, h2, c, rho)

            # Inject source
            h1 = self.source(xi.squeeze(-1), h1)

            if self.probe is not None:
                # Measure probes
                y_all.append(self.WaveProbe(h1))
            else:
                # No probe, so just return the fields
                y_all.append(h1)

        # Combine outputs into a single tensor
        y = torch.stack(y_all, dim=1)

        return y
