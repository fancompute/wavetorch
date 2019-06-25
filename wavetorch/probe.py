import torch
import numpy as np
import skimage

import wavetorch.plot.props as props
from .utils import to_tensor

class Probe(torch.nn.Module):
    def __init__(self, x, y):
        super(Probe, self).__init__()

        self.register_buffer('x', to_tensor(x))
        self.register_buffer('y', to_tensor(y))

    def forward(self, x):
        return x[:, :, self.x, self.y]

    def plot(self, ax, color):
        marker, = ax.plot(self.x.numpy(), self.y.numpy(), 'o', markeredgecolor=color, **props.point_properties)
        return marker


class IntensityProbe(Probe):
    def __init__(self, x, y):
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        if x.size != y.size:
            raise ValueError("Length of x and y must be equal")

        super(IntensityProbe, self).__init__(x, y)

    def forward(self, x, integrated=False):
        out = torch.abs(x[:, :, self.x, self.y]).pow(2)
        if out.dim() == 4:
            sum_dims = (2,3)
        elif out.dim() == 3:
            sum_dims = 2
        else:
            raise ValueError("Don't know how to integrate")

        out = torch.sum(out, dim=sum_dims)

        if integrated:
            out = torch.sum(out, dim=1)

        return out
