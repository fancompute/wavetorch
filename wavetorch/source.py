import torch
import skimage

import wavetorch.plot.props as props
from .utils import to_tensor

class WaveSource(torch.nn.Module):
    def __init__(self, x, y):
        super().__init__()

        self.register_buffer('x', to_tensor(x))
        self.register_buffer('y', to_tensor(y))

    def forward(self, Y, X, dt=1.0):
        Y[:, self.x, self.y] = Y[:, self.x, self.y] + dt**2 * X.expand_as(Y[:, self.x, self.y])

    def plot(self, ax, color):
        marker, = ax.plot(self.x.numpy(), self.y.numpy(), 'o', markeredgecolor=color, **props.point_properties)
        return marker

# class LineWaveSource(WaveSource):
#     def __init__(self, r0, c0, r1, c1):
#         x, y = skimage.draw.line(r0, c0, r1, c1)
#         super().__init__(x, y)
