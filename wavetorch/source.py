import torch

from .utils import to_tensor

class WaveSource(torch.nn.Module):
    def __init__(self, x, y):
        super().__init__()

        self.register_buffer('x', to_tensor(x, dtype=torch.int64))
        self.register_buffer('y', to_tensor(y, dtype=torch.int64))

    def forward(self, Y, X, dt=1.0):
        Y[:, self.x, self.y] = Y[:, self.x, self.y] + dt**2 * X.expand_as(Y[:, self.x, self.y])

    def plot(self, ax, color='r'):
        marker, = ax.plot(self.x.numpy(), self.y.numpy(), 'o', color=color)
        return marker

# class LineWaveSource(WaveSource):
#     def __init__(self, r0, c0, r1, c1):
#         x, y = skimage.draw.line(r0, c0, r1, c1)
#         super().__init__(x, y)
