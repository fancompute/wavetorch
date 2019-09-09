import torch

from .utils import to_tensor

class WaveProbe(torch.nn.Module):
    def __init__(self, x, y):
        super().__init__()

        self.register_buffer('x', to_tensor(x))
        self.register_buffer('y', to_tensor(y))

    def forward(self, x, integrated=False):
        return x[:, :, self.x, self.y]

    # def plot(self, ax, color):
    #     marker, = ax.plot(self.x.numpy(), self.y.numpy(), 'o', markeredgecolor=color, **props.point_properties)
    #     return marker