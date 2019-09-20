import torch

from .utils import to_tensor


class WaveProbe(torch.nn.Module):
	def __init__(self, x, y):
		super().__init__()

		# Need to be int64
		self.register_buffer('x', to_tensor(x, dtype=torch.int64))
		self.register_buffer('y', to_tensor(y, dtype=torch.int64))

	def forward(self, x):
		return x[:, self.x, self.y]

	def plot(self, ax, color='k'):
		marker, = ax.plot(self.x.numpy(), self.y.numpy(), 'o', color=color)
		return marker


class WaveIntensityProbe(WaveProbe):
	def __init__(self, x, y):
		super().__init__(x, y)

	def forward(self, x):
		return super().forward(x).pow(2)
