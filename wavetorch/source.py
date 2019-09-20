import skimage
import torch

from .utils import to_tensor


class WaveSource(torch.nn.Module):
	def __init__(self, x, y):
		super().__init__()

		# These need to be longs for advanced indexing to work
		self.register_buffer('x', to_tensor(x, dtype=torch.int64))
		self.register_buffer('y', to_tensor(y, dtype=torch.int64))

	def forward(self, Y, X, dt=1.0):
		# Y[:, self.x, self.y] = Y[:, self.x, self.y] + dt**2 * X.expand_as(Y[:, self.x, self.y])

		# Thanks to Erik Peterson for this fix
		X_expanded = torch.zeros(Y.size()).detach()
		X_expanded[:, self.x, self.y] = X

		return Y + dt ** 2 * X_expanded

	def plot(self, ax, color='r'):
		marker, = ax.plot(self.x.numpy(), self.y.numpy(), 'o', markeredgecolor=color, markerfacecolor='none', markeredgewidth=1.0, markersize=4)
		return marker


class WaveLineSource(WaveSource):
	def __init__(self, r0, c0, r1, c1):
		x, y = skimage.draw.line(r0, c0, r1, c1)

		self.r0 = r0
		self.c0 = c0
		self.r1 = r1
		self.c1 = c1
		super().__init__(x, y)

	def plot(self, ax, color='r'):
		line, = ax.plot([self.r0, self.r1], [self.c0, self.c1], '-', color=color, lw=2)
		return line
