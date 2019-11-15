from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from torch.nn.functional import conv2d
from skimage.draw import circle

from .utils import to_tensor

class WaveGeometry(torch.nn.Module):
	def __init__(self, domain_shape: Tuple, h: float, c0: float, c1: float, abs_N: int = 20, abs_sig: float = 11,
				 abs_p: float = 4.0):
		super().__init__()

		assert len(
			domain_shape) == 2, "len(domain_shape) must be equal to 2: only two-dimensional (2D) domains are supported"

		self.domain_shape = domain_shape
		self.register_buffer("h", to_tensor(h))
		self.register_buffer("c0", to_tensor(c0))
		self.register_buffer("c1", to_tensor(c1))

		self.register_buffer("abs_N", to_tensor(abs_N, dtype=torch.uint8))
		self.register_buffer("abs_sig", to_tensor(abs_sig))
		self.register_buffer("abs_p", to_tensor(abs_p, dtype=torch.uint8))

		self._init_b(abs_N, abs_sig, abs_p)

	def state_reconstruction_args(self):
		return {"domain_shape": self.domain_shape,
				"h": self.h.item(),
				"c0": self.c0.item(),
				"c1": self.c1.item(),
				"abs_N": self.abs_N.item(),
				"abs_sig": self.abs_sig.item(),
				"abs_p": self.abs_p.item()}

	def __repr__(self):
		return "WaveGeometry shape={}, h={}".format(self.domain_shape, self.h)

	def forward(self):
		raise NotImplementedError("WaveGeometry forward() is not implemented. " \
								  "Although WaveGeometry is a subclass of a torch.nn.Module, its forward() method should never be called. " \
								  "It only exists as a torch.nn.Module to hook into pytorch as a component of a WaveCell.")

	@property
	def c(self):
		raise NotImplementedError

	@property
	def b(self):
		return self._b

	@property
	def cmax(self):
		"""Helper function for getting the maximum wave speed for calculating CFL"""
		return np.max([self.c0.item(), self.c1.item()])

	def constrain_to_design_region(self):
		pass

	def _init_b(self, abs_N: int, abs_sig: float, abs_p: float):
		"""Initialize the distribution of the damping parameter for the PML"""

		Nx, Ny = self.domain_shape

		assert Nx > 2 * abs_N + 1, "The domain isn't large enough in the x-direction to fit absorbing layer. Nx = {} and N = {}".format(
			Nx, abs_N)
		assert Ny > 2 * abs_N + 1, "The domain isn't large enough in the y-direction to fit absorbing layer. Ny = {} and N = {}".format(
			Ny, abs_N)

		b_vals = abs_sig * torch.linspace(0.0, 1.0, abs_N + 1) ** abs_p

		b_x = torch.zeros(Nx, Ny)
		b_y = torch.zeros(Nx, Ny)

		if abs_N > 0:
			b_x[0:abs_N + 1, :] = torch.flip(b_vals, [0]).repeat(Ny, 1).transpose(0, 1)
			b_x[(Nx - abs_N - 1):Nx, :] = b_vals.repeat(Ny, 1).transpose(0, 1)

			b_y[:, 0:abs_N + 1] = torch.flip(b_vals, [0]).repeat(Nx, 1)
			b_y[:, (Ny - abs_N - 1):Ny] = b_vals.repeat(Nx, 1)

		self.register_buffer("_b", torch.sqrt(b_x ** 2 + b_y ** 2))


class WaveGeometryHoley(WaveGeometry):
	def __init__(self, domain_shape: Tuple, h: float, c0: float, c1: float, abs_N: int = 20, abs_sig: float = 11,
				 abs_p: float = 4.0, eta: float = 0.5, beta: float = 100.0, x = None, y = None, r = None):

		super().__init__(domain_shape, h, c0, c1, abs_N, abs_sig, abs_p)

		self.x = torch.nn.Parameter(to_tensor(x))
		self.y = torch.nn.Parameter(to_tensor(y))
		self.r = torch.nn.Parameter(to_tensor(r))

		self.register_buffer("eta", to_tensor(eta))
		self.register_buffer("beta", to_tensor(beta))

	def state_reconstruction_args(self):
		my_args = {"eta": self.eta.item(),
				   "beta": self.beta.item(),
				   "x": deepcopy(self.x.detach()),
				   "y": deepcopy(self.y.detach()),
				   "r": deepcopy(self.r.detach())}
		return {**super().state_reconstruction_args(), **my_args}

	def _rho(self):
		eta = self.eta.item()
		beta = self.beta.item()

		xv = torch.arange(0, self.domain_shape[0], dtype=torch.get_default_dtype())
		yv = torch.arange(0, self.domain_shape[1], dtype=torch.get_default_dtype())
		x, y = torch.meshgrid(xv, yv)

		rho = torch.zeros(self.domain_shape)

		for i, (ri, xi, yi) in enumerate(zip(self.r, self.x, self.y)):
			r = torch.sqrt((x-xi).pow(2) + (y-yi).pow(2))
			rho = rho + torch.exp(-r/ri)

		return (np.tanh(beta * eta) + torch.tanh(beta * (rho - eta))) / (
				np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))

	@property
	def rho(self):
		return self._rho()

	@property
	def c(self):
		return self.c0.item() + (self.c1.item() - self.c0.item()) * self._rho()



class WaveGeometryFreeForm(WaveGeometry):
	def __init__(self, domain_shape: Tuple, h: float, c0: float, c1: float, abs_N: int = 20, abs_sig: float = 11,
				 abs_p: float = 4.0, eta: float = 0.5, beta: float = 100.0, design_region=None, rho='half',
				 blur_radius : int = 1, blur_N : int = 1):

		super().__init__(domain_shape, h, c0, c1, abs_N, abs_sig, abs_p)

		self.register_buffer("eta", to_tensor(eta))
		self.register_buffer("beta", to_tensor(beta))

		self._init_design_region(design_region, domain_shape)
		self._init_rho(rho, domain_shape)

		rr, cc = circle(blur_radius, blur_radius, blur_radius+1)
		blur_kernel = torch.zeros((2*blur_radius+1, 2*blur_radius+1), dtype=torch.get_default_dtype())
		blur_kernel[rr, cc] = 1
		blur_kernel=blur_kernel/blur_kernel.sum()

		self.register_buffer("blur_kernel", blur_kernel.unsqueeze(0).unsqueeze(0))
		self.register_buffer("blur_N", to_tensor(blur_N, dtype=torch.int))
		self.register_buffer("blur_radius", to_tensor(blur_N, dtype=torch.int))

		self.constrain_to_design_region()

	def state_reconstruction_args(self):
		my_args = {"eta": self.eta.item(),
				   "beta": self.beta.item(),
				   "design_region": deepcopy(self.design_region),
				   "rho": deepcopy(self.rho.detach()),
				   "blur_radius": self.blur_radius.item(),
				   "blur_N": self.blur_N.item()}
		return {**super().state_reconstruction_args(), **my_args}

	def __repr__(self):
		return super().__repr__() + ", " + str(self.design_region.sum().item()) + " DOFs"

	def _init_design_region(self, design_region, domain_shape):
		if design_region is not None:
			# Use the specified design region
			assert design_region.shape == domain_shape, "The design region shape must match domain shape; design_region.shape = {} domain_shape = {}".format(
				design_region.shape, domain_shape)
			if type(design_region) is np.ndarray:
				design_region = torch.from_numpy(design_region, dtype=torch.unit8)
		else:
			# Just use the whole domain as the design region
			design_region = torch.ones(domain_shape, dtype=torch.uint8)

		self.register_buffer("design_region", design_region)

	def _init_rho(self, rho, domain_shape):
		if isinstance(rho, torch.Tensor) | isinstance(rho, np.ndarray):
			assert rho.shape == domain_shape
			self.rho = torch.nn.Parameter(to_tensor(rho))
		elif isinstance(rho, str):
			if rho == 'rand':
				self.rho = torch.nn.Parameter(torch.round(torch.rand(domain_shape)))
			elif rho == 'half':
				self.rho = torch.nn.Parameter(torch.ones(domain_shape) * 0.5)
			elif rho == 'blank':
				self.rho = torch.nn.Parameter(torch.zeros(domain_shape))
			else:
				raise ValueError('The domain initialization defined by `rho = %s` is invalid' % init)
		else:
			raise ValueError('The domain initialization is invalid')

	def constrain_to_design_region(self):
		"""Clip the wave speed to its background value outside of the design region."""
		with torch.no_grad():
			self.rho[self.design_region == 0] = 0.0
			self.rho[self.b > 0] = 0.0

	def _apply_blur(self, rho):
		"""Applies the blur parameterization operator"""
		blur_N = self.blur_N.item()
		blur_radius = self.blur_radius.item()

		for i in range(blur_N):
			rho = conv2d(rho.unsqueeze(0).unsqueeze(0), self.blur_kernel, padding=self.blur_kernel.shape[-1]//2).squeeze() 

		return rho

	def _apply_projection(self, rho):
		"""Applies the projection parameterization operator"""
		eta = self.eta.item()
		beta = self.beta.item()
		return (np.tanh(beta * eta) + torch.tanh(beta * (rho - eta))) / (
				np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))

	def _rho_model(self):
		"""Runs the complete parameterization model to return rho"""
		rho = self.rho
		rho = self._apply_blur(rho)
		rho = self._apply_projection(rho)
		return rho

	@property
	def c(self):
		return self.c0.item() + (self.c1.item() - self.c0.item()) * self._rho_model()
