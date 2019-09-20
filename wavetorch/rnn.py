import torch


class WaveRNN(torch.nn.Module):
	def __init__(self, cell, sources, probes=[]):

		super().__init__()

		self.cell = cell

		if type(sources) is list:
			self.sources = torch.nn.ModuleList(sources)
		else:
			self.sources = torch.nn.ModuleList([sources])

		if type(probes) is list:
			self.probes = torch.nn.ModuleList(probes)
		else:
			self.probes = torch.nn.ModuleList([probes])

	def forward(self, x, output_fields=False):
		"""Propagate forward in time for the length of the inputs

		Parameters
		----------
		x :
			Input sequence(s), batched in first dimension
		output_fields :
			Override flag for probe output (to get fields)
		"""

		# Hacky way of figuring out if we're on the GPU from inside the model
		device = "cuda" if next(self.parameters()).is_cuda else "cpu"

		# First dim is batch
		batch_size = x.shape[0]

		# Init hidden states
		hidden_state_shape = (batch_size,) + self.cell.geom.domain_shape
		h1 = torch.zeros(hidden_state_shape, device=device)
		h2 = torch.zeros(hidden_state_shape, device=device)
		y_all = []

		# Because these will not change with time we should pull them out here to avoid unnecessary calculations on each
		# tme step, dramatically reducing the memory load from backpropagation
		c = self.cell.geom.c
		rho = self.cell.geom.rho

		# Loop through time
		for i, xi in enumerate(x.chunk(x.size(1), dim=1)):

			# Propagate the fields
			h1, h2 = self.cell(h1, h2, c, rho)

			# Inject source(s)
			for source in self.sources:
				h1 = source(h1, xi.squeeze(-1))

			if len(self.probes) > 0 and not output_fields:
				# Measure probe(s)
				probe_values = []
				for probe in self.probes:
					probe_values.append(probe(h1))
				y_all.append(torch.stack(probe_values, dim=-1))
			else:
				# No probe, so just return the fields
				y_all.append(h1)

		# Combine outputs into a single tensor
		y = torch.stack(y_all, dim=1)

		return y
