import copy
import os

import torch

from . import geom
from .cell import WaveCell
from .probe import WaveIntensityProbe
from .rnn import WaveRNN
from .source import WaveSource


def save_model(model,
			   name,
			   savedir='./study/',
			   history=None,
			   history_geom_state=None,
			   cfg=None,
			   verbose=True):
	"""Save the model state and history to a file
	"""
	str_filename = name + '.pt'
	if not os.path.exists(savedir):
		os.makedirs(savedir)
	str_savepath = savedir + str_filename

	if history_geom_state is None:
		history_geom_state = [model.cell.geom.state_reconstruction_args()]

	data = {'model_geom_class_str': model.cell.geom.__class__.__name__,
			# Class name so we know which constructor to call in load()
			'model_state': model.state_dict(),
			# For now just store model state without history (only geom is likely to change)
			'history': history,
			'history_geom_state': history_geom_state,  # Full history of the geometry state,
			'cfg': cfg}

	if verbose:
		print("Saving model to %s" % str_savepath)
	torch.save(data, str_savepath)


def new_geometry(class_str, state):
	WaveGeometryClass = getattr(geom, class_str)
	geom_state = copy.deepcopy(state)
	return WaveGeometryClass(**geom_state)


def load_model(str_filename, which_iteration=-1):
	"""Load a previously saved model and its history from a file
	"""

	print("Loading model from %s" % str_filename)

	data = torch.load(str_filename)

	# Set the type for floats from the save
	try:
		set_dtype(data['cfg']['dtype'])
	except:
		pass

	# Reconstruct Geometry
	new_geom = new_geometry(data['model_geom_class_str'], data['history_geom_state'][which_iteration])

	# Get model state to recreate probes and sources
	model_state = copy.deepcopy(data['model_state'])

	# Parse out the probe and source coords
	px = [model_state[k].item() for k in model_state if 'probes' in k and 'x' in k]
	py = [model_state[k].item() for k in model_state if 'probes' in k and 'y' in k]
	sx = [model_state[k].item() for k in model_state if 'sources' in k and 'x' in k]
	sy = [model_state[k].item() for k in model_state if 'sources' in k and 'y' in k]

	# Manually add the probes and sources
	new_probes = []
	for (x, y) in zip(px, py):
		new_probes.append(WaveIntensityProbe(x, y))
		# TODO(ian): here we should actually try to infer the type of probe (e.g. intensity or not)

	new_sources = []
	for (x, y) in zip(sx, sy):
		new_sources.append(WaveSource(x, y))

	new_cell = WaveCell(model_state['cell.dt'].item(), new_geom)
	new_model = WaveRNN(new_cell, new_sources, new_probes)
	# Put into eval mode (doesn't really matter for us but whatever)
	new_model.eval()

	return new_model, data['history'], data['history_geom_state'], data['cfg']
