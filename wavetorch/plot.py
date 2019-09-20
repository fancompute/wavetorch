import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import seaborn as sns

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import torch

from .io import new_geometry
from .geom import WaveGeometry
from .rnn import WaveRNN

point_properties = {'markerfacecolor': 'none',
					'markeredgewidth': 1.0,
					'ms': 3}

bbox_white = {'boxstyle': 'round,pad=0.3',
			  'fc': 'white',
			  'ec': 'none',
			  'alpha': 0.75}

color_dim = {'light': '#cccccc',
			 'dark': '#555555'}

color_txt = {'light': '#000000',
			 'dark': '#ffffff'}

color_highlight = '#a1d99b'


def total_field(model, yb, ylabel, block=False, ax=None, fig_width=4, cbar=True, cax=None, vmin=1e-3, vmax=1.0):
	"""Plot the total (time-integrated) field over the computatonal domain for a given vowel sample
	"""
	with torch.no_grad():
		y_tot = torch.abs(yb).pow(2).sum(dim=1)

		if ax is None:
			fig, ax = plt.subplots(1, 1, constrained_layout=True,
								   figsize=(1.1 * fig_width, model.Ny.item() / model.Nx.item() * fig_width))

		Z = y_tot[0, :, :].numpy().transpose()
		Z = Z / Z.max()
		h = ax.imshow(Z, cmap=plt.cm.magma, origin="bottom", norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
		if cbar:
			if cax is None:
				ax_divider = make_axes_locatable(ax)
				cax = ax_divider.append_axes("top", size="5%", pad="20%")
			if vmax < 1.0:
				extend = 'both'
			else:
				extend = 'min'
			plt.colorbar(h, cax=cax, orientation='horizontal', label=r"$\sum_t{ { u_t{\left(x,y\right)} }^2 }$")
			# cax.set_title(r"$\sum_n \vert u_n \vert^2$")

		geometry(model, ax=ax, outline=True, outline_pml=True, vowel_probe_labels=None, highlight_onehot=ylabel,
				 bg='dark', alpha=0.5)

		ax.set_xticks([])
		ax.set_yticks([])
		# ax.annotate(r"$\sum_n \vert u_n \vert^2$", xy=(0.5, 0.01), fontsize="smaller", ha="center", va="bottom", color="w", xycoords="axes fraction")
		if ax is not None:
			plt.show(block=block)


def _plot_probes(probes, ax, vowel_probe_labels=None, highlight_onehot=None, bg='light'):
	markers = []
	for i, probe in enumerate(probes):
		if highlight_onehot is None:
			color_probe = 'k'
		else:
			color_probe = color_highlight if highlight_onehot[0, i].item() == 1 else color_dim[bg]

		marker = probe.plot(ax, color=color_probe)
		markers.append(marker)

	return markers


def _plot_sources(sources, ax, bg='light'):
	markers = []
	for i, source in enumerate(sources):
		marker = source.plot(ax, color=color_dim[bg])
		markers.append(marker)

	return markers


def geometry(input,
			 ax=None,
			 outline=False,
			 outline_pml=True,
			 vowel_probe_labels=None,
			 highlight_onehot=None,
			 bg='light',
			 alpha=1.0,
			 cbar=False):
	"""Plot the spatial distribution of the wave speed
	"""
	lc = '#000000' if bg == 'light' else '#ffffff'

	if isinstance(input, WaveGeometry):
		geom = input
		probes = None
		source = None
	elif isinstance(input, WaveRNN):
		geom = input.cell.geom
		probes = input.probes
		sources = input.sources
	else:
		raise ValueError("Invalid input for plot.geometry(); should be either a WaveGeometry or a WaveCell")

	rho = geom.rho.detach().numpy().transpose()

	# Make axis if needed
	show = False
	if ax is None:
		show = True
		fig, ax = plt.subplots(1, 1, constrained_layout=True)

	markers = []
	if outline:
		h = ax.contour(rho, levels=[0.5], colors=[lc], linewidths=[0.75], alpha=alpha)
		markers += h.collections
	else:
		limits = np.array([geom.c0, geom.c1])
		cmap = plt.cm.Greens if geom.c0 < geom.c1 else plt.cm.Greens_r
		h = ax.imshow(geom.c.detach().numpy().transpose(), origin="bottom", rasterized=True, cmap=cmap,
					  vmin=limits.min(), vmax=limits.max())

	if cbar and not outline:
		plt.colorbar(h, ax=ax, label='Wave speed', ticks=limits)

	if outline_pml:
		b_boundary = geom.b.numpy().transpose()
		h2 = ax.contour(b_boundary > 0, levels=[0], colors=[lc], linestyles=['dotted'], linewidths=[0.75], alpha=alpha)

	if probes is not None:
		markers += _plot_probes(probes, ax, vowel_probe_labels=vowel_probe_labels, highlight_onehot=highlight_onehot,
								bg=bg)
		markers += h2.collections

	if sources is not None:
		markers += _plot_sources(sources, ax, bg=bg)

	ax.set_xticks([])
	ax.set_yticks([])

	if show:
		plt.show()

	return h, markers


def geometry_evolution(model, model_geom_class_str, history_geom_state, quantity='c', figsize=(5.6, 1.5)):
	"""Plot the spatial distribution of material for the given epochs
	"""
	Nx = int(len(history_geom_state))
	Ny = 1

	fig, axs = plt.subplots(Ny, Nx, constrained_layout=True, figsize=figsize)
	axs = np.asarray(axs)
	axs = axs.ravel()
	for i, state in enumerate(history_geom_state):
		new_geom = new_geometry(model_geom_class_str, state)
		model.cell.geom = new_geom
		h, _ = geometry(model, ax=axs[i], outline=False, outline_pml=True, vowel_probe_labels=None,
						highlight_onehot=None, bg='light', alpha=1.0)
		axs[i].set_title('Epoch %d' % i)

	for j in range(i + 1, len(axs)):
		axs[j].set_xticks([])
		axs[j].set_yticks([])
		axs[j].axis('image')
		axs[j].axis('off')

	plt.colorbar(h, ax=axs, shrink=0.5, label='Wave speed', ticks=np.array([model.c0.item(), model.c1.item()]))


def probe_integrals(model, fields_in, ylabel, x, block=False, ax=None):
	"""Plot the time integrated probe signals
	"""
	# probe_fields = fields_in[0, :, model.px, model.py].numpy()

	# I = np.cumsum(np.abs(probe_fields)**2, axis=0)

	# if ax is None:
	#     fig, axs = plt.subplots(3, 2, constrained_layout=True, figsize=(3.7, 2))

	# for j in range(I.shape[1]):
	#     ax[j,1].plot(I[:,j], "-" if ylabel[0,j].item() == 1 else "--")

	# ax[ylabel[0,:].argmax().item(),0].plot(x.squeeze().numpy(), linewidth=0.75)
	# plt.show(block=block)
	pass


def field_snapshot(model, fields, times, ylabel, fig_width=6, block=False, axs=None, label=True, cbar=True, Ny=1,
				   sat=1.0):
	"""Plot snapshots in time of the scalar wave field
	"""
	field_slices = fields[0, times, :, :]

	if axs is None:
		Nx = int(len(times) / Ny)
		fig, axs = plt.subplots(Ny, Nx, constrained_layout=True)

	axs = np.atleast_1d(axs)
	axs = axs.ravel()

	field_max = field_slices.max().item()

	for i, time in enumerate(times):
		field = field_slices[i, :, :].numpy().transpose()

		h = axs[i].imshow(field, cmap=plt.cm.RdBu, vmin=-sat * field_max, vmax=+sat * field_max, origin="bottom",
						  rasterized=True)
		geometry(model, ax=axs[i], outline=True, outline_pml=True, highlight_onehot=ylabel, bg='light')

		axs[i].set_xticks([])
		axs[i].set_yticks([])

		if label:
			axs[i].text(0.5, 0.03, "time step %d/%d" % (time, fields.shape[1]), transform=axs[i].transAxes, ha="center",
						va="bottom", bbox=bbox_white, fontsize='smaller')

	if cbar:
		plt.colorbar(h, ax=axs, label=r"$u_n{(x,y)}$", shrink=0.80)

	for j in range(i + 1, len(axs)):
		axs[j].set_xticks([])
		axs[j].set_yticks([])
		axs[j].axis('image')
		axs[j].axis('off')

	plt.show(block=block)

	return axs


def animate_fields(model, fields, ylabel, batch=0, block=True, filename=None, interval=30, window_length=None,
				   fig_width=3.5):
	field_max = fields[batch, :, :, :].max().item()

	fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(fig_width, fig_width * model.Ny / model.Nx))
	im = ax.imshow(np.zeros((model.Ny, model.Nx)), cmap=plt.cm.RdBu, animated=True, vmin=-field_max, vmax=+field_max,
				   origin="bottom")

	_, markers = geometry(model, ax=ax, outline=True, outline_pml=True, highlight_onehot=ylabel, bg='light')
	markers = tuple(markers)

	title = ax.text(0.03, 0.03, "", transform=ax.transAxes, ha="left", va="bottom", bbox=bbox_white)

	def animate(i):
		title.set_text("Time step n = %d" % i)
		im.set_array(fields[batch, i, :, :].numpy().transpose())
		return (im, title, *markers)

	frames = None if window_length == None else range(int(fields.shape[1] / 2 - window_length / 2),
													  int(fields.shape[1] / 2 + window_length / 2))
	anim = animation.FuncAnimation(fig, animate, interval=interval, frames=frames, blit=True, repeat_delay=1)

	if filename is not None:
		anim.save(filename, writer='imagemagick')

	plt.show(block=block)


def confusion_matrix(cm, ax=None, figsize=(4, 4), title=None, normalize=False, labels="auto"):
	N_classes = cm.shape[0]

	if normalize:
		cm = 100 * (cm.transpose() / cm.sum(axis=1)).transpose()
		fmt = ".1f"
	else:
		# fmt = "d"
		fmt = ".1f"

	if ax is None:
		fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=figsize)

	mask1 = np.eye(N_classes) == 0
	mask2 = np.eye(N_classes) == 1

	pal1 = sns.blend_palette(["#f7f7f7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061"], as_cmap=True)
	pal2 = sns.blend_palette(["#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"], as_cmap=True)

	sns.heatmap(cm.transpose(),
				fmt=fmt,
				annot=True,
				cmap=pal1,
				linewidths=0,
				cbar=False,
				mask=mask1,
				ax=ax,
				xticklabels=labels,
				yticklabels=labels,
				square=True,
				annot_kws={'size': 'small'})

	sns.heatmap(cm.transpose(),
				fmt=fmt,
				annot=True,
				cmap=pal2,
				linewidths=0,
				cbar=False,
				mask=mask2,
				ax=ax,
				xticklabels=labels,
				yticklabels=labels,
				square=True,
				annot_kws={'size': 'small'})

	for _, spine in ax.spines.items():
		spine.set_visible(True)
	ax.set_xlabel('Input')
	ax.set_ylabel('Predicted')
	ax.axis("image")

	if title is not None:
		ax.set_title(title)
