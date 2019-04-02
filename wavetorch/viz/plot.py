import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import seaborn as sns
import librosa
import librosa.display

from string import ascii_lowercase
from numpy import in1d

from matplotlib import rcParams
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

import torch

bbox_white = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.75)

def plot_stft_spectrum(y, n_fft=256, block=False, ax=None, sr=None):
    N_classes = probe_series.shape[0]
    for j in range(0, N_classes):
        i = yb.argmax().item()
        ax = axs[i, j]
        input_stft = np.abs(librosa.stft(xb.numpy().squeeze(), n_fft=n_fft))
        output_stft = np.abs(librosa.stft(probe_series[:,j].numpy(), n_fft=n_fft))

        librosa.display.specshow(
            librosa.amplitude_to_db(output_stft,ref=np.max(input_stft)),
            sr=sr,
            vmax=0,
            ax=ax,
            vmin=-50,
            y_axis='linear',
            x_axis='time',
            cmap=plt.cm.inferno
        )
        ax.set_ylim([0,sr/4])

        if i == 0:
            ax.set_title("probe %d" % (j+1), weight="bold")
        if j == N_classes-1:
            ax.text(1.05, 0.5, vowels[i], transform=ax.transAxes, ha="left", va="center", fontsize="large", rotation=-90, weight="bold")
        
        if j > 0:
            ax.set_ylabel('')
        if i < N_classes-1:
            ax.set_xlabel('')
        # if i == j:
            # ax.text(0.5, 0.95, '%s at probe #%d' % (vowels[i], j+1), color="w", transform=ax.transAxes, ha="center", va="top", fontsize="large")


def plot_total_field(model, yb, ylabel, block=False, ax=None, fig_width=4, cbar=True, cax=None, vmin=1e-3, vmax=1.0):
    with torch.no_grad():
        y_tot = torch.abs(yb).pow(2).sum(dim=1)

        if ax is None:
            fig, ax= plt.subplots(1, 1, constrained_layout=True, figsize=(1.1*fig_width,model.Ny/model.Nx*fig_width))

        Z = y_tot[0,:,:].numpy().transpose()
        Z = Z / Z.max()
        h = ax.imshow(Z, cmap=plt.cm.magma,  origin="bottom",  norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
        if cbar:
            if cax is None:
                ax_divider = make_axes_locatable(ax)
                cax = ax_divider.append_axes("top", size="5%", pad="20%")
            if vmax < 1.0:
                extend='both'
            else:
                extend='min'
            plt.colorbar(h, cax=cax, extend=extend, orientation='horizontal', label=r"$\sum_t \vert u_t \vert^2$")
            # cax.set_title(r"$\sum_n \vert u_n \vert^2$")

        plot_structure(model, ax=ax, outline=True, outline_pml=True, vowel_probe_labels=None, highlight_onehot=ylabel, bg='dark', alpha=0.5)

        ax.set_xticks([])
        ax.set_yticks([])
        # ax.annotate(r"$\sum_n \vert u_n \vert^2$", xy=(0.5, 0.01), fontsize="smaller", ha="center", va="bottom", color="w", xycoords="axes fraction")
        if ax is not None:
            plt.show(block=block)


def plot_structure_evolution(model, model_states, epochs=[0, 1], quantity='c', fig_width=5):
    Nx = int(np.ceil(np.sqrt(len(epochs))))
    Ny = int(np.ceil(np.sqrt(len(epochs))))

    Wx = model.Nx.item()
    Wy = model.Ny.item()

    fig_height = fig_width * Nx*Wx/Ny/Wy

    fig, axs = plt.subplots(Ny, Nx, constrained_layout=True, figsize=(fig_width, fig_height))
    axs = axs.ravel()
    for i, epoch in enumerate(epochs):
        model.load_state_dict(model_states[i])
        plot_structure(model, ax=axs[i], quantity='c')
        axs[i].text(0.5, 0.01, 'epoch %d' % epoch, transform=axs[i].transAxes, ha="center", va="bottom", fontsize="small")

    for j in range(i+1,len(axs)):
        axs[j].set_xticks([])
        axs[j].set_yticks([])
        axs[j].axis('image')
        axs[j].axis('off')

def _plot_probes(model, ax, vowel_probe_labels=None, highlight_onehot=None, bg='light'):
    color_dim = '#cccccc' if bg == 'light' else '#555555'
    color_txt = '#000000' if bg == 'light' else '#ffffff'
    color_highlight = '#a1d99b'

    px = model.px.numpy()
    py = model.py.numpy()

    for i in range(0, len(px)):
        if highlight_onehot is None:
            ax.plot(px[i], py[i], "ro", ms=4, mew=0.0)
        else:
            ax.plot(px[i], py[i], "o", markeredgecolor=color_highlight if highlight_onehot[0,i].item() == 1 else color_dim, markerfacecolor="none", markeredgewidth=1.0, ms=4)
        if vowel_probe_labels is not None:
            ax.annotate(vowel_probe_labels[i], xy=(px[i], py[i]), xytext=(5,0), textcoords="offset points", ha="left", va="center", fontsize="small", bbox=bbox_white, color=color_txt)

    if highlight_onehot is None:
        ax.plot(model.src_x.numpy(), model.src_y.numpy(), "ko", ms=4, mew=0.0)
    else:
        ax.plot(model.src_x.numpy(), model.src_y.numpy(), "o", markeredgecolor=color_dim, markerfacecolor="none", markeredgewidth=1.0, ms=4)

    if vowel_probe_labels is not None:
        ax.annotate("source", rotation=90, xy=(model.src_x.numpy(), model.src_y.numpy()), xytext=(-5,0), textcoords="offset points", ha="right", va="center", fontsize="small", bbox=bbox_white, color=color_txt)


def plot_structure(model, ax=None, outline=False, outline_pml=True, vowel_probe_labels=None, highlight_onehot=None, bg='light', alpha=1.0):
    lc = '#000000' if bg == 'light' else '#ffffff'

    rho = model.proj_rho().detach().numpy().transpose()

    # Make axis if needed
    show = False
    if ax is None:
        show = True
        fig, ax = plt.subplots(1, 1, constrained_layout=True)

    if outline:
        h = ax.contour(rho, levels=[0.5], colors=[lc], linewidths=[0.75], alpha=alpha)
    else:
        Z = model.c0.item() + (model.c1.item()-model.c0.item())*rho
        limits = np.array([model.c0.item(), model.c1.item()])
        if model.c0.item() < model.c1.item():
            cmap = plt.cm.Purples
        else:
            cmap = plt.cm.Purples_r
        h = ax.imshow(Z, origin="bottom", rasterized=True, cmap=cmap, vmin=limits.min(), vmax=limits.max())
    
    if outline_pml:
        b_boundary = model.b_boundary.numpy().transpose()
        ax.contour(b_boundary>0, levels=[0], colors=[lc], linestyles=['dotted'], linewidths=[0.75], alpha=alpha)

    _plot_probes(model, ax, vowel_probe_labels=vowel_probe_labels, highlight_onehot=highlight_onehot, bg=bg)

    ax.set_xticks([])
    ax.set_yticks([])

    if show:
        plt.show()

    return h

def plot_probe_integrals(model, fields_in, ylabel, fig_width=6, block=False, ax=None):
    probe_fields = fields_in[0, :, model.px, model.py].numpy()

    I = np.cumsum(np.abs(probe_fields)**2, axis=0)

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(4, 3))

    # ax.plot(I, "-")
    for j in range(I.shape[1]):
        ax.plot(I[:,j], "-", color="#98df8a" if ylabel[0,j].item() == 1 else "#cccccc")

    plt.show(block=block)


def plot_field_snapshot(model, fields_in, times, ylabel, fig_width=6, block=False, axs=None):
    fields = fields_in[0, times, :, :]

    if axs is None:
        # Nx = int(np.ceil(np.sqrt(len(times))))
        # Ny = int(np.ceil(np.sqrt(len(times))))
        Nx = len(times)
        Ny = 1

        Wx = model.Nx.item()
        Wy = model.Ny.item()

        fig_height = fig_width * Ny*Wy/Nx/Wx
        fig, axs = plt.subplots(Ny, Nx, constrained_layout=True, figsize=(fig_width, fig_height))

    axs = axs.ravel()

    field_max = fields.max().item()

    for i, time in enumerate(times):
        field = fields[i, :, :].numpy().transpose()
        
        h = axs[i].imshow(field, cmap=plt.cm.RdBu, vmin=-field_max, vmax=+field_max, origin="bottom", rasterized=True)
        plot_structure(model, ax=axs[i], outline=True, outline_pml=True, highlight_onehot=ylabel, bg='light')

        axs[i].set_xticks([])
        axs[i].set_yticks([])

        axs[i].text(0.5, 0.03, "time step %d/%d" % (time, fields_in.shape[1]), transform=axs[i].transAxes, ha="center", va="bottom", bbox=bbox_white)

    plt.colorbar(h, ax=axs, label=r"$u_n{(x,y)}$", shrink=0.80)

    for j in range(i+1,len(axs)):
        axs[j].set_xticks([])
        axs[j].set_yticks([])
        axs[j].axis('image')
        axs[j].axis('off')

    plt.show(block=block)


def animate_fields(model, field_dist, ylabel, block=True, filename=None, interval=1, fps=30, bitrate=768, crop=0.9, fig_width=6):

    field_max = field_dist.max().item()

    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(fig_width, fig_width*model.Ny/model.Nx))
    im = ax.imshow(np.zeros((model.Ny, model.Nx)), cmap=plt.cm.RdBu, animated=True, vmin=-field_max, vmax=+field_max, origin="bottom")
    
    markers = []
    for i in range(0, len(model.px)):
        if ylabel[0,i].item() == 1:
            color = "#98df8a"
        else:
            color = "#7f7f7f"
        marker, = ax.plot(model.px[i], model.py[i], "o", color=color)
        markers.append(marker)

    marker, =ax.plot(model.src_x, model.src_y, "o", color="#7f7f7f")
    markers.append(marker)
    markers = tuple(markers)

    title = ax.text(0.05, 0.05, "", transform=ax.transAxes, ha="left", fontsize="large")

    def animate(i):
        title.set_text("Time step n = %d" % i)
        im.set_array(field_dist[0, i, :, :].numpy().transpose())
        return (im, title, *markers)

    anim = animation.FuncAnimation(fig, animate, interval=interval, frames=int(crop*field_dist.shape[1])-1, blit=True, repeat_delay=10)

    if filename is not None:
        Writer = animation.writers['ffmpeg']
        anim.save(filename, writer=Writer(fps=fps, bitrate=bitrate))
        plt.close(fig)
    else:
        plt.show(block=block)


def plot_confusion_matrix(cm, ax=None, figsize=(4,4), title=None, normalize=False, labels="auto"):
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

    if title is not None:
        ax.set_title(title)

def apply_sublabels(axs, x=[-50], y=0, size='medium', weight='bold', ha='right', va='top', prefix='', postfix='', invert_color_inds=[], bg=None):
    '''
    Applys panel labels (a, b, c, ... ) in order to the axis handles stored in the list axs
    
    Most of the function arguments should be self-explanatory
    
    invert_color_inds, specifies which labels should use white text, which is useful for darker pcolor plots
    '''

    # assert len(x) == len(axs), "Lengths must match"
    
    if bg is not None:
        bbox_props = dict(boxstyle="round,pad=0.1", fc=bg, ec="none", alpha=0.9)
    else:
        bbox_props = None
    
    # If using latex we need to manually insert the \textbf command
    if rcParams['text.usetex'] and weight == 'bold':
        prefix  = '\\textbf{' + prefix
        postfix = postfix + '}'
    
    for n, ax in enumerate(axs):
        if in1d(n, invert_color_inds):
            color='w'
        else:
            color='k'
        
        ax.annotate(prefix + ascii_lowercase[n] + postfix,
                    xy=(0, 1),
                    xytext=(x, y),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    size=size,
                    color=color,
                    weight=weight,
                    horizontalalignment=ha,
                    verticalalignment=va,
                    bbox=bbox_props)
