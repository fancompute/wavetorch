import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import seaborn as sns
import librosa
import librosa.display

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

import torch

def plot_stft_spectrum(y, n_fft=512, block=False, ax=None, sr=None):
    show=False
    if ax is None:
        fig, ax= plt.subplots(1, 1, constrained_layout=True, figsize=(4,3))
        show = True

    data_stft = np.abs(librosa.stft(y, n_fft=n_fft))
    librosa.display.specshow(librosa.amplitude_to_db(data_stft, ref=np.max),
                             y_axis='linear',
                             x_axis='time',
                             sr=sr, 
                             ax=ax,
                             cmap="cividis")
    # plt.colorbar(h, format='%+2.0f dB', ax=ax)
    if show:
        plt.show(block=block)


def plot_total_field(model, yb, ylabel, block=False, ax=None, fig_width=4, cbar=True, cax=None, vmin=1e-3):
    with torch.no_grad():
        y_tot = torch.abs(yb).pow(2).sum(dim=1)

        if ax is None:
            fig, ax= plt.subplots(1, 1, constrained_layout=True, figsize=(1.1*fig_width,model.Ny/model.Nx*fig_width))

        Z = y_tot[0,:,:].numpy().transpose()
        Z = Z / Z.max()
        h = ax.imshow(Z, cmap=plt.cm.magma,  origin="bottom",  norm=mpl.colors.LogNorm(vmin=vmin, vmax=1.0))
        if cbar:
            if cax is None:
                ax_divider = make_axes_locatable(ax)
                cax = ax_divider.append_axes("top", size="5%", pad="20%")
            plt.colorbar(h, cax=cax, extend='min', orientation='horizontal')
        ax.contour(model.b_boundary.numpy().transpose()>0, levels=[0], colors=("w",), linestyles=("dotted"), alpha=0.75)
        for i in range(0, len(model.px)):
            if ylabel[0,i].item() == 1:
                color = "#98df8a"
            else:
                color = "#7f7f7f"
            ax.plot(model.px[i], model.py[i], "o", markeredgecolor=color, markerfacecolor="none" )
        ax.plot(model.src_x, model.src_y, "o", mew=0, color="#7f7f7f")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.annotate(r"$\sum_n \vert u_n \vert^2$", xy=(0.5, 0.01), fontsize="smaller", ha="center", va="bottom", color="w", xycoords="axes fraction")
        if ax is not None:
            plt.show(block=block)


def plot_confusion_matrix(cm, ax=None, figsize=(4,4), title=None, normalize=False, labels="auto"):
    N_classes = cm.shape[0]

    if normalize:
        cm = 100 * (cm.transpose() / cm.sum(axis=1)).transpose()
        fmt = ".1f"
    else:
        fmt = "d"

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=figsize)

    mask1 = np.eye(N_classes) == 0
    mask2 = np.eye(N_classes) == 1

    pal1 = sns.blend_palette(["#f7f7f7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061"], as_cmap=True)
    pal2 = sns.blend_palette(["#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"], as_cmap=True)

    sns.heatmap(cm,
                fmt=fmt,
                annot=True,
                cmap=pal2,
                linewidths=0,
                cbar=False,
                mask=mask2,
                ax=ax,
                xticklabels=labels,
                yticklabels=labels,
                square=True)

    sns.heatmap(cm,
                fmt=fmt,
                annot=True,
                cmap=pal1,
                linewidths=0,
                cbar=False,
                mask=mask1,
                ax=ax,
                xticklabels=labels,
                yticklabels=labels,
                square=True)

    for _, spine in ax.spines.items():
        spine.set_visible(True)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    if title is not None:
        ax.set_title(title)

def plot_structure_evolution(model, model_states, epochs=[0, 1], quantity='c'):

    fig, axs = plt.subplots(len(epochs), 1, constrained_layout=True)
    for i, epoch in enumerate(epochs):
        model.load_state_dict(model_states[i])
        plot_structure(model, ax=axs[i], quantity='c')


def plot_structure(model, ax=None, quantity='c', vowels=None, cbar=False):
    assert quantity in ['c', 'rho'], "Quantity must be one of `c` or `rho`"

    show = False
    if ax is None:
        show = True
        fig, ax = plt.subplots(1, 1, constrained_layout=True)

    rho = model.proj_rho().detach().numpy().transpose()
    if quantity == 'c':
        Z = model.c0.item() + (model.c1.item()-model.c0.item())*rho
        limits = np.array([model.c0.item(), model.c1.item()])
    else:
        Z = rho
        limits = np.array([0.0, 1.0])

    b_boundary = model.b_boundary.numpy().transpose()

    cmap = plt.cm.Purples_r
    h=ax.imshow(Z, origin="bottom", rasterized=True, cmap=cmap, vmin=limits.min(), vmax=limits.max())

    if cbar:
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="5%", pad="1%")
        # cax.yaxis.set_ticks_position("left")
        plt.colorbar(h, cax=cax, orientation='vertical')
    
    ax.contour(b_boundary>0, levels=[0], colors=("k",), linestyles=("dotted"), alpha=0.75)

    px = model.px.numpy()
    py = model.py.numpy()
    for i in range(0, len(px)):
        ax.plot(px[i], py[i], "ro")
        if vowels is not None:
            if i == 0:
                ax.annotate("source", rotation=90, xy=(model.src_x.numpy(), model.src_y.numpy()), xytext=(-5,0), textcoords="offset points", ha="right", va="center", fontsize="small")
            ax.annotate(vowels[i], xy=(px[i], py[i]), xytext=(5,0), textcoords="offset points", ha="left", va="center", fontsize="small")

    ax.plot(model.src_x.numpy(), model.src_y.numpy(), "ko")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    ax.set_xticks([])
    ax.set_yticks([])


    if show:
        plt.show()

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
