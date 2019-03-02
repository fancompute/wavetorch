import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import seaborn as sns

import torch

def plot_total_field(model, yb, ylabel, block=False, ax=None, fig_width=4):
    with torch.no_grad():
        y_tot = torch.abs(yb).pow(2).sum(dim=1)

        if ax is None:
            fig, ax= plt.subplots(1, 1, constrained_layout=True, figsize=(1.1*fig_width,model.Ny/model.Nx*fig_width))

        Z = y_tot[0,:,:].numpy().transpose()
        Z = Z / Z.max()
        h = ax.imshow(Z, cmap=plt.cm.inferno,  origin="bottom",  norm=mpl.colors.LogNorm(vmin=1e-3, vmax=1.0))
        ax.contour(model.b.numpy().transpose()>0, levels=[0], colors=("w",), linestyles=("dotted"), alpha=0.75)
        pts_x = np.ones(len(model.probe_y)) * model.probe_x
        pts_y = model.probe_y.numpy()
        for i in range(0,len(pts_x)):
            if ylabel[0,i].item() == 1:
                color = "#98df8a"
            else:
                color = "#7f7f7f"
            ax.plot(pts_x[i], pts_y[i], "o", color=color, mew=0)
        ax.plot(model.src_x, model.src_y, "o", mew=0, color="#7f7f7f")
        plt.colorbar(h, extend='min', ax=ax, aspect=10)
        if ax is not None:
            plt.show(block=block)


def plot_cm(cm, ax=None, figsize=(4,4), title=None, normalize=False, labels="auto"):
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
                fmt=".1f",
                annot=True,
                cmap=pal1,
                linewidths=1,
                cbar=False,
                mask=mask1,
                ax=ax,
                linecolor="#ffffff",
                xticklabels=labels,
                yticklabels=labels)

    sns.heatmap(cm,
                fmt=".1f",
                annot=True,
                cmap=pal2,
                linewidths=1,
                cbar=False,
                mask=mask2,
                ax=ax,
                linecolor="#ffffff",
                xticklabels=labels,
                yticklabels=labels)

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    if title is not None:
        ax.set_title(title)


def plot_c(model, block=False, fig_width=6):
    fig, ax = plt.subplots(1,1,figsize=(1.1*fig_width,model.Ny/model.Nx*fig_width), constrained_layout=True)
    c = model.c().detach()
    h=ax.imshow(c.numpy().transpose(), origin="bottom", rasterized=True, cmap=plt.cm.viridis_r)
    plt.colorbar(h,ax=ax,label="wave speed $c{(x,y)}$")
    ax.contour(model.b.numpy().transpose()>0, levels=[0], colors=("w",), linestyles=("dotted"), alpha=0.75)
    ax.plot(np.ones(len(model.probe_y)) * model.probe_x, model.probe_y.numpy(), "ro")
    ax.plot(model.src_x, model.src_y, "ko")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show(block=block)


def model_animate(model, x, block=True, batch_ind=0, filename=None, interval=1, fps=30, bitrate=768, crop=0.33, fig_width=8):

    with torch.no_grad():
        y = model(x[batch_ind].unsqueeze(0), probe_output=False)
        y_max = torch.max(y).item()

    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(fig_width, fig_width*model.Ny/model.Nx))
    im = ax.imshow(np.zeros((model.Ny, model.Nx)), cmap=plt.cm.RdBu, animated=True, vmin=-y_max, vmax=+y_max, origin="bottom")
    h1, = ax.plot(np.ones(len(model.probe_y)) * model.probe_x, model.probe_y.numpy(), "ks", alpha=0.2)
    h2, = ax.plot(model.src_x, model.src_y, "ko", alpha=0.2)
    title = ax.text(0.05, 0.05, "", transform=ax.transAxes, ha="left", fontsize="large")

    def animate(i):
        title.set_text("Time step n = %d" % i)
        im.set_array(y[0, i, :, :].numpy().transpose())
        return im, title, h1, h2

    anim = animation.FuncAnimation(fig, animate, interval=1, frames=int(crop*y.shape[1])-1, blit=True, repeat_delay=250)

    if filename is not None:
        Writer = animation.writers['ffmpeg']
        anim.save(filename, writer=Writer(fps=fps, bitrate=bitrate))
        plt.close(fig)
    else:
        plt.show(block=block)
