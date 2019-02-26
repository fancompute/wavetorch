import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_total_field(yb):
    with torch.no_grad():
        y_tot = torch.abs(yb).pow(2).sum(dim=1)

    for batch_ind in range(0, y_tot.shape[0]):
        plt.figure(constrained_layout=True)
        Z = y_tot[batch_ind,:,:].numpy().transpose()
        Z = Z / Z.max()
        h = plt.imshow(Z, 
                       cmap=plt.cm.inferno, 
                       origin="bottom", 
                       norm=mpl.colors.LogNorm(vmin=1e-3, vmax=1.0))
        plt.title(r"$\int \vert u{\left(x, y, t\right)} \vert^2\ dt$")
        plt.colorbar(h, extend='min')
        plt.show(block=False)

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
