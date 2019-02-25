import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cm(cm, ax=None, figsize=(4,4), title=None, normalize=False):
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
                linecolor="#ffffff")

    sns.heatmap(cm,
                fmt=".1f",
                annot=True,
                cmap=pal2,
                linewidths=1,
                cbar=False,
                mask=mask2,
                ax=ax,
                linecolor="#ffffff")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    if title is not None:
        ax.set_title(title)
