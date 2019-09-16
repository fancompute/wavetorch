from wavetorch.core import load_model
from wavetorch.viz import plot_structure

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

model, history, history_state, cfg = load_model("./study/nonlinear_speed/nonlinear_speed_616.pt")

fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(6.5, 6.5*model.Ny/model.Nx))
im = ax.imshow(np.zeros((model.Ny, model.Nx)), animated=True, origin='bottom')
title = ax.text(0.5, 0.95, "", transform=ax.transAxes, ha="center", va="top", size=20)

def animate(i):
    title.set_text('Epoch %d' % i)
    model.load_state_dict(history_state[i])
    im, _ = plot_structure(model, ax=ax);
    return (im, title)

anim = animation.FuncAnimation(fig, animate, frames=range(0,30), blit=True, repeat_delay=5)

anim.save('./tmp.gif', writer='imagemagick')
