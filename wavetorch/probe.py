import torch
import numpy as np
import skimage

import wavetorch.plot.props as props

class Probe(object):
    def __init__(self, x, y, label=None):
        super(Probe, self).__init__()
        self.x = x
        self.y = y
        self.label = label

    def __call__(self, x):
        return x[:, :, self.x, self.y]

    def plot(self, ax, color):
        marker, = ax.plot(self.x, self.y, 'o', markeredgecolor=color, **props.point_properties)
        ax.annotate(self.label, xy=(self.x, self.y), xytext=(5,0), textcoords='offset points', ha='left', va='center', fontsize='small', bbox=props.bbox_white, color=props.color_txt['light'])
        return marker


class IntensityProbe(Probe):
    def __init__(self, x, y, label=None):
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        self.label = label

        if x.size != y.size:
            raise ValueError("Length of x and y must be equal")

        self.x = x
        self.y = y

    def __call__(self, x, integrated=False):
        out = torch.abs(x[:, :, self.x, self.y]).pow(2)
        if out.dim() == 4:
            sum_dims = (2,3)
        elif out.dim() == 3:
            sum_dims = 2
        else:
            raise ValueError("Don't know how to integrate")

        out = torch.sum(out, dim=sum_dims)

        if integrated:
            out = torch.sum(out, dim=1)

        return out
