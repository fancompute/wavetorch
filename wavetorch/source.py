import torch
import skimage

import wavetorch.plot.props as props

class Source(object):
    def __init__(self, x, y, label=None):
        super(Source, self).__init__()
        self.x = x
        self.y = y
        self.label = label

    def __call__(self, Y, X):
        Y[:, self.x, self.y] = Y[:, self.x, self.y] + X.expand_as(Y[:, self.x, self.y])

    def plot(self, ax, color):
        marker, = ax.plot(self.x, self.y, 'o', markeredgecolor=color, **props.point_properties)
        return marker

class LineSource(Source):
    def __init__(self, r0, c0, r1, c1, label=None):
        x, y = skimage.draw.line(r0, c0, r1, c1)
        super(LineSource, self).__init__(x, y, label=label)
